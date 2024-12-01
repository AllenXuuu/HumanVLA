import os
import numpy as np
import trimesh
from utils import config,utils
import pickle as pkl
import tqdm
from utils import torch_utils
import torch
from scipy.spatial.transform import Rotation as R
from . import retarget_tool as tool

from smplx import joint_names
import smplx

ELBOW_OFFSET = 0.02
HIP_OFFSET = 0.07
KNEE_PLANE_OFFSET_SCALE = 0.2
FOOT_UP_SCALE = 0.5
FOOT_BIGTOE_COEF = 1
def preprocess(actions, joint_names):
    name2index = {n : i for i, n in enumerate(joint_names)}
    actions = actions.permute(1,0,2)
    
    PLVS_IDX = name2index['pelvis']
    NECK_IDX = name2index['neck']
    L_SLD_IDX = name2index['left_shoulder']
    R_SLD_IDX = name2index['right_shoulder']
    L_HIP_IDX = name2index['left_hip']
    R_HIP_IDX = name2index['right_hip']
    L_KNE_IDX = name2index['left_knee']
    R_KNE_IDX = name2index['right_knee']
    L_WST_IDX = name2index['left_wrist']
    R_WST_IDX = name2index['right_wrist']
    L_ANK_IDX = name2index['left_ankle']
    R_ANK_IDX = name2index['right_ankle']
    L_ELB_IDX = name2index['left_elbow']
    R_ELB_IDX = name2index['right_elbow']

    dir_shoulder    = actions[L_SLD_IDX] - actions[R_SLD_IDX]
    dir_hip         = actions[L_HIP_IDX] - actions[R_HIP_IDX]
    left_dir        = torch_utils.normalize(0.5 * (dir_shoulder + dir_hip))
    up_dir      = torch_utils.normalize(actions[NECK_IDX] - actions[PLVS_IDX])
    forward_dir = torch_utils.normalize(torch.cross(left_dir, up_dir))
    up_dir      = torch_utils.normalize(torch.cross(forward_dir, left_dir))


    actions[L_ELB_IDX] += ELBOW_OFFSET * forward_dir
    actions[R_ELB_IDX] += ELBOW_OFFSET * forward_dir

    actions[L_HIP_IDX]  -= HIP_OFFSET * forward_dir
    actions[R_HIP_IDX] -= HIP_OFFSET * forward_dir


    left2right_dir      = torch_utils.normalize(actions[R_HIP_IDX] - actions[L_HIP_IDX])
    left_leg_dir        = torch_utils.normalize(actions[L_ANK_IDX] - actions[L_HIP_IDX])
    left_leg_normal     = torch_utils.normalize(torch.cross(left_leg_dir, torch.cross(left_leg_dir, left2right_dir)))
    left_knee_orth_dist = tool.dot(actions[L_KNE_IDX] - actions[L_HIP_IDX], left_leg_normal)
    actions[L_KNE_IDX]  -= left_knee_orth_dist  * (1-KNEE_PLANE_OFFSET_SCALE) * left_leg_normal
    right_leg_dir       = torch_utils.normalize(actions[R_ANK_IDX] - actions[R_HIP_IDX])
    right_leg_normal    = torch_utils.normalize(torch.cross(right_leg_dir, torch.cross(right_leg_dir, left2right_dir)))
    right_knee_orth_dist= tool.dot(actions[R_KNE_IDX] - actions[R_HIP_IDX], right_leg_normal)
    actions[R_KNE_IDX]  -= right_knee_orth_dist * (1-KNEE_PLANE_OFFSET_SCALE) * right_leg_normal

    actions = actions.permute(1,0,2)
    return actions

def retarget(actions, joint_names):    
    BZ = actions.shape[0]
    device = actions.device
    name2index = {n : i for i, n in enumerate(joint_names)}
    actions = actions.permute(1,0,2)  
    PLVS_IDX = name2index['pelvis']
    NECK_IDX = name2index['neck']
    L_SLD_IDX = name2index['left_shoulder']
    R_SLD_IDX = name2index['right_shoulder']
    L_HIP_IDX = name2index['left_hip']
    R_HIP_IDX = name2index['right_hip']
    L_KNE_IDX = name2index['left_knee']
    R_KNE_IDX = name2index['right_knee']
    L_WST_IDX = name2index['left_wrist']
    R_WST_IDX = name2index['right_wrist']
    L_ANK_IDX = name2index['left_ankle']
    R_ANK_IDX = name2index['right_ankle']
    L_ELB_IDX = name2index['left_elbow']
    R_ELB_IDX = name2index['right_elbow']
    SPIN2_IDX = name2index['spine2']
    HEAD_IDX    = name2index['head']
    L_HEEL_IDX  = name2index['left_heel']
    R_HEEL_IDX  = name2index['right_heel']
    L_BIGTOE_IDX    = name2index['left_big_toe']
    R_BIGTOE_IDX    = name2index['right_big_toe']
    L_SMALLTOE_IDX  = name2index['left_small_toe']
    R_SMALLTOE_IDX  = name2index['right_small_toe']

    dir_shoulder    = actions[L_SLD_IDX] - actions[R_SLD_IDX]
    dir_hip         = actions[L_HIP_IDX] - actions[R_HIP_IDX]
    left_dir        = torch_utils.normalize(0.5 * (dir_shoulder + dir_hip))

    up_dir      = torch_utils.normalize(actions[NECK_IDX] - actions[PLVS_IDX])
    forward_dir = torch_utils.normalize(torch.cross(left_dir, up_dir))
    up_dir      = torch_utils.normalize(torch.cross(forward_dir, left_dir))

    rot_mat = torch.stack([forward_dir, left_dir, up_dir],dim=2)
    
    root_pos = actions[PLVS_IDX]
    root_rot_q = torch_utils.matrix_to_quat(rot_mat)
    Q_root = root_rot_q
    # root_rot_q = root_rot_q * root_rot_q[:,-1:]
    # Q_root = R.from_matrix(rot_mat.cpu().numpy()).as_quat()
    # Q_root = torch.from_numpy(Q_root).to(root_rot_q.device)
    # Q_root = Q_root * Q_root[:,-1:]
    
    dof_pos  = torch.zeros((BZ, 28), device=actions.device, dtype=torch.double)
    tensor100 = torch.tensor([1.,0.,0.],device=device, dtype=torch.double).tile(BZ, 1)
    tensor010 = torch.tensor([0.,1.,0.],device=device, dtype=torch.double).tile(BZ, 1)
    tensor001 = torch.tensor([0.,0.,1.],device=device, dtype=torch.double).tile(BZ, 1)
    
    ############# head
    abdomen_dof_ids     = tool.query_dof(['abdomen_x', 'abdomen_y' , 'abdomen_z'])
    torso2left_src      = torch_utils.normalize(actions[L_SLD_IDX] - actions[SPIN2_IDX])
    torso2right_src     = torch_utils.normalize(actions[R_SLD_IDX] - actions[SPIN2_IDX])
    torso_normal_src    = torch_utils.normalize(torch.cross(torso2left_src,torso2right_src))
    torso_up_src        = torch_utils.normalize(torso2left_src + torso2right_src)
    Q_torso_link, DOFS_abdomen = tool.ik_sphere(
        tensor100,
        tensor001,
        torso_normal_src,
        torso_up_src,
        Q_root
    )
    dof_pos[:, abdomen_dof_ids] = DOFS_abdomen

    neck_dof_ids        = tool.query_dof(['neck_x','neck_y','neck_z'])
    head_src            = torch_utils.normalize(actions[HEAD_IDX] - actions[NECK_IDX])
    head_toleft_normal  = torch_utils.normalize(torch.cross(head_src,forward_dir))
    Q_head_link, DOFS_neck = tool.ik_sphere(
        tensor010,
        tensor001,
        head_toleft_normal,
        head_src,
        Q_torso_link
    )
    dof_pos[:, neck_dof_ids] = DOFS_neck

    ########## left leg
    left_hip_dof_ids    = tool.query_dof(['left_hip_x', 'left_hip_y', 'left_hip_z'])
    left_thigh_src      = torch_utils.normalize(actions[L_KNE_IDX] - actions[L_HIP_IDX])
    left_shin_src       = torch_utils.normalize(actions[L_ANK_IDX] - actions[L_KNE_IDX])
    left_knee_dof_src   = torch_utils.normalize(torch.cross(left_thigh_src, left_shin_src))
    Q_left_thigh_link, DOFS_left_hip = tool.ik_sphere(
        -tensor001,
        tensor010,
        left_thigh_src,
        left_knee_dof_src,
        Q_root
    )
    dof_pos[:, left_hip_dof_ids] = DOFS_left_hip

    left_knee_dof_ids = tool.query_dof('left_knee')
    Q_left_shin_link, DOFS_left_knee = tool.ik_revolute(
        -tensor001,
        left_shin_src,
        Q_left_thigh_link
    )
    dof_pos[:, left_knee_dof_ids] = DOFS_left_knee

    left_ankle_dof_ids      = tool.query_dof(['left_ankle_x', 'left_ankle_y','left_ankle_z'])
    left_foot_bigtoe_src        = torch_utils.normalize(actions[L_BIGTOE_IDX]   - actions[L_HEEL_IDX])
    left_foot_smalltoe_src      = torch_utils.normalize(actions[L_SMALLTOE_IDX] - actions[L_HEEL_IDX])
    left_foot_bigtoe_src[:,2]   *= FOOT_UP_SCALE
    left_foot_smalltoe_src[:,2] *= FOOT_UP_SCALE
    left_foot_up_src       = torch_utils.normalize(torch.cross(left_foot_bigtoe_src, left_foot_smalltoe_src))
    left_foot_forward_src  = torch_utils.normalize(left_foot_bigtoe_src*FOOT_BIGTOE_COEF + left_foot_smalltoe_src*(1-FOOT_BIGTOE_COEF))
    Q_left_foot_link, DOFS_left_ankle = tool.ik_sphere(
        tensor001,
        tensor100,
        left_foot_up_src,
        left_foot_forward_src,
        Q_left_shin_link
    )   
    dof_pos[:, left_ankle_dof_ids] = DOFS_left_ankle
    ########## right leg
    right_hip_dof_ids    = tool.query_dof(['right_hip_x', 'right_hip_y', 'right_hip_z'])
    right_thigh_src      = torch_utils.normalize(actions[R_KNE_IDX] - actions[R_HIP_IDX])
    right_shin_src       = torch_utils.normalize(actions[R_ANK_IDX] - actions[R_KNE_IDX])
    right_knee_dof_src   = torch_utils.normalize(torch.cross(right_thigh_src, right_shin_src))
    Q_right_thigh_link, DOFS_right_hip = tool.ik_sphere(
        -tensor001,
        tensor010,
        right_thigh_src,
        right_knee_dof_src,
        Q_root
    )
    dof_pos[:, right_hip_dof_ids] = DOFS_right_hip

    right_knee_dof_ids = tool.query_dof('right_knee')
    Q_right_shin_link, DOFS_right_knee = tool.ik_revolute(
        -tensor001,
        right_shin_src,
        Q_right_thigh_link
    )
    dof_pos[:, right_knee_dof_ids] = DOFS_right_knee

    right_ankle_dof_ids      = tool.query_dof(['right_ankle_x', 'right_ankle_y','right_ankle_z'])
    right_foot_bigtoe_src        = torch_utils.normalize(actions[R_BIGTOE_IDX]   - actions[R_HEEL_IDX])
    right_foot_smalltoe_src      = torch_utils.normalize(actions[R_SMALLTOE_IDX] - actions[R_HEEL_IDX])
    right_foot_bigtoe_src[:,2]   *= FOOT_UP_SCALE
    right_foot_smalltoe_src[:,2] *= FOOT_UP_SCALE
    right_foot_up_src       = torch_utils.normalize(torch.cross(right_foot_smalltoe_src, right_foot_bigtoe_src))
    right_foot_forward_src  = torch_utils.normalize(right_foot_bigtoe_src*FOOT_BIGTOE_COEF + right_foot_smalltoe_src*(1-FOOT_BIGTOE_COEF))
    Q_right_foot_link, DOFS_right_ankle = tool.ik_sphere(
        tensor001,
        tensor100,
        right_foot_up_src,
        right_foot_forward_src,
        Q_right_shin_link
    )   
    dof_pos[:, right_ankle_dof_ids] = DOFS_right_ankle
    
    ############# left hand
    left_shoulder_dof_ids   = tool.query_dof(['left_shoulder_x','left_shoulder_y','left_shoulder_z'])
    left_upper_arm_src      = torch_utils.normalize(actions[L_ELB_IDX] - actions[L_SLD_IDX])
    left_lower_arm_src      = torch_utils.normalize(actions[L_WST_IDX] - actions[L_ELB_IDX])
    left_elbow_dof_src      = torch_utils.normalize(torch.cross(left_lower_arm_src, left_upper_arm_src))
    Q_left_uparm_link, DOFS_left_shoulder = tool.ik_sphere(
        -tensor001,
        tensor010,
        left_upper_arm_src,
        left_elbow_dof_src,
        Q_torso_link
    )
    dof_pos[:, left_shoulder_dof_ids] = DOFS_left_shoulder

    left_elbow_dof_ids      =  tool.query_dof('left_elbow')
    Q_left_lowarm_link, DOFS_left_elbow = tool.ik_revolute(
        -tensor001,
        left_lower_arm_src,
        Q_left_uparm_link
    )
    dof_pos[:, left_elbow_dof_ids] = DOFS_left_elbow


    ############# right hand
    right_shoulder_dof_ids   = tool.query_dof(['right_shoulder_x','right_shoulder_y','right_shoulder_z'])
    right_upper_arm_src      = torch_utils.normalize(actions[R_ELB_IDX] - actions[R_SLD_IDX])
    right_lower_arm_src      = torch_utils.normalize(actions[R_WST_IDX] - actions[R_ELB_IDX])
    right_elbow_dof_src      = torch_utils.normalize(torch.cross(right_lower_arm_src, right_upper_arm_src))
    Q_right_uparm_link, DOFS_right_shoulder = tool.ik_sphere(
        -tensor001,
        tensor010,
        right_upper_arm_src,
        right_elbow_dof_src,
        Q_torso_link
    )
    dof_pos[:, right_shoulder_dof_ids] = DOFS_right_shoulder

    right_elbow_dof_ids      =  tool.query_dof('right_elbow')
    Q_right_lowarm_link, DOFS_right_elbow = tool.ik_revolute(
        -tensor001,
        right_lower_arm_src,
        Q_right_uparm_link
    )
    dof_pos[:, right_elbow_dof_ids] = DOFS_right_elbow
    return root_pos, Q_root, dof_pos



def main():
    args = tool.parse_args()
    env = tool.RetargetEnv(args)
    utils.set_np_formatting()
    utils.set_seed(0)
    torch.no_grad()
    device = torch.device('cuda:0')
    
    model = smplx.create('/home/allenxyxu/dataset/smplmodels', model_type='smplx', gender='male', use_face_contour=True, num_betas=16, num_expression_coeffs=10, ext='npz', use_pca=False, create_global_orient=False, create_body_pose=False, create_left_hand_pose=False, create_right_hand_pose=False, create_jaw_pose=False, create_leye_pose=False, create_reye_pose=False, create_betas=False, create_expression=False, create_transl=False).double()    
    model = model.to(device)

    data_root = './data/SAMP/pkl' 
    save_root = './data/HITR/HITR_motions'
    motion_files = [
        dict(infile = 'locomotion_random_stageII.pkl',  outfile = 'locomotion_random.pkl',      start = 1200, end = 7500, interval = 4, object_pos = [0. , 0.0, 0.]),

        dict(infile = 'armchair_stageII.pkl',           outfile = 'armchair000.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair001_stageII.pkl',        outfile = 'armchair001.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair002_stageII.pkl',        outfile = 'armchair002.pkl', start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair003_stageII.pkl',        outfile = 'armchair003.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair004_stageII.pkl',        outfile = 'armchair004.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair005_stageII.pkl',        outfile = 'armchair005.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair006_stageII.pkl',        outfile = 'armchair006.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair007_stageII.pkl',        outfile = 'armchair007.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair008_stageII.pkl',        outfile = 'armchair008.pkl', start = 400, end = 700,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair009_stageII.pkl',        outfile = 'armchair009.pkl', start = 400, end = 700,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair010_stageII.pkl',        outfile = 'armchair010.pkl', start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair011_stageII.pkl',        outfile = 'armchair011.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair012_stageII.pkl',        outfile = 'armchair012.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair013_stageII.pkl',        outfile = 'armchair013.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair014_stageII.pkl',        outfile = 'armchair014.pkl', start = 400, end = 1500, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair015_stageII.pkl',        outfile = 'armchair015.pkl', start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair016_stageII.pkl',        outfile = 'armchair016.pkl', start = 400, end = 1500, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair017_stageII.pkl',        outfile = 'armchair017.pkl', start = 400, end = 1600, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair018_stageII.pkl',        outfile = 'armchair018.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'armchair019_stageII.pkl',        outfile = 'armchair019.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),

        dict(infile = 'chair_mo_sit2sit_stageII.pkl',   outfile = 'chair_mo_sit2sit.pkl',       start = 400,  end = 1000, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo_sit2sit001_stageII.pkl',outfile = 'chair_mo_sit2sit001.pkl',    start = 400,  end = 900, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo_stageII.pkl',           outfile = 'chair_mo000.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo001_stageII.pkl',        outfile = 'chair_mo001.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo002_stageII.pkl',        outfile = 'chair_mo002.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo003_stageII.pkl',        outfile = 'chair_mo003.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo004_stageII.pkl',        outfile = 'chair_mo004.pkl', start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo005_stageII.pkl',        outfile = 'chair_mo005.pkl', start = 400, end = 1500, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo006_stageII.pkl',        outfile = 'chair_mo006.pkl', start = 400, end = 1500, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo007_stageII.pkl',        outfile = 'chair_mo007.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo008_stageII.pkl',        outfile = 'chair_mo008.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo009_stageII.pkl',        outfile = 'chair_mo009.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo010_stageII.pkl',        outfile = 'chair_mo010.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo011_stageII.pkl',        outfile = 'chair_mo011.pkl', start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo012_stageII.pkl',        outfile = 'chair_mo012.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo013_stageII.pkl',        outfile = 'chair_mo013.pkl', start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo014_stageII.pkl',        outfile = 'chair_mo014.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo015_stageII.pkl',        outfile = 'chair_mo015.pkl', start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo016_stageII.pkl',        outfile = 'chair_mo016.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo017_stageII.pkl',        outfile = 'chair_mo017.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo018_stageII.pkl',        outfile = 'chair_mo018.pkl', start = 400, end = 1600, interval = 4, object_pos = [0.  ,0.9, 0.]),
        dict(infile = 'chair_mo019_stageII.pkl',        outfile = 'chair_mo019.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.9, 0.]),

        dict(infile = 'highstool_stageII.pkl',           outfile = 'highstool000.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool001_stageII.pkl',        outfile = 'highstool001.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool002_stageII.pkl',        outfile = 'highstool002.pkl', start = 400, end = 1500, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool003_stageII.pkl',        outfile = 'highstool003.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool004_stageII.pkl',        outfile = 'highstool004.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool005_stageII.pkl',        outfile = 'highstool005.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool006_stageII.pkl',        outfile = 'highstool006.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool007_stageII.pkl',        outfile = 'highstool007.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool008_stageII.pkl',        outfile = 'highstool008.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool009_stageII.pkl',        outfile = 'highstool009.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool010_stageII.pkl',        outfile = 'highstool010.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool011_stageII.pkl',        outfile = 'highstool011.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool012_stageII.pkl',        outfile = 'highstool012.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool013_stageII.pkl',        outfile = 'highstool013.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool014_stageII.pkl',        outfile = 'highstool014.pkl', start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool015_stageII.pkl',        outfile = 'highstool015.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool016_stageII.pkl',        outfile = 'highstool016.pkl', start = 400, end = 2000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool017_stageII.pkl',        outfile = 'highstool017.pkl', start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool018_stageII.pkl',        outfile = 'highstool018.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'highstool019_stageII.pkl',        outfile = 'highstool019.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),


        dict(infile = 'reebokstep_stageII.pkl',           outfile = 'reebokstep.pkl',    start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep001_stageII.pkl',        outfile = 'reebokstep001.pkl', start = 400, end = 700,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep002_stageII.pkl',        outfile = 'reebokstep002.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep003_stageII.pkl',        outfile = 'reebokstep003.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep004_stageII.pkl',        outfile = 'reebokstep004.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep005_stageII.pkl',        outfile = 'reebokstep005.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep006_stageII.pkl',        outfile = 'reebokstep006.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep007_stageII.pkl',        outfile = 'reebokstep007.pkl', start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep008_stageII.pkl',        outfile = 'reebokstep008.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep009_stageII.pkl',        outfile = 'reebokstep009.pkl', start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep011_stageII.pkl',        outfile = 'reebokstep011.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep012_stageII.pkl',        outfile = 'reebokstep012.pkl', start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep013_stageII.pkl',        outfile = 'reebokstep013.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep014_stageII.pkl',        outfile = 'reebokstep014.pkl', start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep015_stageII.pkl',        outfile = 'reebokstep015.pkl', start = 400, end = 700,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep016_stageII.pkl',        outfile = 'reebokstep016.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep017_stageII.pkl',        outfile = 'reebokstep017.pkl', start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep018_stageII.pkl',        outfile = 'reebokstep018.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep020_stageII.pkl',        outfile = 'reebokstep020.pkl', start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep021_stageII.pkl',        outfile = 'reebokstep021.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'reebokstep022_stageII.pkl',        outfile = 'reebokstep022.pkl', start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),


        dict(infile = 'sofa_stageII.pkl',             outfile = 'sofa.pkl',     start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa001_stageII.pkl',          outfile = 'sofa001.pkl',  start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa002_stageII.pkl',          outfile = 'sofa002.pkl',  start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa003_stageII.pkl',          outfile = 'sofa003.pkl',  start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa004_stageII.pkl',          outfile = 'sofa004.pkl',  start = 400, end = 1200, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa005_stageII.pkl',          outfile = 'sofa005.pkl',  start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa006_stageII.pkl',          outfile = 'sofa006.pkl',  start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa007_stageII.pkl',          outfile = 'sofa007.pkl',  start = 400, end = 1000, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa008_stageII.pkl',          outfile = 'sofa008.pkl',  start = 400, end = 700,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa009_stageII.pkl',          outfile = 'sofa009.pkl',  start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa010_stageII.pkl',          outfile = 'sofa010.pkl',  start = 400, end = 800,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa011_stageII.pkl',          outfile = 'sofa011.pkl',  start = 400, end = 1300, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa012_stageII.pkl',          outfile = 'sofa012.pkl',  start = 400, end = 1100, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa013_stageII.pkl',          outfile = 'sofa013.pkl',  start = 400, end = 900,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa014_stageII.pkl',          outfile = 'sofa014.pkl',  start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa015_stageII.pkl',          outfile = 'sofa015.pkl',  start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa016_stageII.pkl',          outfile = 'sofa016.pkl',  start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa017_stageII.pkl',          outfile = 'sofa017.pkl',  start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa018_stageII.pkl',          outfile = 'sofa018.pkl',  start = 400, end = 1400, interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'sofa019_stageII.pkl',          outfile = 'sofa019.pkl',  start = 400, end = 1700, interval = 4, object_pos = [0.  ,0.6, 0.]),



        dict(infile = 'table_stageII.pkl',             outfile = 'table.pkl',     start = 400, end = 700,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table001_stageII.pkl',          outfile = 'table001.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table002_stageII.pkl',          outfile = 'table002.pkl',  start = 400, end = 1200,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table003_stageII.pkl',          outfile = 'table003.pkl',  start = 400, end = 900,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table004_stageII.pkl',          outfile = 'table004.pkl',  start = 400, end = 700,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table005_stageII.pkl',          outfile = 'table005.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table006_stageII.pkl',          outfile = 'table006.pkl',  start = 400, end = 700,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table007_stageII.pkl',          outfile = 'table007.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table008_stageII.pkl',          outfile = 'table008.pkl',  start = 400, end = 900,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table009_stageII.pkl',          outfile = 'table009.pkl',  start = 400, end = 1200,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table010_stageII.pkl',          outfile = 'table010.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table011_stageII.pkl',          outfile = 'table011.pkl',  start = 400, end = 1500,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table012_stageII.pkl',          outfile = 'table012.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table013_stageII.pkl',          outfile = 'table013.pkl',  start = 400, end = 600,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table014_stageII.pkl',          outfile = 'table014.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table015_stageII.pkl',          outfile = 'table015.pkl',  start = 400, end = 1200,  interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table016_stageII.pkl',          outfile = 'table016.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table017_stageII.pkl',          outfile = 'table017.pkl',  start = 400, end = 500,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table018_stageII.pkl',          outfile = 'table018.pkl',  start = 400, end = 900,   interval = 4, object_pos = [0.  ,0.6, 0.]),
        dict(infile = 'table019_stageII.pkl',          outfile = 'table019.pkl',  start = 400, end = 800,   interval = 4, object_pos = [0.  ,0.6, 0.]),
    ]


    for motion_file in motion_files:
        motion = pkl.load(open(os.path.join(data_root, motion_file['infile']), 'rb'), encoding='latin1')        
        nf = motion['pose_est_fullposes'].shape[0]
        fullpose = torch.from_numpy(motion['pose_est_fullposes'])
        fulltrans = torch.from_numpy(motion['pose_est_trans'])
        fullbetas = torch.from_numpy(motion['shape_est_betas'])[None, :16].expand(nf, -1)

        start       = motion_file['start']
        end         = motion_file['end']
        interval    = motion_file['interval']
        if end < 0:
            end = nf + end
        end = min(nf, end)

        frame_index = list(range(start, end, interval))
        fps = motion['mocap_framerate'] / interval
        num_frame = len(frame_index)
        time_len = num_frame / fps
        print(f"File: {motion_file['infile']} FPS {fps}. Time {time_len:.3f}s. NumFrame {num_frame}. Range ({start}, {end}, {interval})")
        

        root_pos = []
        root_rot = []
        dof_pos = []
        bz = 256
        for i in range(0,len(frame_index),bz):
            frame_index_batch = frame_index[i : i + bz]
            pose  = fullpose[frame_index_batch].double().to(device)
            trans = fulltrans[frame_index_batch].double().to(device)
            betas = fullbetas[frame_index_batch].double().to(device)
            expression = torch.zeros((pose.shape[0],10)).double().to(device)
            smplx_data = model(betas=betas, transl=trans, global_orient=pose[..., :3], body_pose=pose[..., 3:66], jaw_pose=pose[..., 66:69], leye_pose=pose[..., 69:72], reye_pose=pose[..., 72:75], left_hand_pose=pose[..., 75:120], right_hand_pose=pose[..., 120:165], expression=expression, return_verts=True, return_shaped=False, dense_verts=True)
            action = smplx_data.joints.data

            action = preprocess(action, joint_names.JOINT_NAMES)
            root_pos_batch, root_rot_batch, dof_pos_batch = retarget(action, joint_names.JOINT_NAMES)
            dof_pos_batch = tool.validate_dof(dof_pos_batch)

            root_pos.append(root_pos_batch)
            root_rot.append(root_rot_batch)
            dof_pos.append(dof_pos_batch)

        root_pos = torch.cat(root_pos, dim=0).float()
        root_rot = torch.cat(root_rot, dim=0).float()
        dof_pos = torch.cat(dof_pos, dim=0).float()

        assert root_pos.shape[0] == len(frame_index)
        assert root_rot.shape[0] == len(frame_index)
        assert dof_pos.shape[0] == len(frame_index)
        
        object_pos = motion_file['object_pos']
        object_pos = torch.tensor(object_pos).expand_as(root_pos).to(device)
        object_vel = torch.zeros_like(object_pos)

        root_pos, root_rot, dof_pos, rigid_body_pos, rigid_body_rot = env.run(root_pos, root_rot, dof_pos, object_pos = object_pos)

        rigid_body_vel = tool.compute_linear_vel(rigid_body_pos, fps)
        rigid_body_anv = tool.compute_angular_vel(rigid_body_rot, fps)
        dof_vel  = tool.compute_linear_vel(dof_pos, fps)
        
        
        out = {
            'fps'               : fps,
            'num_frame'         : num_frame,
            'time_length'       : num_frame / fps,
            'dof_pos'           : dof_pos.float().cpu(),
            'dof_vel'           : dof_vel.float().cpu(),
            'rigid_body_pos'    : rigid_body_pos.float().cpu(),
            'rigid_body_rot'    : rigid_body_rot.float().cpu(),
            'rigid_body_vel'    : rigid_body_vel.float().cpu(),
            'rigid_body_anv'    : rigid_body_anv.float().cpu(),
            'object_pos'        : object_pos.float().cpu(),
            'object_vel'        : object_vel.float().cpu()
        }

        if not motion_file['outfile'].startswith('SAMP'):
            motion_file['outfile'] = 'SAMP_' + motion_file['outfile']
        save_path = os.path.join(save_root, motion_file['outfile'])
        print(f'Save to ==> {save_path}')
        with open(save_path, 'wb') as f:
            pkl.dump(out, f)

                

if __name__ == '__main__':
    main()
