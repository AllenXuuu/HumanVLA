import os
import numpy as np
import trimesh
from utils import config,utils
import pickle as pkl
import tqdm
from utils import torch_utils
from scipy.spatial.transform import Rotation
import torch
import joblib
from human_body_prior.body_model.body_model import BodyModel
import copy
from scipy.spatial.transform import Rotation as R
from . import retarget_tool as tool
import trimesh


HIP_OFFSET = 0.
KNEE_PLANE_OFFSET_SCALE = 0.2
ELBOW_OFFSET_SCALE = 0.9
FOOT_UP_SCALE = 0.3
NUM_PCD = 200

def preprocess(actions, joint_names):
    name2index = {n : i for i, n in enumerate(joint_names)}
    actions = actions.permute(1,0,2)
    
    PLVS_IDX = name2index['Pelvis']
    NECK_IDX = name2index['Neck']
    L_SLD_IDX = name2index['L_Shoulder']
    R_SLD_IDX = name2index['R_Shoulder']
    L_HIP_IDX = name2index['L_Hip']
    R_HIP_IDX = name2index['R_Hip']
    L_ANK_IDX = name2index['L_Ankle']
    R_ANK_IDX = name2index['R_Ankle']
    L_KNE_IDX = name2index['L_Knee']
    R_KNE_IDX = name2index['R_Knee']
    L_WST_IDX = name2index['L_Wrist']
    R_WST_IDX = name2index['R_Wrist']
    L_ELB_IDX = name2index['L_Elbow']
    R_ELB_IDX = name2index['R_Elbow']

    dir_shoulder    = actions[L_SLD_IDX] - actions[R_SLD_IDX]
    dir_hip         = actions[L_HIP_IDX] - actions[R_HIP_IDX]
    left_dir        = torch_utils.normalize(0.5 * (dir_shoulder + dir_hip))

    up_dir      = torch_utils.normalize(actions[NECK_IDX] - actions[PLVS_IDX])
    forward_dir = torch_utils.normalize(torch.cross(left_dir, up_dir))
    up_dir      = torch_utils.normalize(torch.cross(forward_dir, left_dir))

    actions[L_HIP_IDX] -= HIP_OFFSET * forward_dir
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


    left_arm            = torch_utils.normalize(actions[L_WST_IDX] - actions[L_SLD_IDX])
    left_elbow_orth_arm = actions[L_ELB_IDX] - actions[L_WST_IDX]
    left_elbow_orth_arm = left_elbow_orth_arm - tool.dot(left_elbow_orth_arm, left_arm) * left_arm
    actions[L_ELB_IDX] -= left_elbow_orth_arm * (1 - ELBOW_OFFSET_SCALE)
    right_arm           = torch_utils.normalize(actions[R_WST_IDX] - actions[R_SLD_IDX])
    right_elbow_orth_arm = actions[R_ELB_IDX] - actions[R_WST_IDX]
    right_elbow_orth_arm = right_elbow_orth_arm - tool.dot(right_elbow_orth_arm, right_arm) * right_arm
    actions[R_ELB_IDX] -= right_elbow_orth_arm * (1 - ELBOW_OFFSET_SCALE)


    actions = actions.permute(1,0,2)
    return actions

def retarget(actions, joint_names):
    BZ = actions.shape[0]
    device = actions.device
    name2index = {n : i for i, n in enumerate(joint_names)}
    actions = actions.permute(1,0,2)  
    PLVS_IDX = name2index['Pelvis']
    NECK_IDX = name2index['Neck']
    L_SLD_IDX = name2index['L_Shoulder']
    R_SLD_IDX = name2index['R_Shoulder']
    L_HIP_IDX = name2index['L_Hip']
    R_HIP_IDX = name2index['R_Hip']
    L_ANK_IDX = name2index['L_Ankle']
    R_ANK_IDX = name2index['R_Ankle']
    L_FOT_IDX = name2index['L_Foot']
    R_FOT_IDX = name2index['R_Foot']
    L_KNE_IDX = name2index['L_Knee']
    R_KNE_IDX = name2index['R_Knee']
    L_WST_IDX = name2index['L_Wrist']
    R_WST_IDX = name2index['R_Wrist']
    L_ELB_IDX = name2index['L_Elbow']
    R_ELB_IDX = name2index['R_Elbow']  
    SPIN2_IDX = name2index['Spine2']
    HEAD_IDX = name2index['Head']

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
    left_foot_forward_src   = actions[L_FOT_IDX] - actions[L_ANK_IDX]
    left_foot_forward_src[:,2] *= FOOT_UP_SCALE
    left_foot_forward_src   = torch_utils.normalize(left_foot_forward_src)
    left_foot_up_src        = tensor001
    left_foot_side_src      = torch_utils.normalize(torch.cross(left_foot_up_src, left_foot_forward_src))       
    Q_left_foot_link, DOFS_left_ankle = tool.ik_sphere(
        tensor010,
        tensor100,
        left_foot_side_src,
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
    right_foot_forward_src   = actions[R_FOT_IDX] - actions[R_ANK_IDX]
    right_foot_forward_src[:,2] *= FOOT_UP_SCALE
    right_foot_forward_src   = torch_utils.normalize(right_foot_forward_src)
    right_foot_up_src        = tensor001
    right_foot_side_src      = torch_utils.normalize(torch.cross(right_foot_up_src, right_foot_forward_src))       
    Q_right_foot_link, DOFS_right_ankle = tool.ik_sphere(
        tensor010,
        tensor100,
        right_foot_side_src,
        right_foot_forward_src,
        Q_right_shin_link,
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
    dtype = torch.double
    male_model   = BodyModel(bm_fname='/home/allenxyxu/dataset/smplmodels/smplx/SMPLX_MALE.npz',   num_betas=16, num_expressions=None, num_dmpls=None, dmpl_fname=None).type(dtype).to(device)
    female_model = BodyModel(bm_fname='/home/allenxyxu/dataset/smplmodels/smplx/SMPLX_FEMALE.npz', num_betas=16, num_expressions=None, num_dmpls=None, dmpl_fname=None).type(dtype).to(device)
    joint_names = np.load('/home/allenxyxu/dataset/smplmodels/smplx/SMPLX_MALE.npz',encoding='latin1',allow_pickle=True)['joint2num'].item()
    joint_names = sorted([(v,k) for k,v in joint_names.items()])
    joint_names = [a[1] for a in joint_names]

    data_root = './data/omomo'
    save_root = './data/HITR/HITR_motions'
    train_motions = joblib.load(os.path.join(data_root,'./train_diffusion_manip_seq_joints24.p'))
    test_motions  = joblib.load(os.path.join(data_root,'./test_diffusion_manip_seq_joints24.p'))
    full_motions  = {}
    for k,v in train_motions.items():
        full_motions[v['seq_name']] = v
    for k,v in test_motions.items():
        full_motions[v['seq_name']] = v

    object_root = './data/omomo/captured_objects'
    object_pcds = {}
    for fn in os.listdir(object_root):
        objname = fn.replace('.obj', '').replace('_cleaned_simplified','')
        mesh = trimesh.load(os.path.join(object_root, fn))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh_sampled_pcd = mesh.sample(NUM_PCD)
        object_pcds[objname] = mesh_sampled_pcd
    
    objset = set() 
    motion_keys = sorted(full_motions.keys())
    keep_motion_keys = []
    for key in motion_keys:
        subid, objname, index = key.split('_')
        objset.add(objname)
        if objname in ['whitechair', 'suitcase' , 'woodchair', 'largebox', 'plasticbox', 'largetable', 'trashcan']:
            keep_motion_keys.append(key)
    for i, seq_name in enumerate(tqdm.tqdm(keep_motion_keys,ncols=100)):
        motion = full_motions[seq_name]
        _, objname, _ = seq_name.split('_')
        gender = motion['gender'].item()
        body_model = {
            'male' : male_model,
            'female' : female_model,
        }[gender]

        nf = motion['pose_body'].shape[0]
        start   = 0
        end     = nf
        interval = 1
        
        frame = list(range(start, end, interval))
        
        pose_body   = torch.from_numpy(motion['pose_body'][frame]).to(device).type(dtype)
        betas       = torch.from_numpy(motion['betas']).to(device).type(dtype)
        betas       = torch.tile(betas,(len(frame), 1))
        root_orient = torch.from_numpy(motion['root_orient'][frame]).to(device).type(dtype)
        trans       = torch.from_numpy(motion['trans'][frame]).to(device).type(dtype)
        pred_body   = body_model(pose_body=pose_body, pose_hand=None, betas=betas, root_orient=root_orient, trans=trans)
        joint_poses = pred_body.Jtr    


        ############ robot
        action = joint_poses
        action = preprocess(action, joint_names)
        root_pos, root_rot, dof_pos = retarget(action, joint_names)
        dof_pos = tool.validate_dof(dof_pos)

        ############ object
        object_scale = motion['obj_scale']
        object_trans = motion['obj_trans'][..., 0]
        object_rot   = motion['obj_rot']
        object_rot = Rotation.from_matrix(object_rot).as_quat()
        object_pcd = object_pcds[objname]
        assert object_scale.shape == (nf,)
        assert object_trans.shape == (nf,3)
        assert object_rot.shape == (nf,4)
        assert object_pcd.shape == (NUM_PCD,3)

        object_scale = torch.from_numpy(object_scale).to(device).type(dtype)
        object_trans = torch.from_numpy(object_trans).to(device).type(dtype)
        object_rot = torch.from_numpy(object_rot).to(device).type(dtype)
        object_pcd = torch.from_numpy(object_pcd).to(device).type(dtype)

        global_pcd = torch_utils.transform_pcd(
            object_pcd.unsqueeze(0) * object_scale.unsqueeze(-1).unsqueeze(-1),
            object_trans,
            object_rot
        )
        object_bottom_center = torch.stack([
            global_pcd[:,:,0].mean(-1),
            global_pcd[:,:,1].mean(-1),
            global_pcd[:,:,2].min(-1)[0]
            ], dim=-1)

        root_pos, root_rot, dof_pos, rigid_body_pos, rigid_body_rot = env.run(root_pos, root_rot, dof_pos)

        fps = 30
        rigid_body_vel  = tool.compute_linear_vel(rigid_body_pos, fps)
        rigid_body_anv  = tool.compute_angular_vel(rigid_body_rot, fps)
        dof_vel         = tool.compute_linear_vel(dof_pos, fps)
        object_vel      = tool.compute_linear_vel(object_bottom_center, fps)

         
        out = {
            'fps'               : fps,
            'num_frame'         : nf,
            'time_length'       : nf / fps,
            'dof_pos'           : dof_pos.float().cpu(),
            'dof_vel'           : dof_vel.float().cpu(),
            'rigid_body_pos'    : rigid_body_pos.float().cpu(),
            'rigid_body_rot'    : rigid_body_rot.float().cpu(),
            'rigid_body_vel'    : rigid_body_vel.float().cpu(),
            'rigid_body_anv'    : rigid_body_anv.float().cpu(),
            'object_pos'        : object_bottom_center.float().cpu(),
            'object_vel'        : object_vel.float().cpu()
        }

        save_path = os.path.join(save_root, 'OMOMO_' + seq_name + '.pkl')
        # print(f'Save to ==> {save_path}')
        with open(save_path, 'wb') as f:
            pkl.dump(out, f)


if __name__ == '__main__':
    main()
