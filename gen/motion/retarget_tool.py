import torch
import numpy as np
from utils import torch_utils
from envs.humanoid import HumanoidEnv
from easydict import EasyDict
from isaacgym import gymapi,gymtorch,gymutil
import argparse
import time

class RetargetEnv(HumanoidEnv):
    def __init__(self, args) -> None:
        self.args = args
        headless = args.headless
        use_gpu = args.gpu
        cfg = EasyDict(
            sim_device = 'cuda:0',
            sim_device_type = 'cuda',
            compute_device_id = 0,
            graphics_device_id = -1 if headless else 0,
            num_envs = args.num_envs,
            num_action = 1,
            data_prefix = '.',
            actor = EasyDict(asset_root='./data/HITR/humanoid_assets', asset_file='amp_humanoid_extend.xml', fix_base_link = False),
            camera = EasyDict(pos=[-3, -3, 3], tgt=[0, 0.5, 1]),
            spacing = 5,
            control_freq_inv = 2,
            plane = EasyDict(staticFriction=1,dynamicFriction=1,restitution=0),
            physics_engine = gymapi.SIM_PHYSX,
            sim = EasyDict(
                up_axis = 'z',
                dt = 0.016666,
                substeps = 2,
                num_client_threads = 0,
                use_gpu_pipeline  = use_gpu,
                physx = EasyDict(
                    use_gpu = use_gpu,
                    num_threads = 4,
                    solver_type = 1,  # 0: pgs, 1: tgs
                    num_position_iterations = 4,
                    num_velocity_iterations = 0,
                    contact_offset = 0.02,
                    rest_offset = 0.0,
                    bounce_threshold_velocity = 0.2,
                    max_depenetration_velocity = 10.0,
                    default_buffer_size_multiplier = 10.0,
                    num_subscenes = 0
                )
            )
        )
        super().__init__(cfg)

        self._robot_root_states = self._root_states.view(self.num_envs, 13)
        self._robot_dof_states = self._dof_state.view(self.num_envs, self.num_dof,  2) 
        self._robot_rigid_body_states = self._rigid_body_state.view(self.num_envs, self.num_body, 13)
       
        
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 0))
        self.extra = 0.03

    def draw_point(self,p):
        pose = gymapi.Transform()   
        pose.p = gymapi.Vec3(*p)
        gymutil.draw_lines(self.sphere_geom,self.gym,self.viewer,self.envs[0],pose)

    def forward_kinematics(self, root_pos, root_rot, dof_pos, object_pos=None, fast = None, offset = [0, 0, 1]):
        if fast is None:
            fast = self.args.headless
        orig_device = root_pos.device
        
        offset = torch.tensor(offset,device=root_pos.device).float()
        root_pos = root_pos + offset
        root_pos = root_pos.float().to(self.device)
        root_rot = root_rot.float().to(self.device)
        dof_pos = dof_pos.float().to(self.device)

        rigid_body_pos = []
        rigid_body_rot = []

        interval = self.num_envs if fast else 1
        for i in range(0, root_pos.shape[0], interval):
            # if fast:
            bz = root_pos[i:i+interval].shape[0]
            self._robot_root_states[:bz, 0:3] = root_pos[i:i+bz]
            self._robot_root_states[:bz, 3:7] = root_rot[i:i+bz]
            self._robot_dof_states[:bz, :, 0] = dof_pos[i:i+bz]
            self._robot_root_states[:,7:13] = 0.
            self._robot_dof_states[:, :, 1]= 0.
        
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self._root_states))
            self.gym.set_dof_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self._dof_state))
            _dof_pos = self._robot_dof_states[:,:,0].contiguous()
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(_dof_pos))
            _dof_vel = torch.zeros_like(_dof_pos)
            self.gym.set_dof_velocity_target_tensor(
                self.sim, gymtorch.unwrap_tensor(_dof_vel))
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self.render()
            if self.viewer is not None and object_pos is not None:
                self.gym.clear_lines(self.viewer)
                self.draw_point(object_pos[i] + offset)
            rigid_body_pos.append(
                self._robot_rigid_body_states[:bz, :, 0:3].clone()
            )
            rigid_body_rot.append(
                self._robot_rigid_body_states[:bz, :, 3:7].clone()
            )
        self.gym.clear_lines(self.viewer)
        rigid_body_pos = torch.cat(rigid_body_pos, dim=0)
        rigid_body_rot = torch.cat(rigid_body_rot, dim=0)

        assert rigid_body_pos.shape == (root_pos.shape[0], 15, 3)
        assert rigid_body_rot.shape == (root_rot.shape[0], 15, 4)
        assert torch.allclose(rigid_body_rot[:,0,:], root_rot, atol = 1e-3, rtol=0)
        assert torch.allclose(rigid_body_pos[:,0,:], root_pos, atol = 1e-3, rtol=0)

        rigid_body_pos = rigid_body_pos.to(offset.device) - offset
        rigid_body_rot = rigid_body_rot.to(offset.device)
        return rigid_body_pos, rigid_body_rot

    def adjust_height_first(self,root_pos,root_rot, dof_pos, rb_pos, extra = None):
        if extra is None:
            extra = self.extra
        orig_device = root_pos.device
        root_pos = root_pos.float().to(self.device)
        root_rot = root_rot.float().to(self.device)
        dof_pos = dof_pos.float().to(self.device)
        for i in range(60):
            if i == 0:
                self._robot_root_states[:, 2]   = root_pos[0,2] + 0.4
            self._robot_root_states[:, 0:2] = root_pos[0,:2]
            self._robot_root_states[:, 3:7] = root_rot[0]
            self._robot_root_states[:, 7:9] = 0.
            self._robot_root_states[:, 9] = torch.clamp_max(self._robot_root_states[:, 9], 0)
            self._robot_root_states[:, 10:13] = 0.
            self._robot_dof_states[:, :, 0] = dof_pos[0]
            self._robot_dof_states[:, :, 1] = 0.
            
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self._root_states))
            self.gym.set_dof_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self._dof_state))
            _dof_pos = self._robot_dof_states[:, :, 0].contiguous()
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(_dof_pos))
            _dof_vel = torch.zeros_like(_dof_pos)
            self.gym.set_dof_velocity_target_tensor(
                self.sim, gymtorch.unwrap_tensor(_dof_vel))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self.render()    
        height = self._robot_root_states[0,2] + extra
        height_offset = height - root_pos[0,2]
        root_pos[:,2] += height_offset
        rb_pos[:,:,2] += height_offset
        root_pos = root_pos.to(orig_device)
        rb_pos = rb_pos.to(orig_device)
        return rb_pos, root_pos

    def adjust_height_full(self,root_pos,root_rot, dof_pos, rb_pos, extra = None):
        if extra is None:
            extra = self.extra
        orig_device = root_pos.device
        root_pos = root_pos.float().to(self.device)
        root_rot = root_rot.float().to(self.device)
        dof_pos = dof_pos.float().to(self.device)

        heights = []
        for i in range(0, root_pos.shape[0], self.num_envs):
            bz = root_pos[i:i+self.num_envs].shape[0]
            batch_root_pos = root_pos[i:i+self.num_envs]
            batch_root_rot = root_rot[i:i+self.num_envs]
            batch_dof_pos = dof_pos[i:i+self.num_envs]
            for i in range(60):
                if i == 0:
                    self._robot_root_states[:bz, 2]   = batch_root_pos[:,2] + 0.4
                self._robot_root_states[:bz, 0:2] = batch_root_pos[:,:2]
                self._robot_root_states[:bz, 3:7] = batch_root_rot
                self._robot_root_states[:bz, 7:9] = 0.
                self._robot_root_states[:bz, 9] = torch.clamp_max(self._robot_root_states[:bz, 9], 0)
                self._robot_root_states[:bz, 10:13] = 0.
                self._robot_dof_states[:bz, :, 0] = batch_dof_pos
                self._robot_dof_states[:bz, :, 1] = 0.
                
                self.gym.set_actor_root_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self._root_states))
                self.gym.set_dof_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self._dof_state))
                _dof_pos = self._robot_dof_states[:, :, 0].contiguous()
                self.gym.set_dof_position_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(_dof_pos))
                _dof_vel = torch.zeros_like(_dof_pos)
                self.gym.set_dof_velocity_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(_dof_vel))
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self._refresh_sim_tensors()
                self.render()    
            height = self._robot_root_states[:bz,2].clone()
            heights.append(height)
        heights = torch.cat(heights, dim=0) + extra

        heights = heights.to(orig_device)
        root_pos = root_pos.to(orig_device)
        rb_pos = rb_pos.to(orig_device)
        height_offset = heights - root_pos[:,2]

        root_pos[:,2] += height_offset
        rb_pos[:,:,2] += height_offset.unsqueeze(-1)
        return rb_pos, root_pos



    def run(self, root_pos, root_rot, dof_pos, object_pos=None):
        root_pos = root_pos.float()
        root_rot = root_rot.float()
        dof_pos = dof_pos.float()
        rb_pos, rb_rot = self.forward_kinematics(root_pos, root_rot, dof_pos, object_pos)
        rb_pos, root_pos = self.adjust_height_full(root_pos, root_rot, dof_pos, rb_pos)
        assert torch.allclose(rb_rot[:,0,:], root_rot, atol = 1e-3, rtol=0)
        assert torch.allclose(rb_pos[:,0,:], root_pos, atol = 1e-3, rtol=0)
        return root_pos, root_rot, dof_pos, rb_pos, rb_rot



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--num_envs', type=int, default=1)
    args = parser.parse_args()
    return args




DOF_LIMIT_LOWER = [
    -1.0472, -1.0472, -0.8727, -0.8727, -0.6981, -0.7854, -3.1416, -3.1416,
    -2.0944, -2.7925, -1.0472, -3.1416, -2.0944, -2.7925, -2.0944, -2.4435,
    -2.7925,  0.0000, -0.5236, -0.9599, -0.6981, -2.0944, -2.4435, -2.0944,
    0.0000, -0.5236, -0.9599, -0.6981
    ]

DOF_LIMIT_UPPER = [
    1.0472, 1.5708, 0.8727, 0.8727, 1.0472, 0.7854, 1.0472, 1.3963, 
    2.0944, 0.0000, 3.1416, 1.3963, 2.0944, 0.0000, 2.0944, 1.0472, 
    2.0944, 2.7925, 0.5236, 0.9599, 0.6981, 2.0944, 1.0472, 2.7925, 
    2.7925, 0.5236, 0.9599, 0.6981
    ]


def validate_dof(dof, thresh = np.pi/4, eps = 0.01):
    T = dof.shape[0]
    dof_lower = torch.tensor(DOF_LIMIT_LOWER,device=dof.device,dtype=dof.dtype)
    dof_upper = torch.tensor(DOF_LIMIT_UPPER,device=dof.device,dtype=dof.dtype)
    dof = torch.clamp(dof, dof_lower, dof_upper)

    last_exceed_lower = dof[0] <= dof_lower + eps
    last_exceed_upper = dof[0] >= dof_upper - eps
    for t in range(T):
        change_to_upper = torch.logical_and(last_exceed_upper, dof[t] < dof_upper - thresh)
        dof[t, change_to_upper] = dof_upper[change_to_upper]
        change_to_lower = torch.logical_and(last_exceed_lower, dof[t] > dof_lower + thresh)
        dof[t, change_to_lower] = dof_lower[change_to_lower]

        last_exceed_lower = dof[t] <= dof_lower + eps
        last_exceed_upper = dof[t] >= dof_upper - eps
    return dof

DOF_NAMES = [
    'abdomen_x', 'abdomen_y', 'abdomen_z', 
    'neck_x', 'neck_y', 'neck_z', 
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow', 
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow', 
    'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 
    'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z'
    ]
DOF2INDEX = {n:i for i,n in enumerate(DOF_NAMES)}

def query_dof(name):
    if isinstance(name,list):
        return [DOF2INDEX[n] for n in name]
    else:
        return DOF2INDEX[name]


def dot(a,b):
    return (a*b).sum(-1).unsqueeze(-1)

def signed_angle(va, vb, vn):
    # from va to vb
    # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    return torch.arctan2(
        dot(torch.cross(va, vb), vn), 
        dot(va, vb))
    

def ik_revolute(a1, a2, parent_q):
    
    local_dof_axis = torch.tensor([0,1,0],device=a1.device,dtype=a1.dtype).expand_as(a1)
    global_dof_axis = torch_utils.quat_rotate(parent_q, local_dof_axis)
    
    a1 = torch_utils.quat_rotate(parent_q, a1)
    angle = signed_angle(a1, a2, global_dof_axis).squeeze(-1)
    joint_q = torch_utils.quat_from_angle_axis(angle, local_dof_axis)
    chlid_q = torch_utils.quat_mul(parent_q, joint_q)

    return chlid_q, angle


def ik_sphere(a1, b1, a2, b2, parent_q, debug = False):
    a1 = torch_utils.normalize(a1)
    b1 = torch_utils.normalize(b1)
    a2 = torch_utils.normalize(a2)
    b2 = torch_utils.normalize(b2)
    c1 = torch_utils.normalize(torch.cross(a1,b1))
    c2 = torch_utils.normalize(torch.cross(a2,b2))
    

    original_matrix = torch.stack([a1, b1, c1], -1)
    target_matrix   = torch.stack([a2, b2, c2], -1)
    rotation_matrix = target_matrix @ torch.linalg.inv(original_matrix)
    child_rotation = torch_utils.matrix_to_quat(rotation_matrix)
    joint_q = torch_utils.quat_mul(
        torch_utils.quat_conjugate(parent_q),
        child_rotation
    )
    joint_dofs = torch_utils.quat_to_exp_map(joint_q)

    return child_rotation, joint_dofs



def compute_linear_vel(pos, fps):
    nf = pos.shape[0]
    device = pos.device
    unit_time = 1 / fps
    prev_idx = torch.clamp(torch.arange(0, nf, device = device) - 1, 0, nf - 1)
    next_idx = torch.clamp(torch.arange(0, nf, device = device) + 1, 0, nf - 1)
    diff_time = (next_idx - prev_idx) * unit_time

    while len(pos.shape) > len(diff_time.shape):
        diff_time = diff_time.unsqueeze(-1)

    prev_pos = pos[prev_idx]
    next_pos = pos[next_idx]
    vel = (next_pos - prev_pos) / diff_time
    return vel



def compute_angular_vel(rot, fps):
    nf = rot.shape[0]
    device = rot.device
    unit_time = 1 / fps
    prev_idx = torch.clamp(torch.arange(0, nf, device = device) - 1, 0, nf - 1)
    next_idx = torch.clamp(torch.arange(0, nf, device = device) + 1, 0, nf - 1)
    diff_time = (next_idx - prev_idx) * unit_time

    while len(rot.shape) > len(diff_time.shape):
        diff_time = diff_time.unsqueeze(-1)

    prev_rot = rot[prev_idx]
    next_rot = rot[next_idx]
    expmap = torch_utils.quat_to_exp_map(torch_utils.quat_mul(next_rot, torch_utils.quat_conjugate(prev_rot)))
    anv = expmap / diff_time
    return anv