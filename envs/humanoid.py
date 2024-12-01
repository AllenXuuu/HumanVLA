from .base import BaseEnv
from isaacgym import gymapi, gymtorch
import torch,os
import numpy as np
from utils import torch_utils
import pickle as pkl
import yaml

class HumanoidEnv(BaseEnv):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        self._refresh_sim_tensors()

        ######## root tensor
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        
        ######## rb tensor
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        
        ######## dof tensor
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        ######## contact force tensor
        self._contact_force_state = gymtorch.wrap_tensor(contact_force_tensor)
    
        self.build_pd_offset_scale()

        if 'motion' in self.cfg:
            self.motion = self.load_motion(self.cfg.motion)
    

    def build_pd_offset_scale(self):
        pd_low = self.dof_limit_lower.clone()
        pd_high = self.dof_limit_upper.clone()
        revolute = []
        sphere = []
        for a in self.names_dof:
            if a.endswith('_x') or a.endswith('_y')  or a.endswith('_z'):
                sphere.append(a)
            else:
                revolute.append(a)
        set_sphere = list(set([a.replace('_x','').replace('_y','').replace('_z','') for a in sphere]))
        assert len(set_sphere) * 3 == len(sphere)

        self.revolute_dof_offset = []
        self.sphere_dof_offset   = []
        for joint in revolute:
            idx = self.dof2index[joint]
            self.revolute_dof_offset.append(idx)
        for joint in set_sphere:
            idx = self.dof2index[joint + '_x']
            idy = self.dof2index[joint + '_y']
            idz = self.dof2index[joint + '_z']
            self.sphere_dof_offset.append(idx)
            assert idy == idx + 1 and idz == idx + 2
        self.revolute_dof_offset = torch.tensor(self.revolute_dof_offset,device=self.device)
        self.sphere_dof_offset = torch.tensor(self.sphere_dof_offset,device=self.device)

        print('Revolute', revolute)
        print('Sphere', sphere)
        
        for joint in revolute:
            idx = self.dof2index[joint]
            pd_offset = 0.5 * (pd_low[idx] + pd_high[idx])
            pd_scale  = 0.7 * (pd_high[idx] - pd_low[idx])
            pd_low[idx]  = pd_offset - pd_scale
            pd_high[idx] = pd_offset + pd_scale
        
        for joint in set_sphere:
            idx = self.dof2index[joint + '_x']
            idy = self.dof2index[joint + '_y']
            idz = self.dof2index[joint + '_z']
            limit = torch.tensor([
                pd_low[idx], pd_high[idx],
                pd_low[idy], pd_high[idy],
                pd_low[idz], pd_high[idz],
            ])
            limit = torch.max(torch.abs(limit)).item() * 1.2
            limit = min(limit, np.pi)
            pd_low[[idx,idy,idz]] = -limit
            pd_high[[idx,idy,idz]] = limit
        
        print('pd_low',pd_low)
        print('pd_high',pd_high)

        scale = (pd_high - pd_low) * 0.5
        offset = (pd_high + pd_low) * 0.5
        
        self.pd_scale = scale
        self.pd_offset = offset
        self.pd_low = pd_low
        self.pd_high = pd_high       

    
    def create_env(self):
        self.create_ground()
        self.load_asset()
        self.create_scene()
        self.gym.prepare_sim(self.sim)
        pass
    
    def load_asset(self):
        humanoid_asset_options = gymapi.AssetOptions()
        humanoid_asset_options.fix_base_link = self.cfg.actor.get('fix_base_link', False)
        humanoid_asset_options.angular_damping = 0.01
        humanoid_asset_options.max_angular_velocity = 100.0
        humanoid_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset_root = os.path.join(self.cfg.data_prefix, self.cfg.actor.asset_root)
        humanoid_asset_file = self.cfg.actor.asset_file
        self.humanoid_asset = self.gym.load_asset(self.sim, humanoid_asset_root, humanoid_asset_file, humanoid_asset_options)

        print('======================================')
        print('load huamanoid asset: %s' % os.path.join(humanoid_asset_root, humanoid_asset_file))        
        self.num_dof = self.gym.get_asset_dof_count(self.humanoid_asset)
        self.names_dof = self.gym.get_asset_dof_names(self.humanoid_asset)
        self.dof2index = {n:i for i, n in enumerate(self.names_dof)}
        print(f'{self.num_dof} DoFs: {self.names_dof}')

        self.num_body = self.gym.get_asset_rigid_body_count(self.humanoid_asset)
        self.names_body = self.gym.get_asset_rigid_body_names(self.humanoid_asset)
        self.body2index = {n:i for i, n in enumerate(self.names_body)}
        print(f'{self.num_body} Bodies: {self.names_body}')
        # ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']

        self.num_joints = self.gym.get_asset_joint_count(self.humanoid_asset)
        self.names_joints = self.gym.get_asset_joint_names(self.humanoid_asset)
        self.joint2index = {n:i for i, n in enumerate(self.names_joints)}
        print(f'{self.num_joints} Joints: {self.names_joints}')

        dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
        self.dof_limit_lower = torch.tensor([ dof_prop['lower'][j] for j in range(self.num_dof)]).to(self.device)
        self.dof_limit_upper = torch.tensor([ dof_prop['upper'][j] for j in range(self.num_dof)]).to(self.device)
        assert torch.all(self.dof_limit_lower < self.dof_limit_upper)
    
    def create_scene(self):
        lower = gymapi.Vec3(-self.cfg.spacing, -self.cfg.spacing, 0.0)
        upper = gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing)
        self.envs = []
        self.humanoids = []
        for env_id in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, round(self.num_envs ** 0.5))
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0,0,0.89)
            start_pose.r = gymapi.Quat(0,0,0,1)
        
            humanoid_handle = self.gym.create_actor(env_ptr, self.humanoid_asset, start_pose, "humanoid", env_id, 0, 0)
            
            for j in range(self.num_body):
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
            
            dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
            
            self.humanoids.append(humanoid_handle)
            self.envs.append(env_ptr)
        pass

 
    def pre_physics_step(self, actions):
        # assert hasattr(self,'last_action')
        if hasattr(self,'last_action'):
            self.last_action[:] = actions[:]
        actions = actions.to(self.device).clone()
        pd_target =  actions * self.pd_scale + self.pd_offset
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_target)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        return
    

    ##################################################################################### motionlib
    def load_motion(self,paths):
        weights = None
        if isinstance(paths, str) and paths.endswith('.pkl'):
            paths = [paths]
        if isinstance(paths, str) and paths.endswith('.yaml'):
            paths = yaml.load(open(os.path.join(self.cfg.data_prefix, paths)), Loader=yaml.FullLoader)
            if isinstance(paths[0], (list, tuple)):
                weights = [p[1] for p in paths]
                paths = [p[0] for p in paths]
        if weights is None:
            weights = [1 for p in paths]
        motions = {
            'fps'          : [],
            'time_length'  : [],
            'num_frame'    : [],
            'weight'       : [],
            'offset'       : [],
            'rigid_body_pos'  : torch.empty((0,15,3)),
            'rigid_body_rot'  : torch.empty((0,15,4)),
            'rigid_body_vel'  : torch.empty((0,15,3)),
            'rigid_body_anv'  : torch.empty((0,15,3)),
            'dof_pos'       : torch.empty((0,28)),
            'dof_vel'       : torch.empty((0,28)),
            'object_pos'    : torch.empty((0,3)),
            'object_vel'    : torch.empty((0,3)),
            'is_samp'   : []
        }
        offset = 0
        for i,path in enumerate(paths):
            cur_motion = pkl.load(open(os.path.join(self.cfg.data_prefix, path),'rb'))
            motions['fps'].append(cur_motion['fps'])
            motions['time_length'].append(cur_motion['time_length'])
            motions['num_frame'].append(cur_motion['num_frame'])
            motions['weight'].append(weights[i])
            motions['offset'].append(offset)
            offset += cur_motion['num_frame']

            for key in ['rigid_body_pos', 'rigid_body_rot', 'rigid_body_vel', 'rigid_body_anv', 'dof_pos', 'dof_vel', 'object_pos', 'object_vel']:
                assert cur_motion[key].shape[0] == cur_motion['num_frame']
                motions[key] = torch.cat([motions[key],cur_motion[key]], dim=0)
            
            if os.path.split(path)[-1].startswith('OMOMO_'):
                motions['is_samp'].append(0)
            elif os.path.split(path)[-1].startswith('SAMP_'):
                motions['is_samp'].append(1)
            else:
                raise NotImplementedError(path, os.path.split(path))
            
                
        for key in motions:
            if isinstance(motions[key],list):
                motions[key] = torch.tensor(motions[key]).to(self.device)
            elif isinstance(motions[key],torch.Tensor):
                motions[key] = motions[key].to(self.device)
        motions['weight'] = motions['weight'] / motions['weight'].sum()
        for k,v in motions.items():
            assert v.shape[0] in [offset, len(paths)],(k,v.shape, offset, len(paths))
        print(f'# RefMotionFiles = {len(paths)}')
        print(f'# RefMotionFrames = {offset}')
        return motions
    

    def query_motion_frame_index(self, motion_ids, motion_times):
        fps = self.motion['fps'][motion_ids]
        nf  = self.motion['num_frame'][motion_ids]
        motion_frames = motion_times * fps
        motion_frames = torch.round(motion_frames).long()
        motion_frames = torch.clamp(motion_frames, torch.zeros_like(motion_frames), nf - 1)
        return motion_frames

    def query_motion_frame_length(self, motion_ids):
        return self.motion['num_frame'][motion_ids]
    
    def query_motion_time_length(self, motion_ids):
        return self.motion['time_length'][motion_ids]
    
    def query_motion_fps(self, motion_ids):
        return self.motion['fps'][motion_ids]
    
    def query_motion_state(self, motion_ids, motion_times):
        fps = self.motion['fps'][motion_ids]
        nf  = self.motion['num_frame'][motion_ids]
        offset = self.motion['offset'][motion_ids]
        motion_frames = motion_times * fps
        motion_frames = torch.round(motion_frames).long()
        motion_frames = torch.clamp(motion_frames, torch.zeros_like(motion_frames), nf - 1)
        motion_frames_index = offset + motion_frames

        rigid_body_pos = self.motion['rigid_body_pos'][motion_frames_index]
        rigid_body_rot = self.motion['rigid_body_rot'][motion_frames_index]
        rigid_body_vel = self.motion['rigid_body_vel'][motion_frames_index]
        rigid_body_anv = self.motion['rigid_body_anv'][motion_frames_index]
        dof_pos = self.motion['dof_pos'][motion_frames_index]
        dof_vel = self.motion['dof_vel'][motion_frames_index]
        object_pos = self.motion['object_pos'][motion_frames_index]
        object_vel = self.motion['object_vel'][motion_frames_index]

        return {
            'motion_frames'     : motion_frames,
            'rigid_body_pos'    : rigid_body_pos,
            'rigid_body_rot'    : rigid_body_rot,
            'rigid_body_vel'    : rigid_body_vel,
            'rigid_body_anv'    : rigid_body_anv,
            'dof_pos'           : dof_pos,
            'dof_vel'           : dof_vel,
            'object_pos'        : object_pos,
            'object_vel'        : object_vel,
        }

    def sample_motion_ids(self, n, samp = False):
        if samp:
            is_samp = self.motion['is_samp'].bool()
            weight = self.motion['weight']
            ids = torch.arange(len(weight), device=self.device)
            weight = weight[is_samp]
            ids = ids[is_samp]
        else:
            weight = self.motion['weight']
            ids = torch.arange(len(weight), device=self.device)
                
        weight = weight / weight.sum()
        sample_ids = torch.multinomial(weight, num_samples=n, replacement=True)
        ids = ids[sample_ids]
        return ids

    def sample_motion_time(self, motion_ids, min_phase = 0., max_phase = 1., truncate_time = 0.):
        phase = torch.rand(len(motion_ids)).to(self.device)
        phase = min_phase + phase * (max_phase - min_phase)
        timelength = self.motion['time_length'] - truncate_time
        timelength = timelength[motion_ids]
        return phase * timelength
        