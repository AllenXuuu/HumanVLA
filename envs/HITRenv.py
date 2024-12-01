from isaacgym import gymapi, gymtorch
import torch,os,json
import numpy as np
from utils import torch_utils
import pickle as pkl
from .humanoid import HumanoidEnv
from collections import OrderedDict
import open3d as o3d
import trimesh
import imageio

class HITREnv(HumanoidEnv):
    def __init__(self, cfg) -> None:
        self.high_vhacd_resolution_assets = [
            '0d8931055c366e725f69bf5444f2d91331d16966',
            '9f0427019d5a329e5410547e4291b2c4b8b20195',
            '2479fa118c490958b3850a843177d870b2bc6f7a',
            '347414deea3205ff5e0c36c1acad28944fb5f1b2',
            '4d82784c7882172aba10d6a91421211c581758a4',
            '514731fa03071a52318224f7be986fd25864a47f',

            '01b0483516935b5ac613f1aec72fd518f73e23c3',
            '01cc0bbc121ef31692b03269b6946aeb2aca5f10',
            '1e2d034bba5bd59a9700af82f907340a83eb4cc3',
            '00366b86401aa16b702c21de49fd59b75ab9c57b',
            '2247f72a2a88ce31cd37aba1f687165054b1253a',
            '05206ad5b8ad9956a076ab73038089b964ddb2fd',
        ]

        self.num_pcd = cfg.num_pcd
        self.tasks = json.load(open(os.path.join(cfg.data_prefix, cfg.task_json)))
        # np.random.shuffle(self.tasks)
        if cfg.debug:
            self.tasks = self.tasks[:5]
        
        self.eval = cfg.eval
        self.test = cfg.test
        if self.eval:
            cfg.graphics_device_id = -1
            cfg.num_envs = len(self.tasks)
            cfg.enable_early_termination = False


        env_ids = range(cfg.num_envs*cfg.rank//cfg.world_size, cfg.num_envs*(cfg.rank+1)//cfg.world_size)
        self.num_envs = cfg.num_envs = len(env_ids)    
        self.env2task = []
        for env_id in env_ids:
            task_id = env_id % len(self.tasks)
            self.env2task.append(task_id)        
        self.eval_success_thresh = cfg.eval_success_thresh

        self.max_guide = 8
        self.num_guidance_obs = 3
        self.guide_proceed_dist = cfg.guide_proceed_dist

        self.enable_camera = cfg.get('enable_camera', False)
        self.enable_camera_tensor = cfg.get('enable_camera_tensor', False)
        super().__init__(cfg)
        self._robot_dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 0]
        self._robot_dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 1]
        
        self.hand_index = [self.body2index['left_hand'], self.body2index['right_hand']]
        self.foot_index = [self.body2index['left_foot'], self.body2index['right_foot']]

        self.consecutive_success_thresh = self.cfg.consecutive_success_thresh
        if self.test:
            self.consecutive_success_thresh = 5
        self.enable_early_termination = cfg.enable_early_termination


        print(f'#Num Tasks {len(self.tasks)}')
        print(f'#Num Envs {self.num_envs}')
        print(f'#Num Roots {self.num_root}')
        print(f'#Num Objects {self.num_object}')
        print(f'#Num RigidBodys {self.num_rigid_body}')

    def create_sim(self):
        if self.enable_camera:
            self.graphics_device_id = self.sim_device_id
        return super().create_sim()

    @property
    def obs_space(self):
        raise NotImplementedError
    
    def create_buffer(self):
        self.last_action = torch.zeros((self.num_envs, self.num_action), device=self.device, dtype=torch.float32)
        self.reward_buf     = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.progress_buf   = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_termination_buf  = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_timeout_buf      = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.timeout_limit          = torch.zeros(self.num_envs, device=self.device, dtype=torch.long).fill_(self.cfg.max_episode_length)
        self.success_steps          = torch.zeros(self.num_envs, device=self.device, dtype=torch.long).fill_(self.cfg.max_episode_length)
        self.guide_buf  = torch.zeros((self.num_envs, self.num_guidance_obs), device=self.device, dtype=torch.float32)

    def load_asset(self):
        super().load_asset()
        
        filefix_keys = []
        self.asset_files = []
        self.file2asset = {}

        for env_id, task_id in enumerate(self.env2task):
            task = self.tasks[task_id]
            for name, obj in task['object'].items():
                file = obj['file']
                fix = obj['fix_base_link']
                key = (file, fix)
                if file in self.file2asset:
                    pass
                else:
                    self.file2asset[file] = len(self.asset_files)
                    self.asset_files.append(file)
                if key not in filefix_keys:
                    filefix_keys.append(key)
        
        self.object_assets = {}
        for file, fix in filefix_keys:
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.use_mesh_materials = True
            object_asset_options.fix_base_link = fix
            object_asset_options.armature = 0.01
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 3000000 if file in self.high_vhacd_resolution_assets else 100000
            object_asset_options.vhacd_params.max_num_vertices_per_ch = 512
            object_asset_root = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, file)
            object_asset_file = f'{file}.urdf'
            object_asset = self.gym.load_asset(self.sim, object_asset_root, object_asset_file, object_asset_options)
            self.object_assets[(file,fix)] = object_asset
    
        self.asset_pcd = []
        for idx, file in enumerate(self.asset_files):
            pcdpath = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, file, f'{file}_pcd1000.xyz')
            pcd = np.loadtxt(pcdpath)
            pcd = torch.from_numpy(pcd).float().to(self.device)
            self.asset_pcd.append(pcd)
        self.asset_pcd = torch.stack(self.asset_pcd, 0)
        centrioids, self.asset_pcd = torch_utils.farthest_point_sample(self.asset_pcd, self.num_pcd)

    def create_scene(self):
        lower = gymapi.Vec3(-self.cfg.spacing, -self.cfg.spacing, 0.0)
        upper = gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing)
        
        self.num_rigid_body = 0
        self.num_root = 0
        self.num_object = 0
        self.num_robot = 0

        self.env_handle     = [] # [<handle>, ...] (num_envs)
        self.robot_handle   = [] # [<handle>, ...] (num_envs)
        self.object_handle  = [] # [{'bed': <handle>, ...} ,...]
        self.camera_handle  = []
        self.camera_tensors = []

        self.init_trans = [] # (num_root, 3) 
        self.init_rot   = [] # (num_root, 4)
        self.goal_trans = [] # (num_root, 3)
        self.goal_rot   = [] # (num_root, 4)

        self.root2env           = [] # (num_root)
        self.env2rootlist       = [] # [[...], [...] , ...]
        self.env2objectsize     = [] # (num_env)
        self.robot2root         = [] # (num_robot) 
        self.robot2rb           = [] # (num_robot, 15) 
        self.object2env         = [] # (num_object)
        self.object2name        = [] # (num_object)
        self.object2root        = [] # (num_object)
        self.object2fix         = [] # (num_object)
        self.object2asset       = [] # (num_object)
        self.object2scale       = [] # (num_object)
        
        for env_id in range(self.num_envs):
            ##### load task
            task_id = self.env2task[env_id]
            task = self.tasks[task_id]
            
            ##### spawn env
            env_ptr = self.gym.create_env(self.sim, lower, upper, round(self.num_envs ** 0.5))
            self.env_handle.append(env_ptr)
            self.env2rootlist.append([])
            self.object_handle.append({})

            ##### spawn robot
            init_trans = task['robot']['init_pos']
            init_rot = task['robot']['init_rot']
                
            init_robot_pose = gymapi.Transform(p=gymapi.Vec3(*init_trans), r=gymapi.Quat(*init_rot))
            robot = self.gym.create_actor(env_ptr, self.humanoid_asset, init_robot_pose, "robot", env_id, 0, 0)
            for j in range(self.num_body):
                self.gym.set_rigid_body_color(env_ptr, robot, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
            dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, robot, dof_prop)
        
            self.init_trans.append(init_trans)
            self.init_rot.append(init_rot)
            # not used goal trans/rot for robot
            self.goal_trans.append(init_trans)
            self.goal_rot.append(init_rot)
            
            self.robot_handle.append(robot)
            self.robot2root.append(self.num_root)
            self.env2rootlist[env_id].append(self.num_root)
            self.root2env.append(env_id)

            self.num_root += 1
            self.num_robot += 1
            for i in range(self.num_body):
                self.robot2rb.append(self.num_rigid_body)
                self.num_rigid_body += 1

            if self.enable_camera:
                cam_props = gymapi.CameraProperties()
                cam_props.horizontal_fov = self.cfg.horizontal_fov
                cam_props.width = self.cfg.camera_width
                cam_props.height = self.cfg.camera_height
                cam_props.enable_tensors = self.enable_camera_tensor
                self.camera_pos_offset = self.cfg.camera_pos
                self.camera_rot_offset = self.cfg.camera_rot
                pos = gymapi.Vec3(*self.camera_pos_offset)
                rot = gymapi.Quat(*self.camera_rot_offset)

                camera_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.attach_camera_to_body(
                    camera_handle, env_ptr, self.gym.find_actor_rigid_body_handle(env_ptr, robot, 'head'), 
                    gymapi.Transform(p=pos, r=rot), gymapi.FOLLOW_TRANSFORM)
                self.camera_handle.append(camera_handle)

                if self.enable_camera_tensor:
                    cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                    self.camera_tensors.append(torch_cam_tensor)
                
            ##### spawn objects
            self.env2objectsize.append(len(task['object']))
            for name in sorted(task['object'].keys()):
                object_info = task['object'][name]
                
                init_trans  = object_info['init_pos']
                init_rot    = object_info['init_rot']
                goal_trans  = object_info['goal_pos']
                goal_rot    = object_info['goal_rot']
                self.init_trans.append(init_trans)
                self.init_rot.append(init_rot)
                self.goal_trans.append(goal_trans)
                self.goal_rot.append(goal_rot)
                
                init_object_pose = gymapi.Transform(p = gymapi.Vec3(*init_trans), r = gymapi.Quat(*init_rot))
                object_handle = self.gym.create_actor(
                    env_ptr, self.object_assets[(object_info['file'],object_info['fix_base_link'])], init_object_pose, name, env_id, 0, segmentationId = 0) # todo segmentationId
                # self.gym.set_actor_scale(env_ptr, object_handle, object_info['scale'])
                self.object_handle[env_id][name] = object_handle
                
                self.root2env.append(env_id)
                self.env2rootlist[env_id].append(self.num_root)
                self.object2name.append(name)
                self.object2root.append(self.num_root)
                self.object2env.append(env_id)
                self.object2fix.append(object_info['fix_base_link'] )
                self.object2asset.append(self.file2asset[object_info['file']])
                self.object2scale.append(object_info['scale'])


                self.num_root += 1
                self.num_rigid_body += 1 
                self.num_object += 1


        assert self.num_robot == self.num_envs
        assert self.num_robot + self.num_object == self.num_root
        assert self.num_robot * self.num_body + self.num_object == self.num_rigid_body

        assert len(self.env_handle) == self.num_envs
        assert len(self.robot_handle) == self.num_envs
        assert len(self.object_handle) == self.num_envs
        

        self.init_trans = torch.tensor(self.init_trans).to(self.device)
        self.init_rot   = torch.tensor(self.init_rot).to(self.device)
        self.goal_trans = torch.tensor(self.goal_trans).to(self.device)
        self.goal_rot   = torch.tensor(self.goal_rot).to(self.device)
        self.root2env   = torch.tensor(self.root2env).to(self.device)

        self.env2objectsize = torch.tensor(self.env2objectsize).to(self.device)
        self.robot2root     = torch.tensor(self.robot2root, device=self.device) 
        self.robot2rb       = torch.tensor(self.robot2rb, device=self.device).view(self.num_robot, self.num_body)  
        self.object2root    = torch.tensor(self.object2root, device=self.device) 
        self.object2env     = torch.tensor(self.object2env, device=self.device) 
        self.object2fix     = torch.tensor(self.object2fix).to(self.device)
        self.object2asset   = torch.tensor(self.object2asset).to(self.device)
        self.object2scale   = torch.tensor(self.object2scale).to(self.device)
        
        assert len(self.env_handle)     == self.num_envs
        assert len(self.robot_handle)   == self.num_envs
        assert len(self.object_handle)  == self.num_envs
        assert self.init_trans.shape    == (self.num_root, 3)
        assert self.init_rot.shape      == (self.num_root, 4)
        assert self.goal_trans.shape    == (self.num_root, 3)
        assert self.goal_rot.shape      == (self.num_root, 4)
        assert self.root2env.shape      == (self.num_root,)  
        assert len(self.env2rootlist)   == self.num_envs
        assert self.env2objectsize.shape== (self.num_envs,)  
        assert self.robot2root.shape    == (self.num_robot,)  
        assert self.robot2rb.shape      == (self.num_robot, 15)  
        assert len(self.object2name)    == self.num_object
        assert self.object2root.shape   == (self.num_object,)  
        assert self.object2env.shape    == (self.num_object,)  
        assert self.object2fix.shape    == (self.num_object,)  
        assert self.object2asset.shape  == (self.num_object,)  
        assert self.object2scale.shape  == (self.num_object,)      

        ##################### todo tasks
        self.envname2object = {(env_id.item(), name) : obj_id for obj_id, (env_id, name) in enumerate(zip(self.object2env, self.object2name))}

        self.movetask_objectid = torch.zeros((self.num_envs,), dtype=torch.long,device=self.device)
        self.movetask_rootid = torch.zeros((self.num_envs,), dtype=torch.long,device=self.device)

        self.preguide_full = torch.zeros((self.num_envs, self.max_guide, 3), dtype=torch.float32, device=self.device)        
        self.preguide_length = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        self.postguide_full = torch.zeros((self.num_envs, self.max_guide, 3), dtype=torch.float32, device=self.device)
        self.postguide_length = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        
        for env_id in range(self.num_envs):
            task_id = self.env2task[env_id]
            task = self.tasks[task_id]
            
            object_name = task['move']
            object_id = self.envname2object[(env_id, object_name)]
            root_id = self.object2root[object_id].item()
            
            self.movetask_objectid[env_id] = object_id
            self.movetask_rootid[env_id] = root_id

            plan = task['plan']
            
            init_z = self.init_trans[root_id,2].item()
            goal_z = self.goal_trans[root_id,2].item()
            guide_z = max(init_z, goal_z) + self.cfg.carry_height if init_z > 0.1 or goal_z > 0.1 else goal_z

            for guide_id, p in enumerate(plan['pre_waypoint']):
                p[2] = guide_z
                self.preguide_full[env_id, guide_id] = torch.tensor(p, dtype=torch.float32, device=self.device)
                self.preguide_length[env_id] += 1
                
            for guide_id, p in enumerate(plan['post_waypoint']):
                p[2] = guide_z
                self.postguide_full[env_id, guide_id] = torch.tensor(p, dtype=torch.float32, device=self.device)
                self.postguide_length[env_id] += 1
        
        ########### running time buffer
        self.movenow_preguidecomplete = torch.zeros(self.num_envs, dtype=torch.bool,device=self.device)
        self.movenow_guideidx = torch.zeros(self.num_envs, dtype=torch.long,device=self.device)
        self.movenow_guide = self.preguide_full[:,0,:].clone()
        
    def reset(self):
        reset = torch.logical_or(self.reset_termination_buf == 1, self.reset_timeout_buf == 1)
        reset_env_ids = torch.where(reset)[0]
        self.eval_last_runs(reset_env_ids)
        self.reset_env(reset_env_ids)
        self.success_steps[reset_env_ids] = self.cfg.max_episode_length
        self.compute_guidance()
        self.guide_buf[:]=self.compute_guide_obs(
            self._root_states[self.robot2root, 0:3],
            self._root_states[self.robot2root, 3:7],
            self.movenow_guide,
        )
        return self.reset_output()

        
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self._refresh_sim_tensors()
        self.compute_observation()
        self.compute_reward()
        self.compute_termination()
        self.compute_success_steps()

    def compute_success_steps(self):
        dist = torch.norm(
            self._root_states[self.movetask_rootid,0:3] - self.goal_trans[self.movetask_rootid], 
            p=2, dim=-1)
        success_mask = dist < self.eval_success_thresh
        self.success_steps[success_mask] = torch.min(
            self.progress_buf[success_mask],
            self.success_steps[success_mask]
            )
    

    def calc_diff_pos(self, p1 , p2):
        return torch.norm(p1-p2,dim=-1)
    
    def calc_diff_rot(self, r1, r2):
        diff_ang, _ = torch_utils.quat_to_angle_axis(torch_utils.quat_mul(
            r1, torch_utils.quat_conjugate(r2)))
        diff_ang = diff_ang.abs()
        return diff_ang    


    def compute_guidance(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        n = len(env_ids)
        robot_rb_state = self._rigid_body_state[self.robot2rb[env_ids]].view(n, self.num_body, 13)

        foot_pos = robot_rb_state[:,self.foot_index,:2].mean(1)
        guide_dist = torch.norm(foot_pos - self.movenow_guide[env_ids,:2], dim=-1)
        guide_ok = guide_dist < self.guide_proceed_dist

        guide_proceed_envs = env_ids[guide_ok]
        self.movenow_guideidx[guide_proceed_envs] += 1
        postguide_alldone_mask = torch.logical_and(self.movenow_preguidecomplete[env_ids], self.movenow_guideidx[env_ids]>=self.postguide_length[env_ids])
        postguide_alldone_envs = env_ids[postguide_alldone_mask]
        self.movenow_guideidx[postguide_alldone_envs] = self.postguide_length[postguide_alldone_envs]
        preguide_alldone_mask = torch.logical_and(~self.movenow_preguidecomplete[env_ids], self.movenow_guideidx[env_ids]==self.preguide_length[env_ids])
        preguide_alldone_envs = env_ids[preguide_alldone_mask]
        self.movenow_guideidx[preguide_alldone_envs] = 0.
        self.movenow_preguidecomplete[preguide_alldone_envs] = True

        self.movenow_guide[env_ids] = torch.where(
            self.movenow_preguidecomplete[env_ids].unsqueeze(-1).tile(3),
            self.postguide_full[env_ids, self.movenow_guideidx[env_ids]],
            self.preguide_full[env_ids, self.movenow_guideidx[env_ids]]
        )
        self.movenow_guide[postguide_alldone_envs] = self.goal_trans[self.movetask_rootid[postguide_alldone_envs]]
    
    def eval_last_runs(self,env_ids):
        pass
    
    def reset_output(self):
        raise NotImplementedError

    def step_output(self):
        raise NotImplementedError    
    
    def compute_observation(self,env_ids = None):
        raise NotImplementedError
    
    def compute_termination(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def export_stats(self):
        raise NotImplementedError
    
    def export_logging_stats(self):
        raise NotImplementedError
    
    def export_evaluation(self,):
        object_state = self._root_states[self.movetask_rootid]
        object_pos = object_state[:,0:3]
        goal_pos = self.goal_trans[self.movetask_rootid]
        dist = torch.norm(goal_pos - object_pos, p=2, dim=-1)
        result = {
            'l2_dist' : dist.mean().item(),
            'success' : (dist < self.eval_success_thresh).float().mean().item(),
            'success_steps' : self.success_steps.float().mean().item() * self.dt
        }
        # result = {
        #     'l2_dist' : dist,
        #     'success' : (dist < self.eval_success_thresh).float(),
        #     'success_steps' : self.success_steps.float() * self.dt
        # }
        return result
    

    ######################### prepare feats
    def dof_to_obs(self, dof_pos):
        dof_obs = []
        for offset in self.revolute_dof_offset:
            angle = dof_pos[:, offset]
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=angle.dtype, device=self.device)
            q = torch_utils.quat_from_angle_axis(angle, axis)
            obs  = torch_utils.quat_to_tan_norm(q)
            dof_obs.append(obs)
        for offset in self.sphere_dof_offset:
            angle = dof_pos[:,offset : offset + 3]
            q = torch_utils.exp_map_to_quat(angle)
            obs = torch_utils.quat_to_tan_norm(q)
            dof_obs.append(obs)
        dof_obs = torch.cat(dof_obs,1)
        return dof_obs
    
    def compute_ref_frame_obs(
            self, root_pos, root_rot, root_vel, root_anv, dof_pos, dof_vel, key_body_pos, obj_pos):
        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        root_rot_obs = torch_utils.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
        
        root_h_obs = root_h
        
        local_root_vel = torch_utils.quat_rotate(heading_rot, root_vel)
        local_root_anv = torch_utils.quat_rotate(heading_rot, root_anv)

        bz, nkb, _ = key_body_pos.shape
        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_body_pos - root_pos_expand
        heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, nkb, 1)).view(bz * nkb, 4)
        local_key_body_pos = local_key_body_pos.view(bz * nkb, 3)
        local_key_body_pos = torch_utils.quat_rotate(heading_rot_expand, local_key_body_pos)
        local_key_body_pos = local_key_body_pos.view(bz, nkb * 3)
        
        dof_obs = self.dof_to_obs(dof_pos)

        local_obj_pos = torch_utils.quat_rotate(heading_rot, obj_pos - root_pos)
        local_obj_pos[:,2] = 0.
        obs = torch.cat((
            root_h_obs, 
            root_rot_obs, 
            local_root_vel, 
            local_root_anv, 
            dof_obs, 
            dof_vel, 
            local_key_body_pos,
            local_obj_pos), dim=-1)
        
        return obs
    
    def compute_guide_obs(self, root_pos, root_rot, guide_pos):
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_guide_pos = torch_utils.quat_rotate(heading_rot, guide_pos - root_pos)
        return local_guide_pos
    
    def compute_goal_obs(self, root_pos, root_rot, goal_pos, goal_rot):
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_goal_pos = torch_utils.quat_rotate(heading_rot, goal_pos - root_pos)
        local_goal_rot = torch_utils.quat_mul(heading_rot, goal_rot)
        local_goal_rot_tannorm = torch_utils.quat_to_tan_norm(local_goal_rot)
        obs = torch.cat((
            local_goal_pos,
            local_goal_rot_tannorm,
            ),    
        dim = -1)
        return obs
        
    def compute_object_obs(self, root_pos, root_rot, obj_pos, obj_rot, obj_vel, obj_anv):
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_obj_pos = torch_utils.quat_rotate(heading_rot, obj_pos - root_pos)
        local_obj_rot = torch_utils.quat_mul(heading_rot, obj_rot)
        local_obj_rot_tannorm = torch_utils.quat_to_tan_norm(local_obj_rot)
        local_obj_vel = torch_utils.quat_rotate(heading_rot, obj_vel)
        local_obj_anv = torch_utils.quat_rotate(heading_rot, obj_anv)
        obs = torch.cat((
            local_obj_pos,
            local_obj_rot_tannorm,
            local_obj_vel,
            local_obj_anv,
            ),    
        dim = -1)
        return obs

    def compute_prop_obs(self, body_pos, body_rot, body_vel, body_ang_vel):
        bz, nbody, _ = body_pos.shape
        assert nbody == self.num_body
        
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        root_h_obs = root_h
        
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        flat_heading_rot = heading_rot.unsqueeze(-2).repeat((1, nbody, 1)).reshape(bz * nbody, 4)
        
        local_body_pos = body_pos - root_pos.unsqueeze(-2)
        local_body_pos = local_body_pos.reshape(bz * nbody, 3)
        local_body_pos = torch_utils.quat_rotate(flat_heading_rot, local_body_pos)
        local_body_pos = local_body_pos.reshape(bz, nbody * 3)
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        local_body_rot = body_rot.reshape(bz * nbody, 4)
        local_body_rot = torch_utils.quat_mul(flat_heading_rot, local_body_rot)
        local_body_rot_tannorm = torch_utils.quat_to_tan_norm(local_body_rot)
        local_body_rot_tannorm = local_body_rot_tannorm.reshape(bz, nbody * 6)
        
        local_body_vel = body_vel.reshape(bz * nbody,3)
        local_body_vel = torch_utils.quat_rotate(flat_heading_rot, local_body_vel)
        local_body_vel = local_body_vel.reshape(bz, nbody * 3)
        
        local_body_anv = body_ang_vel.reshape(bz * nbody,3)
        local_body_anv = torch_utils.quat_rotate(flat_heading_rot, local_body_anv)
        local_body_anv = local_body_anv.reshape(bz, nbody * 3)


        obs = torch.cat((
            root_h_obs, 
            local_body_pos, 
            local_body_rot_tannorm, 
            local_body_vel, 
            local_body_anv,
            ),    
        dim = -1)
        return obs