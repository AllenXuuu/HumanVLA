from isaacgym import gymapi, gymtorch
import torch,os,json
import numpy as np
from utils import torch_utils
import pickle as pkl
from .HITRenv import HITREnv
from collections import OrderedDict
import open3d as o3d
import trimesh
import imageio



class HITRCarryEnv(HITREnv):
    def __init__(self, cfg) -> None:
        self.num_prop_obs = 15 * (3 + 6 + 3 + 3) - 2
        self.num_goal_obs = 3 + 6
        self.num_obj_obs = 3 + 6 + 3 + 3
        self.num_guidance_obs = 3
        self.num_obs = self.num_prop_obs + self.num_goal_obs + self.num_obj_obs + self.num_guidance_obs

        self.num_ref_obs_frames = cfg.num_ref_obs_frames
        self.num_ref_obs_per_frame = cfg.num_ref_obs_per_frame
        self.num_bps = cfg.num_bps


        super().__init__(cfg)

        self.amp_body_name = cfg.amp_body_name
        self.amp_body_idx = [self.body2index[n] for n in self.amp_body_name]
        
        print(f'#Num Envs {self.num_envs}')
        print(f'#Num Roots {self.num_root}')
        print(f'#Num Objects {self.num_object}')
        print(f'#Num RigidBodys {self.num_rigid_body}')


        self.fall_thresh = cfg.fall_thresh
        self.ignore_contact_name = cfg.ignore_contact_name
        self.ignore_contact_idx = [self.body2index[n] for n in self.ignore_contact_name]

        

        
        self.stats_step = {}

    @property
    def obs_space(self):
        return {
            'obs' : self.num_prop_obs + self.num_goal_obs + self.num_obj_obs + self.num_guidance_obs,
            'bps' : (self.num_bps, 3)
        }
    
    def reset_output(self):
        obs = torch.cat([self.prop_buf, self.obj_buf, self.goal_buf, self.guide_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.movetask_objectid]]
        obs_buf = {
            'obs' : obs,
            'bps' : bps
        }
        return obs_buf

    def step_output(self):
        obs = torch.cat([self.prop_buf, self.obj_buf, self.goal_buf, self.guide_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.movetask_objectid]]
        obs_buf = {
            'obs' : obs,
            'bps' : bps
        }
        extra = {
            'amp_obs' : self.amp_obs_buf.reshape(self.num_envs, self.num_ref_obs_frames * self.num_ref_obs_per_frame)
        }

        return obs_buf, self.reward_buf, self.reset_termination_buf, self.reset_timeout_buf, extra


    def load_asset(self):
        super().load_asset()
        bps_path = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, f'bps_{self.num_bps:d}.pkl')
        bps_dict = pkl.load(open(bps_path, 'rb'))
        self.bps_points = torch.from_numpy(bps_dict['bps_points']).to(self.device)
        self.asset_bps = []

        for idx, file in enumerate(self.asset_files):
            bps = bps_dict[file]
            bps = torch.from_numpy(bps).float().to(self.device)
            self.asset_bps.append(bps)
        self.asset_bps = torch.stack(self.asset_bps, 0)

    def create_buffer(self):
        super().create_buffer()
        self.prop_buf   = torch.zeros((self.num_envs, self.num_prop_obs), device=self.device, dtype=torch.float32)
        self.obj_buf    = torch.zeros((self.num_envs, self.num_obj_obs), device=self.device, dtype=torch.float32)
        self.goal_buf   = torch.zeros((self.num_envs, self.num_goal_obs), device=self.device, dtype=torch.float32)
        self.amp_obs_buf = torch.zeros((self.num_envs, self.num_ref_obs_frames, self.num_ref_obs_per_frame), device=self.device,dtype=torch.float32)
        
        self.consecutive_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_carry_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    
    def eval_last_runs(self,env_ids):
        if env_ids is not None and len(env_ids) > 0:
            object_state = self._root_states[self.movetask_rootid[env_ids]]
            object_pos = object_state[:,0:3]
            object_rot = object_state[:,3:7]

            init_pos = self.init_trans[self.movetask_rootid[env_ids]]
            init_rot = self.init_rot[self.movetask_rootid[env_ids]]   

            geom_object_pcd = self.asset_pcd[self.object2asset[self.movetask_objectid[env_ids]]] * self.object2scale[self.movetask_objectid[env_ids]].unsqueeze(-1).unsqueeze(-1)
            global_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = object_pos, rot = object_rot)
            init_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = init_pos, rot = init_rot)
            init_z_pt = init_object_pcd[..., -1].min(1)[0]
            object_z_pt = global_object_pcd[...,-1].min(1)[0]

            success = (object_z_pt - init_z_pt) > 0.2        
            self.last_carry_success[env_ids] = success.float()

    def reset_env(self, env_ids):
        if env_ids is not None and len(env_ids) > 0:
            n = len(env_ids)
            rootmask = torch.isin(self.root2env, env_ids)
            objmask = torch.isin(self.object2env, env_ids)
            self._root_states[rootmask, 0:3] = self.init_trans[rootmask]
            self._root_states[rootmask, 3:7] = self.init_rot[rootmask]
            self._root_states[rootmask, 7:] = 0.

            ############# AMP
            motion_ids = self.sample_motion_ids(n, samp=True)
            ref_motion_end_frame = torch.randint(90,size=env_ids.shape).to(self.device).float()
            ref_motion_end_time = ref_motion_end_frame / self.query_motion_fps(motion_ids)
            ref_motion_time = ref_motion_end_time.unsqueeze(1).tile(self.num_ref_obs_frames) - torch.arange(self.num_ref_obs_frames).to(self.device) * self.dt
            ref_motion_time = ref_motion_time.flatten()
            motion_ids = torch.repeat_interleave(motion_ids, self.num_ref_obs_frames,)
            state_info = self.query_motion_state(motion_ids, ref_motion_time)
            
            ## transform state
            state_info['rigid_body_pos'] = state_info['rigid_body_pos'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_rot'] = state_info['rigid_body_rot'].view(n, self.num_ref_obs_frames, self.num_body, 4)
            state_info['rigid_body_vel'] = state_info['rigid_body_vel'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_anv'] = state_info['rigid_body_anv'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            now_pos = self._root_states[self.robot2root[env_ids], :3]
            now_pos[:,2] = state_info['rigid_body_pos'][:,0,0,2]
            delta_rot = torch_utils.quat_mul(
                self._root_states[self.robot2root[env_ids], 3:7], 
                torch_utils.calc_heading_quat_inv(state_info['rigid_body_rot'][:,0,0,:])
                )
            now_rot = torch_utils.quat_mul(delta_rot, state_info['rigid_body_rot'][:,0,0,:])
            now_vel = torch_utils.quat_apply(delta_rot, state_info['rigid_body_vel'][:,0,0,:])
            now_anv = torch_utils.quat_apply(delta_rot, state_info['rigid_body_anv'][:,0,0,:])

            delta_rot_exp = delta_rot.unsqueeze(1).unsqueeze(1).tile(1,self.num_ref_obs_frames, self.num_body,1)
            state_info['rigid_body_pos'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_pos'])
            state_info['rigid_body_rot'] = torch_utils.quat_mul(delta_rot_exp, state_info['rigid_body_rot'])
            state_info['rigid_body_vel'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_vel'])
            state_info['rigid_body_anv'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_anv'])
            delta_pos = now_pos - state_info['rigid_body_pos'][:,0,0,:]
            state_info['rigid_body_pos'] = state_info['rigid_body_pos'] + delta_pos.unsqueeze(1).unsqueeze(1)
            
            ## reset sim state
            self._root_states[self.robot2root[env_ids], 0:3] = now_pos
            self._root_states[self.robot2root[env_ids], 3:7] = now_rot
            self._root_states[self.robot2root[env_ids], 7:10] = now_vel
            self._root_states[self.robot2root[env_ids], 10:13] = now_anv
            self._robot_dof_pos[env_ids,:] = state_info['dof_pos'].view(n, self.num_ref_obs_frames, self.num_dof)[:, 0, :]
            self._robot_dof_vel[env_ids,:] = state_info['dof_vel'].view(n, self.num_ref_obs_frames, self.num_dof)[:, 0, :]

            self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self._root_states))
            self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self._dof_state))
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()


            ############# reset tensors
            self.progress_buf[env_ids]  = 0 
            self.reset_termination_buf[env_ids] = 0.
            self.reset_timeout_buf[env_ids] = 0.
            self.timeout_limit[env_ids] = self.cfg.max_episode_length
            self.consecutive_success[env_ids] = 0.

            self.movenow_guideidx[env_ids] = 0.
            self.movenow_preguidecomplete[env_ids] = False
            self.movenow_guide[env_ids] = self.preguide_full[env_ids, 0,]

            ## Note: RB buffer is not flushed without phy step
            rb_state = torch.cat([
                state_info['rigid_body_pos'][:,0],
                state_info['rigid_body_rot'][:,0],
                state_info['rigid_body_vel'][:,0],
                state_info['rigid_body_anv'][:,0],
            ], dim=-1)
            self._rigid_body_state[self.robot2rb[env_ids]] = rb_state
            self.compute_observation(env_ids)
            

            ############# reset AMP
            obj_pos = torch.repeat_interleave(self._root_states[self.movetask_rootid[env_ids], 0:3], self.num_ref_obs_frames, dim=0)
            amp_demo_obs = self.compute_ref_frame_obs(
                root_pos=state_info['rigid_body_pos'][:,:,0,:].view(-1,3),
                root_rot=state_info['rigid_body_rot'][:,:,0,:].view(-1,4),
                root_vel=state_info['rigid_body_vel'][:,:,0,:].view(-1,3),
                root_anv=state_info['rigid_body_anv'][:,:,0,:].view(-1,3),
                dof_pos=state_info['dof_pos'],
                dof_vel=state_info['dof_vel'],
                key_body_pos=state_info['rigid_body_pos'].view(-1,self.num_body,3)[:,self.amp_body_idx,:],
                obj_pos=obj_pos
            ).view(n, self.num_ref_obs_frames, self.num_ref_obs_per_frame)
            self.amp_obs_buf[env_ids] = amp_demo_obs
        
    def post_physics_step(self):
        super().post_physics_step()
        self.update_amp_obs()

    def update_amp_obs(self):
        for j in range(self.num_ref_obs_frames - 1, 0, - 1):
            self.amp_obs_buf[:, j, :] = self.amp_obs_buf[:, j - 1, :]
        robot_rb_state = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)
        object_state = self._root_states[self.movetask_rootid]
        amp_obs = self.compute_ref_frame_obs(
            root_pos=robot_rb_state[:, 0, 0:3],
            root_rot=robot_rb_state[:, 0, 3:7],
            root_vel=robot_rb_state[:, 0, 7:10],
            root_anv=robot_rb_state[:, 0, 10:13],
            dof_pos=self._robot_dof_pos,
            dof_vel=self._robot_dof_vel,
            key_body_pos=robot_rb_state[:,self.amp_body_idx,:3],
            obj_pos     =object_state[:, 0:3],
        )
        self.amp_obs_buf[:, 0, :] = amp_obs


    def fetch_amp_demo(self,n):
        assert self.motion is not None
        motion_ids = self.sample_motion_ids(n)

        motion_times = self.sample_motion_time(motion_ids = motion_ids)
        motion_times = motion_times.unsqueeze(1).tile(self.num_ref_obs_frames)
        motion_times -= torch.arange(self.num_ref_obs_frames).to(motion_times.device) * self.dt
        motion_times = motion_times.flatten()
        motion_ids = torch.repeat_interleave(motion_ids, self.num_ref_obs_frames,)
        
        state_info = self.query_motion_state(motion_ids, motion_times)
        amp_demo_obs = self.compute_ref_frame_obs(
            root_pos=state_info['rigid_body_pos'][:,0,:],
            root_rot=state_info['rigid_body_rot'][:,0,:],
            root_vel=state_info['rigid_body_vel'][:,0,:],
            root_anv=state_info['rigid_body_anv'][:,0,:],
            dof_pos=state_info['dof_pos'],
            dof_vel=state_info['dof_vel'],
            key_body_pos=state_info['rigid_body_pos'][:,self.amp_body_idx,:],
            obj_pos=state_info['object_pos'],
        )
        amp_demo_obs = amp_demo_obs.reshape((n, self.num_ref_obs_frames * self.num_ref_obs_per_frame))
        return amp_demo_obs
    



    def compute_observation(self, env_ids = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        n = len(env_ids)
        robot_rb_state = self._rigid_body_state[self.robot2rb[env_ids]].view(n, self.num_body, 13)
        object_state = self._root_states[self.movetask_rootid[env_ids]]
        self.prop_buf[env_ids] = self.compute_prop_obs(
            body_pos    =robot_rb_state[:, :, 0:3],
            body_rot    =robot_rb_state[:, :, 3:7],
            body_vel    =robot_rb_state[:, :, 7:10],
            body_ang_vel=robot_rb_state[:, :, 10:13],
        )
        self.obj_buf[env_ids] = self.compute_object_obs(
            root_pos    =robot_rb_state[:, 0, 0:3],
            root_rot    =robot_rb_state[:, 0, 3:7],
            obj_pos     =object_state[:, 0:3],
            obj_rot     =object_state[:, 3:7],
            obj_vel     =object_state[:, 7:10],
            obj_anv     =object_state[:, 10:13],
        )
        self.goal_buf[env_ids]=self.compute_goal_obs(
            root_pos    =robot_rb_state[:, 0, 0:3],
            root_rot    =robot_rb_state[:, 0, 3:7],
            goal_pos    =self.goal_trans[self.movetask_rootid[env_ids]],
            goal_rot    =self.goal_rot[self.movetask_rootid[env_ids]],
        )
        self.guide_buf[env_ids]=self.compute_guide_obs(
            root_pos    =robot_rb_state[:, 0, 0:3],
            root_rot    =robot_rb_state[:, 0, 3:7],
            guide_pos   =self.movenow_guide[env_ids]
        )



    def compute_reward(self):
        robot_rb_state = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)
        object_state = self._root_states[self.movetask_rootid]
        object_pos = object_state[:,0:3]
        object_rot = object_state[:,3:7]
        object_vel = object_state[:,7:10]
        object_z_rt = object_pos[:,2]

        init_pos = self.init_trans[self.movetask_rootid]
        init_rot = self.init_rot[self.movetask_rootid]   
        init_z_rt = init_pos[:,2]

        goal_pos = self.goal_trans[self.movetask_rootid]
        goal_rot = self.goal_rot[self.movetask_rootid]   
        goal_z_rt = goal_pos[:,2]

        robot2guide_dir = self.movenow_guide[:, :2] - robot_rb_state[:, 0, :2]
        robot2guide_dist = torch.norm(robot2guide_dir, dim=-1)
        robot2object_dir = object_pos[:,:2] - robot_rb_state[:, 0, :2]
        robot2object_dist = torch.norm(robot2object_dir, dim=-1)

        geom_object_pcd = self.asset_pcd[self.object2asset[self.movetask_objectid]] * self.object2scale[self.movetask_objectid].unsqueeze(-1).unsqueeze(-1)
        global_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = object_pos, rot = object_rot)
        init_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = init_pos, rot = init_rot)
        init_z_pt = init_object_pcd[..., -1].min(1)[0]
        goal_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = goal_pos, rot = goal_rot)
        goal_z_pt = goal_object_pcd[..., -1].min(1)[0]
        object_z_pt = global_object_pcd[...,-1].min(1)[0]
        
        hand_pos = robot_rb_state[:, self.hand_index, :3]
        hand2object_dist = hand_pos.unsqueeze(2) - global_object_pcd.unsqueeze(1)
        hand2object_dist = torch.norm(hand2object_dist, p=2, dim = -1)
        hand2object_dist = torch.min(hand2object_dist, dim=-1)[0]

        thresh_robot2object = 0.5

        ######## approach object
        robot_vel_xy = robot_rb_state[:, 0, 7:9]
        robot2target_dir = torch_utils.normalize(torch.where(self.movenow_preguidecomplete.unsqueeze(-1).tile(2), robot2object_dir, robot2guide_dir))
        robot2target_speed = 1.5
        reward_robot2object_vel = torch.exp(-2 * torch.square(robot2target_speed - torch.sum(robot2target_dir * robot_vel_xy, dim=-1)))
        reward_robot2object_pos = torch.exp(-0.5 * robot2object_dist)

        reward_robot2object_vel[robot2object_dist < thresh_robot2object] = 1
        reward_robot2object_pos[robot2object_dist < thresh_robot2object] = 1

        ######## grasp object
        reward_hand2object = torch.exp(- 5 *  hand2object_dist).mean(-1)
        reward_hand2object[robot2object_dist > 0.5] = 0.
        carry_height = torch.zeros_like(init_z_rt)
        carry_height[goal_z_rt < init_z_rt + self.cfg.carry_height] = self.cfg.carry_height
        target_z_rt = torch.maximum(init_z_rt, goal_z_rt) + carry_height
        reward_height_rt = (torch.minimum(object_z_rt, target_z_rt) - init_z_rt ) / (target_z_rt - init_z_rt)
        
        target_z_pt = torch.maximum(init_z_pt, goal_z_pt) + carry_height
        reward_height_pt = (torch.minimum(object_z_pt, target_z_pt) - init_z_pt ) / (target_z_pt - init_z_pt)
        
        # reward_vel = 1 - (torch.clamp(torch.norm(object_vel, dim=-1),0,1) / 1 - 1) ** 2
        # reward_vel[reward_height_pt > 0.3] = 1.
        
        goal_up_pos = init_pos.clone()
        goal_up_pos[:,2] = target_z_rt
        obj2goal_dist = torch.norm(goal_up_pos - object_pos, p=2, dim = -1)
        reward_goal_pos = torch.exp(- 5 *  obj2goal_dist)


        reward_items = [
            [0.2,   reward_robot2object_vel],
            [0.1,   reward_robot2object_pos],
            [0.2,   reward_hand2object],
            [0.2,   reward_height_rt],
            [0.2,   reward_height_pt],
            # [0.0,   reward_vel],
            [0.1,   reward_goal_pos],
        ]

        reward = sum([a * b for a,b in reward_items])
        self.reward_buf[:] = reward

        self.stats_step['robot2object_dist'] = robot2object_dist
        self.stats_step['hand2object_dist'] = hand2object_dist
        self.stats_step['reward_robot2object_vel'] = reward_robot2object_vel
        self.stats_step['reward_robot2object_pos'] = reward_robot2object_pos
        self.stats_step['reward_hand2object'] = reward_hand2object
        self.stats_step['reward_height_pt'] = reward_height_pt
        self.stats_step['reward_height_rt'] = reward_height_rt
        self.stats_step['reward_goal_pos'] = reward_goal_pos

    def export_logging_stats(self):
        stats = {}
        stats['progress'] = self.progress_buf.float().mean().item()
        stats['success'] = self.last_carry_success.float().mean().item()
        return stats

    def export_stats(self):
        stats = {}
        stats['env/progress'] = self.progress_buf.float().mean().item()
        stats['env/termination'] = self.reset_termination_buf.float().mean().item()
        stats['env/success'] = self.last_carry_success.float().mean().item()
        for k,v in self.stats_step.items():
            stats[f'env/{k}'] = v.mean().item()
        return stats


    def compute_termination(self):
        if self.enable_early_termination:
            force       = self._contact_force_state[self.robot2rb].view(self.num_envs, self.num_body, 3)
            fall_force  = torch.any(torch.abs(force) > 0.1, dim = -1)
            height      = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)[:,:,2]
            fall        = torch.logical_and(fall_force, height < self.fall_thresh)
            fall[:, self.ignore_contact_idx] = False
            fall = torch.any(fall, dim= -1)
            terminate = fall
        else:
            terminate = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        terminate[self.progress_buf <= 2] = False
        timeout = self.progress_buf >= self.timeout_limit
        
        
        reward_height = self.stats_step['reward_height_pt'] 
        movenow_success = reward_height > 0.8
        self.consecutive_success[~movenow_success] = 0
        self.consecutive_success[movenow_success] += 1

        if self.enable_early_termination:
            timeout = torch.logical_or(timeout, self.consecutive_success > self.consecutive_success_thresh)


        
        self.reset_timeout_buf[:] = timeout[:].float()
        self.reset_termination_buf[:] = terminate[:].float()



    