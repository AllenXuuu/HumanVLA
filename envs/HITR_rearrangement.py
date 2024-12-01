from isaacgym import gymapi, gymtorch
import torch,os,json
import numpy as np
from utils import torch_utils
import pickle as pkl
from .HITR_carry import HITRCarryEnv
from collections import OrderedDict
import open3d as o3d
import trimesh
import imageio



class HITRRearrangementEnv(HITRCarryEnv):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def eval_last_runs(self,env_ids):
        if env_ids is not None and len(env_ids) > 0:
            object_state = self._root_states[self.movetask_rootid[env_ids]]
            object_pos = object_state[:,0:3]
            goal_trans = self.goal_trans[self.movetask_rootid[env_ids]]

            obj2goal_dist = torch.norm(object_pos - goal_trans, dim = -1)
            
            success = obj2goal_dist < self.eval_success_thresh        
            self.last_carry_success[env_ids] = success.float()


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
        
        hand_pos = robot_rb_state[:, self.hand_index, :3]
        hand2object_dist = hand_pos.unsqueeze(2) - global_object_pcd.unsqueeze(1)
        hand2object_dist = torch.norm(hand2object_dist, p=2, dim = -1)
        hand2object_dist = torch.min(hand2object_dist, dim=-1)[0]



        objguide = torch.where(self.movenow_preguidecomplete.unsqueeze(-1).tile(3), self.movenow_guide, self.postguide_full[:, 0])
        object2guide_dir = objguide - object_pos
        object2guide_dist = torch.norm(object2guide_dir, dim = -1)

        object2goal_dir = goal_pos - object_pos
        object2goal_dist = torch.norm(object2goal_dir, dim = -1)


        thresh_robot2object = 0.5
        thresh_object2goal = 0.3

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
        carry_height = self.cfg.carry_height
        target_z_rt = torch.maximum(init_z_rt, goal_z_rt) + carry_height
        reward_height_rt = (torch.minimum(object_z_rt, target_z_rt) - init_z_rt ) / (target_z_rt - init_z_rt)
        reward_height_rt[torch.logical_and(init_z_rt < 0.1, goal_z_rt < 0.1)] = 1.

        reward_height_rt        [object2goal_dist < thresh_object2goal] = 1.
        reward_hand2object      [object2goal_dist < thresh_object2goal] = 1.
        reward_robot2object_pos [object2goal_dist < thresh_object2goal] = 1.
        reward_robot2object_vel [object2goal_dist < thresh_object2goal] = 1.

        ######## rearrange
        object_speed = 1.5
        object2target_dir = torch_utils.normalize(object2guide_dir)
        reward_object2goal_vel = torch.exp(-2 * torch.square(object_speed - torch.sum(object2target_dir * object_vel, dim=-1)))
        reward_object2goal_vel[object2goal_dist < thresh_object2goal] = 1.
        
        
        # object pos
        reward_object2goal_pos_near = torch.exp(-5 * object2goal_dist)
        reward_object2goal_pos_far  = torch.exp(-1 * object2guide_dist)
        object2goal_rot         = self.calc_diff_rot(goal_rot, object_rot)
        reward_object2goal_rot  = torch.exp(-2 * object2goal_rot)


        reward_items = [
            [0.1,   reward_robot2object_vel],
            [0.1,   reward_robot2object_pos],
            [0.1,   reward_hand2object],
            [0.1,   reward_height_rt],

            [0.2,   reward_object2goal_vel],
            [0.2,   reward_object2goal_pos_far],
            [0.1,   reward_object2goal_pos_near],
            [0.1,   reward_object2goal_rot],

        ]

        reward = sum([a * b for a,b in reward_items])
        self.reward_buf[:] = reward

        self.stats_step['robot2object_dist']        = robot2object_dist
        self.stats_step['hand2object_dist']         = hand2object_dist
        self.stats_step['object2goal_dist']         = object2goal_dist
        self.stats_step['object2goal_rot']         = object2goal_rot

        self.stats_step['reward_robot2object_vel']  = reward_robot2object_vel
        self.stats_step['reward_robot2object_pos']  = reward_robot2object_pos
        self.stats_step['reward_hand2object']       = reward_hand2object
        self.stats_step['reward_height']            = reward_height_rt

        self.stats_step['reward_object2goal_vel']       = reward_object2goal_vel
        self.stats_step['reward_object2goal_pos_far']   = reward_object2goal_pos_far
        self.stats_step['reward_object2goal_pos_near']  = reward_object2goal_pos_near
        self.stats_step['reward_object2goal_rot']       = reward_object2goal_rot


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
        

        object_state = self._root_states[self.movetask_rootid]
        object_pos = object_state[:,0:3]
        goal_trans = self.goal_trans[self.movetask_rootid]
        obj2goal_dist = torch.norm(object_pos - goal_trans, dim = -1)        
        movenow_success = obj2goal_dist < (self.eval_success_thresh * 0.5)   

        self.consecutive_success[~movenow_success] = 0
        self.consecutive_success[movenow_success] += 1

        if self.enable_early_termination:
            timeout = torch.logical_or(timeout, self.consecutive_success > self.consecutive_success_thresh)
        
        self.reset_timeout_buf[:] = timeout[:].float()
        self.reset_termination_buf[:] = terminate[:].float()