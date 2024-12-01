import os
import time
from isaacgym import gymtorch, gymapi,gymutil

import torch
import numpy as np
from copy import deepcopy

from collections import deque

import sys
import pickle as pkl
import yaml

class BaseEnv():
    def __init__(self,cfg) -> None:
        self.gym = gymapi.acquire_gym()

        self.cfg = cfg
        self.num_envs = self.cfg.num_envs
        self.num_action = self.cfg.num_action

        self.sim_device = cfg.sim_device
        self.sim_device_type = cfg.sim_device_type
        self.sim_device_id = cfg.compute_device_id
        self.device = self.sim_device
        if not cfg.sim.use_gpu_pipeline:
            self.device = 'cpu'
        self.graphics_device_id = cfg.graphics_device_id
        self.headless = (self.graphics_device_id == -1)
        
        self.up_axis = cfg.sim.up_axis
        self.sim_params = self.parse_sim_params(cfg.sim)
        self.dt = self.cfg.control_freq_inv * self.sim_params.dt
        self.control_freq_inv = self.cfg.control_freq_inv
        self.sim = self.create_sim()
        self.viewer = self.create_viewer()
        self.create_env()
        self.enable_viewer_sync = True
        self.create_buffer()
    
    def create_env():
        raise NotImplementedError

    @property
    def obs_space(self):
        return {}
    
    
    def create_buffer(self):
        pass


    def parse_sim_params(self, sim_cfg) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

    
        if sim_cfg.up_axis not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {sim_cfg.up_axis}"
            print(msg)
            raise ValueError(msg)
        if sim_cfg.up_axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0,0,-9.8)
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y
            sim_params.gravity = gymapi.Vec3(0,-9.8,0)

        # assign general sim parameters
        sim_params.dt                   = sim_cfg.dt
        sim_params.num_client_threads   = sim_cfg.num_client_threads
        sim_params.use_gpu_pipeline     = sim_cfg.use_gpu_pipeline
        sim_params.substeps             = sim_cfg.substeps


        # configure physics parameters
        if self.cfg.physics_engine == gymapi.SIM_PHYSX:
            gymutil.parse_physx_config(sim_cfg.physx,sim_params)
        elif self.cfg.physics_engine == gymapi.SIM_FLEX:
            gymutil.parse_flex_config(sim_cfg.flex,sim_params)
        else:
            raise NotADirectoryError(sim_cfg.physics_engine)
        # return the configured params
        return sim_params

    def create_sim(self):
        sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.cfg.physics_engine, self.sim_params)
        if sim is None:
            print("*** Failed to create sim ***")
            exit(0)
        return sim
    

    def create_viewer(self):
        if self.headless :
            return None
        
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")


        cam_pos = gymapi.Vec3(*self.cfg.camera.pos)
        cam_tgt = gymapi.Vec3(*self.cfg.camera.tgt)

        self.gym.viewer_camera_look_at(
            self.viewer, None, cam_pos, cam_tgt)
        return self.viewer
    
    def create_ground(self):
        plane_params = gymapi.PlaneParams()
        if self.up_axis == 'z':
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            self.up_axis_idx =2
        else:
            plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
            self.up_axis_idx =1
        plane_params.static_friction  = self.cfg.plane.staticFriction
        plane_params.dynamic_friction = self.cfg.plane.dynamicFriction
        plane_params.restitution      = self.cfg.plane.restitution
        self.gym.add_ground(self.sim, plane_params)
        pass


    def step(self, actions):
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        # if self.device == 'cpu':
        self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        return self.step_output()

    def reset(self):
        reset = torch.logical_or(self.reset_termination_buf == 1, self.reset_timeout_buf == 1)
        reset_env_ids = torch.where(reset)[0]
        # env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        return self.reset_env(reset_env_ids)

    def step_output(self):
        return self.obs_buf, self.reward_buf, self.reset_termination_buf, self.reset_timeout_buf, self.extra

    def reset_env(self,env_ids=None):
        raise NotImplementedError

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            # if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
        return

    def post_physics_step(self):
        raise NotImplementedError
    

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return
    
    