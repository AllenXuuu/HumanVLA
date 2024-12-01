import torch
import torch.nn as nn
import copy
from datetime import datetime
import numpy as np
import os
import time
import yaml
from ..common.buffer import ExperienceBuffer, ReplayBuffer
from ..common.optimizer import build_optimizer
from utils.utils import build_logger,build_writer
import tqdm
from collections import defaultdict, OrderedDict
from utils.utils import is_main_proc
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .vla_network import VLANetwork
from .dagger_trainer import DaggerTrainer


class DaggerPlayer(DaggerTrainer):
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

        # cfg.network.obs_space = env.obs_space
        self.num_action = cfg.student_network.num_action = cfg.env.num_action
        self.num_envs    = cfg.env.num_envs
        self.action_clip = cfg.action_clip
        self.obs_clip    = cfg.obs_clip


        self.num_game   = cfg.num_game
    
        self.cfg = cfg
        self.device = cfg.device
        self.auto_mixed_precision = self.cfg.auto_mixed_precision


        self.logger = build_logger(
            verbose=is_main_proc(cfg),
            filepath=os.path.join(cfg.root,f'_log_rk{cfg.rank}.txt')
        )
        
        self.prop_dim = cfg.student_network.prop_dim = self.env.num_prop_obs
        self.text_dim = cfg.student_network.text_dim = self.env.num_text_obs
        self.student_network = VLANetwork(cfg.student_network).to(self.device).eval()
        self.img_h, self.img_w, self.image_transform = self.student_network.build_transform()
        
        self.logger.info('===============Image Transform=================')
        self.logger.info(self.image_transform)
        self.logger.info('===============Student Network=================')
        self.logger.info(self.student_network)

    @torch.no_grad()
    def run_game(self, game_idx):
        for step in range(self.cfg.env.max_episode_length):
            obs = self.env_reset()

            actions = []
            for index in range(0, obs['prop'].shape[0], self.cfg.test_bz):
                obs_batch = {
                    k : v[index : index + self.cfg.test_bz]
                    for k,v in obs.items()
                }
                obs_batch['image'] = self.image_transform(obs_batch['image'].float().permute(0,3,1,2)/255.)
                action_batch = self.get_student_action(obs_batch)
                actions.append(action_batch)
            actions = torch.cat(actions, 0)
            next_obs, task_reward, termination, timeout, next_info = self.env_step(actions) 
            
            env_info = self.env.export_logging_stats()
            report = ''
            for k,v in env_info.items():
                report += f'{k} {v:.3f}. '
            self.logger.info(f'Game {game_idx} Step {step}. {report}TaskReward {task_reward.mean().item():.3f}.')
        
        pass

    def run(self):
        self.logger.info('================== Start Playing ====================')
        game_loop = range(1, 1 + self.num_game)
        if self.cfg.use_tqdm:
            game_loop = tqdm.tqdm(game_loop, desc = 'Playing',ncols = 100)


        all_eval_info = defaultdict(list)
        for game_idx in game_loop:
            train_info = self.run_game(game_idx)
            if self.cfg.eval:
                eval_info = self.env.export_evaluation()
                for k,v in eval_info.items():
                    all_eval_info[k].append(v)
                print(eval_info)
        
        for k, v in all_eval_info.items():
            print(k, np.mean(v))
            