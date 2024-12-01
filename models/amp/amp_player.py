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

from .amp_network import AMP_NETWORK
from .amp_trainer import AMPTrainer


class AMPPlayer(AMPTrainer):
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

        cfg.network.obs_space = env.obs_space
        self.num_action  = cfg.network.num_action  = cfg.env.num_action
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

        self.num_amp_obs = cfg.network.num_amp_obs = cfg.env.num_ref_obs_frames * cfg.env.num_ref_obs_per_frame
        self.network = AMP_NETWORK(cfg.network).to(self.device).eval()
        self.logger.info(self.network)

    def get_action(self,obs,rand_prob=None):
        result = self.network.get_action(obs)
        return result['mu']

    def run_game(self, game_idx):
        for step in range(self.cfg.env.max_episode_length):
            obs = self.env_reset()
            action = self.get_action(obs)
            next_obs, task_reward, termination, timeout, next_info = self.env_step(action)       
            disc_reward = self.network.eval_disc_reward(next_info['amp_obs'])
            
            env_info = self.env.export_logging_stats()
            report = ''
            for k,v in env_info.items():
                report += f'{k} {v:.3f}. '
            self.logger.info(f'Game {game_idx} Step {step}. {report}TaskReward {task_reward.mean().item():.3f}. DiscReward {disc_reward.mean().item():.3f}.')
        
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
            