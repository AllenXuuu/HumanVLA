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


class AMPTrainer:
    def __init__(self, cfg,  env) -> None:
        self.cfg = cfg
        self.env = env

        cfg.network.obs_space = env.obs_space
        self.num_action  = cfg.network.num_action  = cfg.env.num_action
        self.num_envs    = cfg.env.num_envs
        self.horizon_length = cfg.horizon_length
        self.max_epoch   = cfg.max_epoch
        self.action_clip = cfg.action_clip
        self.obs_clip    = cfg.obs_clip
        
        self.gamma  = cfg.gamma
        self.tau    = cfg.tau
        self.loss   = cfg.loss
        self.reward = cfg.reward
        self.epsilon_clip = cfg.epsilon_clip
        self.num_iters = cfg.num_iters

        self.cfg = cfg
        self.device = cfg.device
        self.auto_mixed_precision = self.cfg.auto_mixed_precision


        self.logger = build_logger(
            verbose=is_main_proc(cfg),
            filepath=os.path.join(cfg.root,f'_log_rk{cfg.rank}.txt')
        )
        if is_main_proc(cfg) and (not cfg.test):
            self.writer = build_writer(
                log_dir=os.path.join(cfg.root,f'tb')
            )

        self.amp_buffer_size = int(np.ceil(cfg.amp_buffer_size / cfg.world_size))
        if cfg.debug:
            self.amp_buffer_size = self.amp_buffer_size // 100
        self.amp_fetch_demo_bz = cfg.amp_fetch_demo_bz

        self.num_amp_obs = cfg.network.num_amp_obs = cfg.env.num_ref_obs_frames * cfg.env.num_ref_obs_per_frame
        self.network = AMP_NETWORK(cfg.network).to(self.device)
        self.logger.info(self.network)
        if self.cfg.ddp:    
            self.ddp_network = DDP(self.network,device_ids=[self.cfg.rank])
        else:
            self.ddp_network = self.network

        self.rand_action_probs = torch.ones((self.num_envs),dtype=torch.float32,device=self.device)
        self.optimizer = build_optimizer(cfg.optimizer, self.ddp_network.parameters())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.auto_mixed_precision)


        exp_buffer_info_dict = {
            'action'     : dict(shape = (self.horizon_length, self.num_envs, self.num_action), dtype = torch.float32),
            'neglogp'    : dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'mu'         : dict(shape = (self.horizon_length, self.num_envs, self.num_action), dtype = torch.float32),
            'sigma'      : dict(shape = (self.horizon_length, self.num_envs, self.num_action), dtype = torch.float32),
            'value'      : dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'reward'     : dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'reward_task': dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'reward_disc': dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'termination': dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'timeout'    : dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'next_value' : dict(shape = (self.horizon_length, self.num_envs), dtype = torch.float32),
            'amp_obs'    : dict(shape = (self.horizon_length, self.num_envs, self.num_amp_obs), dtype = torch.float32),
        }
        for k, v in self.env.obs_space.items():
            assert k not in exp_buffer_info_dict
            if isinstance(v, int):
                v = (v,)
            exp_buffer_info_dict[k] = dict(shape = (self.horizon_length, self.num_envs) + v, dtype = torch.float32)
        self.experience_buffer = ExperienceBuffer(exp_buffer_info_dict,self.device)
        
        demp_buffer_info_dict = {
            'amp_demo_obs'      :  dict(shape = (self.amp_buffer_size, self.num_amp_obs))
        }
        self.demo_buffer = ReplayBuffer(demp_buffer_info_dict, self.device)
        rep_buffer_info_dict = {
            'amp_replay_obs'    :  dict(shape = (self.amp_buffer_size, self.num_amp_obs))
        }
        self.replay_buffer = ReplayBuffer(rep_buffer_info_dict, self.device)
        self.logger.info('================== Experience Buffer ====================')
        self.logger.info(self.experience_buffer)
        self.logger.info('================== Replay Buffer ====================')
        self.logger.info(self.demo_buffer)
        self.logger.info('================== Replay Buffer ====================')
        self.logger.info(self.replay_buffer)

        self.logger.info('================== Fill AMP Demo Buffer ====================')
        t = int(np.ceil(self.amp_buffer_size / self.amp_fetch_demo_bz))
        for _ in range(t):
            amp_demo_obs = self.env.fetch_amp_demo(self.amp_fetch_demo_bz)
            amp_demo_obs = torch.clamp(amp_demo_obs, - self.obs_clip, self.obs_clip)
            self.demo_buffer.store({
                'amp_demo_obs'   : amp_demo_obs,
            })

    def set_eval(self):
        self.ddp_network.eval()     

    def set_train(self):
        self.ddp_network.train()   


    def iterate_data(self, data_dict, num_iters, shuffle=True):
        randidx = {}
        for a in  set([v.shape[0] for k,v in data_dict.items()]):
            if shuffle:
                randidx[a] = torch.randperm(a , device=self.device)
            else:
                randidx[a] = torch.arange(a, device=self.device)

        for i in range(num_iters):
            out = {}
            for k,v in data_dict.items():
                start = v.shape[0] * i     // num_iters
                end   = v.shape[0] * (i+1) // num_iters
                idx = randidx[v.shape[0]][start : end]
                out[k] = v[idx]
            yield out
    
    def get_action(self,obs,rand_prob=None):
        result = self.network.get_action(obs)
        if rand_prob is not None:
            rand_action_mask = torch.bernoulli(rand_prob)
            result['action'][rand_action_mask == 0] = result['mu'][rand_action_mask == 0]
            result['rand_action_mask'] = rand_action_mask
        return result

    def env_step(self,action):
        action = torch.clamp(action, - self.action_clip, self.action_clip)
        obs, reward, termination, timeout, info = self.env.step(action)
        for k in ['obs']:
            obs[k] = torch.clamp(obs[k], - self.obs_clip, self.obs_clip)
        info['amp_obs'] = torch.clamp(info['amp_obs'], - self.obs_clip, self.obs_clip)
        return obs, reward, termination, timeout, info
        
    def env_reset(self):
        obs = self.env.reset()
        for k in ['obs']:
            obs[k] = torch.clamp(obs[k], - self.obs_clip, self.obs_clip)
        return obs

    def update_amp_buffer(self, batch_dict):
        amp_demo_obs = self.env.fetch_amp_demo(self.amp_fetch_demo_bz)
        amp_demo_obs = torch.clamp(amp_demo_obs, - self.obs_clip, self.obs_clip)
        self.demo_buffer.store({
            'amp_demo_obs'   : amp_demo_obs,
        })
        amp_replay_obs = batch_dict['amp_obs']
        if self.replay_buffer.count >= self.replay_buffer.size:
            keep_mask = torch.bernoulli(torch.zeros(amp_replay_obs.shape[0], device=amp_replay_obs.device) + self.cfg.amp_replay_keep_prob)
            amp_replay_obs = amp_replay_obs[keep_mask == 1]
        self.replay_buffer.store({
            'amp_replay_obs'   : amp_replay_obs,
        })

    def calc_adv(self, termination, timeout, values, rewards, next_values):
        lastgaelam = 0
        advs = torch.zeros_like(rewards)
        for t in reversed(range(self.horizon_length)):
            delta = rewards[t] + self.gamma * (1.0 - termination[t]) * next_values[t] - values[t]
            lastgaelam = delta + self.gamma * self.tau * (1.0 - timeout[t]) * lastgaelam
            advs[t] = lastgaelam
        return advs

    def dist_normalize_adv(self,values):
        mean = values.mean()
        sqaure_mean  = (values ** 2).mean()
        if self.cfg.ddp:
            mean_info = torch.tensor([mean.item(), sqaure_mean.item()], device = self.device,dtype=torch.float32)
            dist.all_reduce(mean_info,  op=dist.ReduceOp.AVG)
            mean, sqaure_mean = mean_info
        std = torch.sqrt(sqaure_mean - mean ** 2)
        values = (values - mean) / (std + 1e-8)
        return values

    def play_steps(self):
        self.set_eval()
        for n in range(self.horizon_length):
            obs = self.env_reset()
            
            action = self.get_action(obs, self.rand_action_probs)
            for k, v in obs.items():
                self.experience_buffer.update(k,        n,  v)
            self.experience_buffer.update('action',     n,  action['action'])
            self.experience_buffer.update('mu',         n,  action['mu'])
            self.experience_buffer.update('sigma',      n,  action['sigma'])
            self.experience_buffer.update('neglogp',    n,  action['neglogp'])
            self.experience_buffer.update('value',      n,  action['value'])
            
            next_obs, task_reward, termination, timeout, next_info = self.env_step(action['action'])
            
            next_val = self.network.eval_critic(next_obs)['value']
            disc_reward = self.network.eval_disc_reward(next_info['amp_obs'])
            disc_reward_thresh = task_reward.clone()
            disc_reward_thresh[disc_reward_thresh < self.reward.disc_thresh_min] =  self.reward.disc_thresh_min
            disc_reward = torch.clamp(disc_reward, torch.zeros_like(disc_reward), disc_reward_thresh)
            reward = disc_reward * self.reward.disc + task_reward * self.reward.task

            self.experience_buffer.update('termination',n,  termination)
            self.experience_buffer.update('timeout',    n,  timeout)
            self.experience_buffer.update('reward',     n,  reward)
            self.experience_buffer.update('reward_task',n,  task_reward)
            self.experience_buffer.update('reward_disc',n,  disc_reward)
            self.experience_buffer.update('next_value', n,  next_val)
            self.experience_buffer.update('amp_obs',    n,  next_info['amp_obs'])

        advantage = self.calc_adv(
            termination = self.experience_buffer.tensor_dict['termination'],
            timeout     = self.experience_buffer.tensor_dict['timeout'],
            values      = self.experience_buffer.tensor_dict['value'],
            rewards     = self.experience_buffer.tensor_dict['reward'],
            next_values = self.experience_buffer.tensor_dict['next_value'],
        )
        returns = self.experience_buffer.tensor_dict['value'] + advantage
        if self.cfg.normalize_adv:
            normalize_advantage = self.dist_normalize_adv(advantage)
        else:
            normalize_advantage = advantage
        batch_dict = {
            'advantage' : normalize_advantage,
            'return' : returns,
        }
        for k,v in self.experience_buffer.export().items():
            batch_dict[k] = v
        for k,v in batch_dict.items():
            assert v.shape[:2] == (self.horizon_length, self.num_envs)
            newshape = (self.horizon_length * self.num_envs,) + v.shape[2:]
            batch_dict[k] = v.view(newshape)
        return batch_dict

    def compute_loss(self, batch_dict):
        with torch.cuda.amp.autocast(enabled=self.auto_mixed_precision):
            amp_obs_pos = batch_dict['amp_demo_obs']
            amp_obs_neg = batch_dict['amp_obs']
            amp_obs_pos.requires_grad_(True)
            obs = {
                k : batch_dict[k] for k in self.env.obs_space
            }
            train_meta  = self.ddp_network(obs, batch_dict['action'], amp_obs_pos, amp_obs_neg)
            
            entropy = train_meta['entropy'].mean(0)
            #################################################################################################### actor loss
            ratio = torch.exp(batch_dict['neglogp'] - train_meta['neglogp'])
            surr1 = batch_dict['advantage'] * ratio
            surr2 = batch_dict['advantage'] * torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
            a_loss = torch.max(-surr1, -surr2)
            a_loss = a_loss.mean()
            a_clip_ratio = torch.abs(ratio - 1) > self.epsilon_clip
            a_clip_ratio = a_clip_ratio.float().mean()
            
            #################################################################################################### critic loss
            c_loss = (batch_dict['norm_return'] - train_meta['norm_value']).square()
            c_loss = c_loss.mean()
            
            #################################################################################################### bound mu loss
            mu_bound = self.action_clip
            mu_loss_high = torch.clamp_min(train_meta['mu'] - mu_bound,    0) ** 2
            mu_loss_low  = torch.clamp_max(train_meta['mu'] + mu_bound,    0) ** 2
            mu_loss  = mu_loss_high + mu_loss_low
            mu_loss  = mu_loss.mean()

            #################################################################################################### adversial loss
            amp_logit_pos = train_meta['amp_logit_pos']
            amp_logit_neg = train_meta['amp_logit_neg']
            
            ############################### adv. prediction
            disc_loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                amp_logit_pos, torch.ones_like(amp_logit_pos)
            )
            disc_loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(
                amp_logit_neg, torch.zeros_like(amp_logit_neg)
            )
            disc_prediction_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

            # ############################### adv. logit reg
            last_linear = self.network.disc_mlp[-1]
            assert isinstance(last_linear, torch.nn.Linear)
            disc_logit_loss = last_linear.weight.square().sum()
            
            ############################### adv. grad penalty
            disc_pos_grad = torch.autograd.grad(
                amp_logit_pos, amp_obs_pos, grad_outputs=torch.ones_like(amp_logit_pos), create_graph=True, retain_graph=True, only_inputs=True)
            disc_pos_grad = disc_pos_grad[0].square().sum(-1)
            disc_grad_penalty = disc_pos_grad.mean()

            ################################ adv. weight_decay
            disc_weights = []
            for m in self.network.disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    disc_weights.append(m.weight.flatten())
            disc_weights = torch.cat(disc_weights)
            disc_weight_decay = disc_weights.square().sum()

            disc_items    = [disc_prediction_loss,  disc_logit_loss,        disc_grad_penalty,      disc_weight_decay]
            disc_items_w  = [self.loss.disc_pred,   self.loss.disc_logit,   self.loss.disc_grad,    self.loss.disc_wd]
            disc_loss = sum([a * b for a,b in zip(disc_items,disc_items_w)])

            #################################################################################################### total loss
            loss_items   = [a_loss,             c_loss,             mu_loss,            -entropy,           disc_loss]
            loss_items_w = [self.loss.actor,    self.loss.critic,   self.loss.bound_mu, self.loss.entropy,  self.loss.disc]
            total_loss = sum([a * b for a,b in zip(loss_items,   loss_items_w)])
            

        step_result = {
            'total_loss'        : total_loss,
            'actor_loss'        : a_loss,
            'critic_loss'       : c_loss,
            'mu_loss'           : mu_loss,
            'entropy'           : entropy,
            'disc_loss'         : disc_loss,

            'disc_pred_loss'    : disc_prediction_loss,
            'disc_logit_loss'   : disc_logit_loss,
            'disc_grad_penalty' : disc_grad_penalty,
            'disc_weight_decay' : disc_weight_decay,

            'disc_pos_score'    : amp_logit_pos.float().mean(),
            'disc_neg_score'    : amp_logit_neg.float().mean(),
            'actor_clip'        : a_clip_ratio,
        }
        return step_result

    def train_epoch(self):
        play_start_time = time.time()
        with torch.no_grad():
            self.set_eval()
            data_dict = self.play_steps()
        play_time = time.time() - play_start_time

        train_time_start = time.time()
        self.set_train() 
        self.update_amp_buffer(data_dict)
        amp_obs = data_dict['amp_obs']
        amp_replay_obs = self.replay_buffer.sample(int(amp_obs.shape[0] * self.cfg.amp_replay_rate) )
        amp_replay_obs = amp_replay_obs['amp_replay_obs']
        amp_obs = torch.cat([amp_obs,amp_replay_obs],0)
        
        amp_demo_obs_dict   = self.demo_buffer.sample(amp_obs.shape[0])
        amp_demo_obs = amp_demo_obs_dict['amp_demo_obs']
        
        ########### normalize
        for key in self.env.obs_space:
            data_dict[key]            = self.network.normalize_obs(key, data_dict[key])
        data_dict['norm_value']     = self.network.normalize_value(data_dict['value'])
        data_dict['norm_return']    = self.network.normalize_value(data_dict['return'])
        data_dict['amp_obs']        = self.network.normalize_disc(amp_obs)
        data_dict['amp_demo_obs']   = self.network.normalize_disc(amp_demo_obs)
        
        if self.cfg.ddp:
            self.network.sync_stats()

        train_info = defaultdict(list)
        for ppo_epoch_idx in range(0, self.cfg.ppo_epoch):
            for batch_idx, batch_dict in enumerate(self.iterate_data(data_dict, self.num_iters)):
                loss_info_step = self.compute_loss(batch_dict)
                loss = loss_info_step['total_loss']

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.cfg.truncate_grads:
                    nn.utils.clip_grad_norm_(self.ddp_network.parameters(), self.cfg.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                for k,v in loss_info_step.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    train_info[k].append(v)
     
        for k,v in train_info.items():
            if k in ['actor_clip', 'actor_loss']:
                per_step_val = train_info[k]
                assert len(per_step_val) % self.cfg.ppo_epoch == 0
                train_info[k] = []
                for ppo_epoch_idx in range(0,self.cfg.ppo_epoch):
                    start_idx = len(per_step_val) * ppo_epoch_idx // self.cfg.ppo_epoch
                    end_idx   = len(per_step_val) * (ppo_epoch_idx + 1) // self.cfg.ppo_epoch
                    train_info[k].append(np.mean(per_step_val[start_idx : end_idx]))
            else:
                train_info[k] = np.mean(train_info[k])

        train_time = time.time() - train_time_start
        
        train_info.update({
            'lr'         : self.optimizer.param_groups[0]['lr'],
            'reward'     : data_dict['reward'].mean().item(),
            'reward_disc': data_dict['reward_disc'].mean().item(),
            'reward_task': data_dict['reward_task'].mean().item(),
            'time'  : {
                'train_time' : train_time,
                'play_time' : play_time
            }
        })
        return train_info 
        

    def run(self):
        self.logger.info('================== Start Training  ====================')
        epoch_loop = range(1, 1 + self.max_epoch)
        if self.cfg.use_tqdm:
            epoch_loop = tqdm.tqdm(epoch_loop, desc = 'Training',ncols = 100)

        best_reward = 0.
        for epoch in epoch_loop:
            train_info = self.train_epoch()
            train_info = dict(train_info)
            train_info['epoch'] = epoch

            ########################### writer
            if is_main_proc(self.cfg):
                self.update_tensorboard(train_info, self.env.export_stats(), epoch)

            ########################### logger
            self.update_logger(train_info, self.env.export_logging_stats(), epoch)
            
            ########################### save weight
            if epoch % self.cfg.save_epoch == 0:
                path = os.path.join(self.cfg.root, f'epoch_{epoch}.pth')
                self.logger.info(f'Save ckpt to === > {path}')
                if is_main_proc(self.cfg):
                    self.save_ckpt(path)
            
            if epoch > self.cfg.save_epoch and train_info['reward'] > best_reward:
                best_reward = train_info['reward']
                path = os.path.join(self.cfg.root, f'best.pth')
                self.logger.info(f'Save ckpt to === > {path}')
                if is_main_proc(self.cfg):
                    self.save_ckpt(path)

    def update_tensorboard(self, train_info, env_info, epoch):
        for key in ['total_loss', 'actor_loss','critic_loss','mu_loss','entropy','disc_loss',
                    'disc_pred_loss','disc_logit_loss','disc_grad_penalty','disc_weight_decay']:
            val = train_info[key]
            if isinstance(val, list):
                val = np.mean(val)
            self.writer.add_scalar(f'loss/{key}', val, epoch)
        
        for key,val in train_info['time'].items():
            self.writer.add_scalar(f'time/{key}', val, epoch)
        
        for key in ['reward', 'reward_disc', 'reward_task', 'lr', 'disc_pos_score', 'disc_neg_score']:
            self.writer.add_scalar(f'info/{key}', train_info[key], epoch)

        for ppo_epoch_idx in range(0,self.cfg.ppo_epoch):
            self.writer.add_scalar(f'actor_clip/ppo_epoch_{ppo_epoch_idx+1}_actor_clip', train_info['actor_clip'][ppo_epoch_idx], epoch)
            self.writer.add_scalar(f'actor_loss/ppo_epoch_{ppo_epoch_idx+1}_actor_loss', train_info['actor_loss'][ppo_epoch_idx], epoch)
        
        for key, value in env_info.items():
            self.writer.add_scalar(key, value, epoch)         
    
    def update_logger(self, train_info, env_info, epoch):
        timer_info = train_info['time']
        report = ''
        for k,v in env_info.items():
            report += f'{k} {v:.3f}. '
            
        self.logger.info(
            f'Epoch {epoch}. Time [PLAY={timer_info["play_time"]:.3f}, TR={timer_info["train_time"]:.3f}]. {report}Reward Disc {train_info["reward_disc"]:.3f}. Task {train_info["reward_task"]:.3f}. Loss {train_info["total_loss"]:.3f}')


    def load_ckpt(self,path):
        self.logger.info(f'Load Ckpt from <== {path}')
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt['weight'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer'])
        
    def save_ckpt(self,path):
        torch.save({
            'weight'        : self.network.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
        },path)