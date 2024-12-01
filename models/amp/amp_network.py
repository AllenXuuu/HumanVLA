import torch
import torch.nn as nn
from ..common.runningmeanstd import RunningMeanStd
from ..common.base_network import BaseNetwork
import numpy as np

class AMP_NETWORK(BaseNetwork):
    def __init__(self, cfg) -> None:
        super().__init__()    
        self.cfg = cfg


        assert cfg.actor.hidden[-1] == cfg.num_action
        assert cfg.critic.hidden[-1] == 1
        assert cfg.disc.hidden[-1] == 1

        prop_dim = cfg.obs_space['obs']
        self.bps_pts, self.bps_dim = cfg.obs_space['bps']
        assert self.bps_dim == 3
        self.bps_dim = self.bps_pts * self.bps_dim
        
        self.bps_pre = self.build_mlp(self.bps_dim, cfg.bps_pre.hidden)
        if len(cfg.bps_pre.hidden) > 0:
            bps_feat_dim = cfg.bps_pre.hidden[-1]
        else:
            bps_feat_dim = self.bps_dim
        num_obs  = prop_dim + bps_feat_dim

        self.actor_mlp  = self.build_mlp(num_obs, cfg.actor.hidden,  last_activation=False)
        self.critic_mlp = self.build_mlp(num_obs, cfg.critic.hidden, last_activation=False)
        self.sigma = nn.Parameter(torch.zeros(cfg.num_action, requires_grad=True, dtype=torch.float32), requires_grad=False)

        amp_dim = cfg.num_amp_obs
        self.disc_mlp = self.build_mlp(amp_dim, cfg.disc.hidden, last_activation = False, last_bias=False)
        
        nn.init.constant_(self.sigma, self.cfg.sigma.init)
        for m in self.modules():         
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

        torch.nn.init.uniform_(self.disc_mlp[-1].weight, -1, 1)
        self.prop_normalizer    = RunningMeanStd(prop_dim)          if self.cfg.normalize_prop      else nn.Identity()
        self.bps_normalizer     = RunningMeanStd(3)                 if self.cfg.normalize_bps       else nn.Identity()
        self.val_normalizer     = RunningMeanStd(1)                 if self.cfg.normalize_value     else nn.Identity()
        self.disc_normalizer    = RunningMeanStd(amp_dim)           if self.cfg.normalize_amp       else nn.Identity()


    def normalize_obs(self, key, val):
        if key == 'obs':
            return self.prop_normalizer(val)
        elif key == 'bps':
            return self.bps_normalizer(val)
        else:
            raise NotImplementedError
    
    def normalize_value(self, val):
        return self.val_normalizer(val)
    
    def normalize_disc(self, val):
        return self.disc_normalizer(val)

    def compute_obs(self, obs, need_normalize = True):
        if need_normalize:
            prop = self.prop_normalizer(obs['obs'])
            bps = self.bps_normalizer(obs['bps']).view(-1, self.bps_dim)
        else:
            prop = obs['obs']
            bps = obs['bps'].view(-1, self.bps_dim)
        bps = self.bps_pre(bps)
        obs = torch.cat([prop, bps],dim=-1)

        return obs

    def forward(self, obs, action, amp_obs_pos, amp_obs_neg):
        ### train
        obs = self.compute_obs(obs, need_normalize=False)
        mu = self.actor_mlp(obs)
        logstd = self.sigma.expand_as(mu)
        std = torch.exp(logstd)
        distribution = torch.distributions.Normal(mu, std)
        entropy = distribution.entropy().sum(-1)
        neglogp = self.neglogp(action, mu, std, logstd)
        
        norm_value = self.critic_mlp(obs)
        amp_logit_pos = self.disc_mlp(amp_obs_pos)
        amp_logit_neg = self.disc_mlp(amp_obs_neg)
        return {
            'mu'            : mu,
            'sigma'         : std,
            'neglogp'       : neglogp,
            'entropy'       : entropy,
            'norm_value'    : norm_value.squeeze(-1),
            'amp_logit_pos' : amp_logit_pos.squeeze(-1),
            'amp_logit_neg' : amp_logit_neg.squeeze(-1),
        }


    def get_action(self,obs,need_normalize=True):
        obs = self.compute_obs(obs, need_normalize=need_normalize)
        mu = self.actor_mlp(obs)
        logstd = self.sigma.expand_as(mu)
        std = torch.exp(logstd)
        distribution = torch.distributions.Normal(mu, std)
        action = distribution.sample()
        neglogp = self.neglogp(action, mu, std, logstd)

        norm_value = self.critic_mlp(obs)
        if self.cfg.normalize_value:
            value = self.val_normalizer(norm_value.detach(), unnorm = True)
        else:
            value = norm_value
        return {
            'obs'           : obs,
            'action'        : action,
            'mu'            : mu,
            'sigma'         : std,
            'neglogp'       : neglogp,
            'value'         : value.squeeze(-1),
            'norm_value'    : norm_value.squeeze(-1)
        }
        

    
    def eval_critic(self,obs, need_normalize = True):
        obs = self.compute_obs(obs, need_normalize=need_normalize)
        norm_value = self.critic_mlp(obs)
        if self.cfg.normalize_value:
            value = self.val_normalizer(norm_value.detach(), unnorm = True)
        else:
            value = norm_value
        return {
            'value'   : value.squeeze(-1),
            'norm_value' : norm_value.squeeze(-1)
        }
    
    
    def eval_disc_reward(self, feat, need_normalize = True):
        if need_normalize:
            feat = self.disc_normalizer(feat)
        logits = self.disc_mlp(feat).squeeze(-1)
        prob = torch.sigmoid(logits)
        disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.01, device=prob.device)))
        return disc_r