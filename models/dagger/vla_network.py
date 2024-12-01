import torch
import torch.nn as nn
from ..common.runningmeanstd import RunningMeanStd
from ..common.base_network import BaseNetwork
import numpy as np
import torchvision

class ResBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.layers(x) + x
    

class VLANetwork(BaseNetwork):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.prop_dim = cfg.prop_dim
        self.text_dim = cfg.text_dim
        self.image_backbone = torchvision.models.efficientnet_b0()
        self.image_backbone.classifier = nn.Sequential()
        image_feat_dim = 1280
        self.image_pre = self.build_mlp(image_feat_dim, self.cfg.image_pre.hidden)
        self.text_pre = self.build_mlp(cfg.text_dim, self.cfg.text_pre.hidden)
        vl_dim = self.cfg.image_pre.hidden[-1] + self.cfg.text_pre.hidden[-1]
        self.prop_normalizer   = RunningMeanStd(self.prop_dim) if self.cfg.normalize_prop else nn.Identity()
        
        self.num_action = 28

        self.actor_mlp  = nn.Sequential(
            nn.Linear(vl_dim + self.prop_dim + self.num_action, 1024),
            ResBlock(1024),
            ResBlock(1024),
            nn.Linear(1024,28)
        )

    def get_action(self, obs):
        return self.forward(
            prop    = self.prop_normalizer(obs['prop']),
            img     = obs['image'],
            text    = obs['text'],
            last_action=obs['last_action']
        )

    def forward(self, prop, img, text, last_action):
        text = self.text_pre(text)
        img = self.image_pre(self.image_backbone(img))
        
        obs = torch.cat([prop, last_action, img, text], dim=1)
        
        action = self.actor_mlp(obs)
        return action
    

    def build_transform(self):
        height, width = 256, 256
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.Normalize(
                mean = (0.5, 0.5, 0.5),
                std  = (0.5, 0.5, 0.5),
            )
        ])
        return height, width, transform
    
    def normalize_prop(self,x):
        return self.prop_normalizer(x)