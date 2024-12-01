import torch
import torch.nn as nn
from ..common.runningmeanstd import RunningMeanStd
import numpy as np


class BaseNetwork(nn.Module):
    def build_mlp(self,in_dim,hiddens,activation='relu',last_activation = True, last_bias = True):
        net = nn.Sequential()
        for i, h in enumerate(hiddens):
            bias = not ( i == len(hiddens) - 1 and (not last_bias))
            net.append(nn.Linear(in_dim, h ,bias=bias))
            net.append(self.build_activation(activation))
            in_dim = h 
        if not last_activation:
            net.pop(-1)
        return net
        

    def build_activation(self,name):
        if name is None:
            return nn.Identity()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'softmax':
            return nn.Softmax(dim=-1)
        elif name == 'softplus':
            return nn.Softplus()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        raise NotImplementedError(name)


    def neglogp(self, x,mean,std,logstd):
        neglogp = 0.5 * (((x - mean) / std)**2)  + 0.5 * np.log(2.0 * np.pi)  + logstd
        neglogp = neglogp.sum(-1)
        return neglogp

    def sync_stats(self):
        for name, module in self.named_modules():
            if isinstance(module, (RunningMeanStd,)):
                module.sync_stats()
