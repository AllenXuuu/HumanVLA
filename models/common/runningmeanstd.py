import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import os
'''
updates statistic from a full data
'''
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        if isinstance(self.insize, int):
            self.insize = (self.insize,)
        self.epsilon = epsilon

        self.running_mean = nn.Parameter(
            torch.zeros(self.insize, dtype = torch.float32), requires_grad=False
        )
        self.running_var = nn.Parameter(
            torch.ones(self.insize, dtype = torch.float32), requires_grad=False
        )
        self.running_count = nn.Parameter(
            torch.zeros(1, dtype = torch.float32), requires_grad=False
        )

        self.trainable = True

    def freeze(self):
        self.trainable = False


    def forward(self, input, unnorm=False):
        input_shape = input.shape
        input = input.view((-1,) + self.insize)

        if self.training and self.trainable:
            assert not unnorm
            bz = input.shape[0]
            delta_mean = input.mean(0) - self.running_mean
            new_count = self.running_count + bz
            new_mean = self.running_mean + delta_mean * bz / new_count
            new_var  = self.running_var * self.running_count / new_count \
                        + input.var(0) * bz / new_count \
                        +  (delta_mean ** 2) * self.running_count * bz / (new_count ** 2)
            
            self.running_mean[:]    = new_mean
            self.running_var[:]     = new_var
            self.running_count[:]   = new_count

        current_mean = self.running_mean
        current_var = self.running_var
        if unnorm:
            y = torch.clamp(input, min = -5, max = 5)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)

        return y.view(input_shape)

    def sync_stats(self):
        dist.all_reduce(self.running_mean,  op=dist.ReduceOp.AVG)
        dist.all_reduce(self.running_var,   op=dist.ReduceOp.AVG)
        dist.all_reduce(self.running_count, op=dist.ReduceOp.AVG)