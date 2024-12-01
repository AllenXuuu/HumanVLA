from isaacgym import gymutil
import argparse
import os
import yaml
from easydict import EasyDict
import numpy as np
import random
import torch
import shutil
import logging
import tensorboardX

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def is_main_proc(cfg):
    return cfg.rank == 0

def init_ddp(cfg):
    torch.distributed.init_process_group("nccl", rank=cfg.rank, world_size=cfg.world_size)

def duplication_check(cfg):
    root = cfg.root
    if is_main_proc(cfg):
        if not os.path.exists(root):
            print(f'>>>> Task root: {root}. <<<<')
            os.makedirs(root)
        elif cfg.force:
            print(f'>>>> Override {root}. Clear All. <<<<')
            shutil.rmtree(root)
            os.makedirs(root)
        else:
            raise FileExistsError(
                f'Existing root: {root}. Use "--force" to override')
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    

def build_writer(log_dir):
    return tensorboardX.SummaryWriter(log_dir)

def build_logger(verbose=True, filepath=None):
    logger = logging.getLogger(name='logger')    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if verbose:
        handler1 = logging.StreamHandler()
        handler1.setLevel(logging.DEBUG)
        handler1.setFormatter(formatter)
        logger.addHandler(handler1)
    if filepath is not None:
        handler2 = logging.FileHandler(filename=filepath, mode='w')
        handler2.setFormatter(formatter)
        handler2.setLevel(logging.DEBUG)
        logger.addHandler(handler2)
    return logger


def set_seed(seed, torch_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


