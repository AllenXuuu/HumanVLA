from isaacgym import gymutil,gymapi
import argparse
import os
import yaml
from easydict import EasyDict
import numpy as np
import random
import torch
from datetime import datetime

def build_args():
    custom_parameters = [
        # {"name": "--name",   "type": str,            "default": datetime.now().strftime("%Y%m%d_%H%M%S")},      
        {"name": "--name",              "type": str,            "default": None},  
        {"name": "--logging_prefix",    "type": str,          "default": "."},  
        {"name": "--data_prefix",       "type": str,          "default": "."},  
        {"name": "--cfg",           "type": str,            "default": None},   
        {"name": "--seed",          "type": int,            "default": 0},   
        {"name": "--force",         "action": "store_true", "default": False},
        {"name": "--device",        "type": int,            "default": None},
        {"name": "--ckpt",          "type": str,            "default": None},
        {"name": "--num_envs",      "type": int,            "default": None},
        {"name": "--debug",         "action": "store_true", "default": False},
        {"name": "--eval",          "action": "store_true", "default": False},
        {"name": "--test",          "action": "store_true", "default": False},
    ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        headless=True,
        no_graphics=True,
        custom_parameters=custom_parameters)

    return args


def build_config(args):
    with open(os.path.join(os.getcwd(), args.cfg), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg = EasyDict(cfg)
    cfg.name = args.name
    if args.name is None:
        cfg.root = None
    else:
        cfg.root = os.path.join(args.logging_prefix, 'logs', args.name)
    cfg.data_prefix = cfg.env.data_prefix = args.data_prefix
    cfg.logging_prefix = cfg.env.logging_prefix = args.logging_prefix
    
    cfg.force = args.force   
    cfg.seed  = args.seed
    
    #### ddp
    cfg.local_rank = cfg.env.local_rank     = int(os.environ.get('LOCAL_RANK',0))
    cfg.rank       = cfg.env.rank           = int(os.environ.get('RANK',0))
    cfg.world_size = cfg.env.world_size     = int(os.environ.get('WORLD_SIZE',1))
    cfg.seed += cfg.rank

    #### isaac gym params
    ################## Forbid other CLI override
    cfg.env.sim.use_gpu_pipeline = args.use_gpu_pipeline
    cfg.env.sim.physx.use_gpu = args.use_gpu
    cfg.env.physics_engine = {
        'physx' : gymapi.SIM_PHYSX,
        'flex' : gymapi.SIM_FLEX
    }[cfg.env.physics_engine]
    assert cfg.env.physics_engine == args.physics_engine

    cfg.env.sim_device = args.sim_device
    cfg.env.sim_device_type = args.sim_device_type
    cfg.env.compute_device_id = args.compute_device_id
    cfg.env.graphics_device_id = args.graphics_device_id
    if args.headless or args.nographics:
        cfg.env.graphics_device_id = -1

    ####
    # if cfg.env.graphics_device_id != -1:
    #     cfg.env.graphics_device_id = args.device
    cfg.ckpt = args.ckpt
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    
    if args.device is not None:
        args.device = [args.device]
    elif cfg.world_size > 1:
        args.device = list(range(cfg.world_size))
    else:
        args.device = [0]

    if cfg.env.sim_device_type =='cuda':
        cfg.env.compute_device_id =  args.device[cfg.local_rank]
        cfg.env.sim_device = f'cuda:{args.device[cfg.local_rank]}'

    cfg.ddp = cfg.world_size > 1
    cfg.device = f'cuda:{args.device[cfg.local_rank]}'
    cfg.env.sim_device = cfg.device

    cfg.debug = cfg.env.debug = args.debug
    cfg.eval = cfg.env.eval = args.eval
    cfg.test = cfg.env.test = args.test
    return cfg
