from utils import config,utils
from envs.builder import build_env
from models.builder import build_model
import yaml,os
import torch
def main():
    args = config.build_args()
    cfg = config.build_config(args)
    
    if cfg.ddp:
        utils.init_ddp(cfg)
    utils.duplication_check(cfg)
    
    if utils.is_main_proc(cfg):
        with open(os.path.join(cfg.root, 'config.yaml'),'w') as f:
            yaml.dump(cfg, f)

    utils.set_np_formatting()
    utils.set_seed(cfg.seed)

    env = build_env(cfg.env)

    model = build_model(cfg, env)
    if cfg.ckpt is not None:
        model.load_ckpt(cfg.ckpt)
    model.run()
    
if __name__ == '__main__':
    main()