import torch



def build_optimizer(cfg,params):
    if cfg.name == 'Adam':
        return torch.optim.Adam(
            params,
            lr = float(cfg.base_lr),
            betas = cfg.betas,
            weight_decay = cfg.weight_decay,
            eps = float(cfg.eps)
        )
    elif cfg.name == 'AdamW':
        return torch.optim.AdamW(
            params,
            lr = float(cfg.base_lr),
            betas = cfg.betas,
            weight_decay = cfg.weight_decay,
            eps = float(cfg.eps)
        )
    else:
        raise NotImplementedError(cfg.name)
    
