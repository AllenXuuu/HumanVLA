from .amp.amp_trainer import AMPTrainer
from .amp.amp_player import AMPPlayer
from .dagger.dagger_trainer import DaggerTrainer
from .dagger.dagger_player  import DaggerPlayer


def build_model(cfg, env):
    model_name = cfg.model
    if cfg.test:
        model_name = model_name + 'Player'
    else:
        model_name = model_name + 'Trainer'
    return eval(model_name)(cfg, env)