import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import random
import torch
import yaml

def get_device(gpu):
    """
    :type gpu: int
    :rtype: torch.device
    """
    if torch.cuda.is_available() and gpu is not None:
        return torch.device(gpu)
    else:
        return torch.device('cpu')

def set_seed(seed):
    """
    :type seed: int
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

def is_generative(type):
    """
    :type type: str
    :rtype: bool
    """
    return type == 'generative'

def get_training_config(load_dir):
    """
    :type load_dir: str
    :rtype: OmegaConf
    """
    absolute_load_dir = hydra.utils.to_absolute_path(load_dir)
    config_path = os.path.join(absolute_load_dir, '.hydra/config.yaml')
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)
    config = OmegaConf.create(config_dict)
    return config