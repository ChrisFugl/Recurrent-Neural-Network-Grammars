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

def get_config(load_dir, name):
    absolute_load_dir = hydra.utils.to_absolute_path(load_dir)
    config_path = os.path.join(absolute_load_dir, f'{name}.yaml')
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)
    config = OmegaConf.create(config_dict)
    return config

def get_training_config(load_dir):
    """
    :type load_dir: str
    :rtype: OmegaConf
    """
    return get_config(load_dir, '.hydra/config')

def batched_index_select(inputs, indices):
    """
    :type inputs: torch.Tensor
    :type indices: torch.Tensor
    :rtype: torch.Tensor
    """
    batch_size = inputs.size(1)
    hidden_shape = inputs.shape[2:]
    indices_expanded = indices.view(1, batch_size, *[1] * len(hidden_shape))
    indices_expanded = indices_expanded.expand(1, batch_size, *hidden_shape)
    selected = torch.gather(inputs, 0, indices_expanded)
    return selected
