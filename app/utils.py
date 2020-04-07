from app.constants import PAD_INDEX
import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence
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

def padded_reverse(sequences, lengths):
    """
    Reverse a tensor according to the length of each sequence.

    :type sequences: torch.Tensor
    :type lengths: torch.Tensor
    :rtype: torch.Tensor
    """
    max_length = sequences.size(0)
    flipped = sequences.flip(dims=[0])
    flipped_sequences = [flipped[max_length - length:, i] for i, length in enumerate(lengths)]
    return pad_sequence(flipped_sequences, padding_value=PAD_INDEX)
