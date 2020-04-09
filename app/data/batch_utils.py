from app.constants import PAD_INDEX
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence

def map_sequences(converter, sequences):
    """
    :type sequences: list of list of object
    :rtype: list of list of object
    """
    converted_sequences = []
    for sequence in sequences:
        converted_sequence = list(map(converter, sequence))
        converted_sequences.append(converted_sequence)
    return converted_sequences

def int_sequence2tensor(device, int_sequence):
    """
    :type device: torch.device
    :type int_sequence: list of int
    :rtype: torch.Tensor
    """
    sequence_length = len(int_sequence)
    shape = (sequence_length,)
    return torch.tensor(int_sequence, device=device, dtype=torch.long).reshape(shape)

def int_sequences2tensor(device, int_sequences):
    """
    :type device: torch.device
    :type int_sequences: list of list of int
    :rtype: torch.Tensor
    """
    tensors = list(map(partial(int_sequence2tensor, device), int_sequences))
    padded = pad_sequence(tensors, batch_first=False, padding_value=PAD_INDEX)
    return padded

def sequences2tensor(device, to_int, sequences):
    """
    :type device: torch.device
    :type sequences: list of list of object
    :rtype: torch.Tensor
    """
    int_sequences = map_sequences(to_int, sequences)
    return int_sequences2tensor(device, int_sequences)

def sequences2lengths(device, sequences):
    """
    :type device: torch.device
    :type sequences: list of list of object
    :rtype: torch.Tensor
    """
    return torch.tensor(list(map(len, sequences)), dtype=torch.long, device=device)
