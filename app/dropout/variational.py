import torch

def variational_dropout_mask(shape, dropout, device=None):
    """
    :type shape: list of int
    :type dropout: float
    :type device: torch.device
    :rtype: torch.Tensor
    """
    mask = torch.empty(shape, dtype=torch.float, device=device)
    mask = mask.bernoulli_(1 - dropout)
    mask = mask / (1 - dropout)
    return mask
