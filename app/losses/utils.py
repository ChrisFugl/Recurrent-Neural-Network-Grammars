import torch

def negative_log_likelihood(device, log_probs, lengths):
    """
    :type device: torch.device
    :type log_probs: torch.Tensor
    :type lengths: torch.Tensor
    :rtype: torch.Tensor
    """
    mask = get_nll_mask(device, log_probs, lengths)
    masked = log_probs * mask
    sum = masked.sum(dim=0)
    mean = sum.mean()
    return - mean

def get_nll_mask(device, log_probs, lengths):
    """
    :type device: torch.device
    :type log_probs: torch.Tensor
    :type lengths: torch.Tensor
    :rtype: torch.Tensor
    """
    max_length, batch_size = log_probs.shape
    batch_size = len(lengths)
    mask = torch.ones((max_length, batch_size), device=device)
    for i, length in enumerate(lengths):
        mask[length:, i] = 0
    return mask
