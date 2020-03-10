from torch import nn

class Loss(nn.Module):
    """
    Abstract class that all losses must subclass.
    """
    pass

    def forward(self, log_probs, actions_lengths):
        """
        :param log_probs: batch of log probabilities of each action (size S x B)
        :type log_probs: torch.Tensor
        :type actions_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
