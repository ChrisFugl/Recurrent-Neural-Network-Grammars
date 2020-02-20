from torch import nn

class Loss(nn.Module):
    """
    Abstract class that all losses must subclass.
    """
    pass

    def forward(self, actions_probabilities, actions_lengths):
        """
        :param actions_probabilities: batch of probabilities of each action (size S x B)
        :type actions_probabilities: torch.Tensor
        :type actions_lengths: list of int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
