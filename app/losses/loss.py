from torch import nn

class Loss(nn.Module):
    """
    Abstract class that all losses must subclass.
    """

    def forward(self, predictions, groundtruths, lengths):
        """
        :param predictions: batch of log probabilities of each action (size S x B x A)
        :param groundtruths: batch of groundtruth actions (size S x B)
        :type predictions: torch.Tensor
        :type groundtruths: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
