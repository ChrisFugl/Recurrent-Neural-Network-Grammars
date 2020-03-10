from app.losses.loss import Loss
from app.losses.utils import negative_log_likelihood

class NegativeLogLikelihoodLoss(Loss):

    def __init__(self, device):
        """
        :type device: torch.device
        """
        super().__init__()
        self._device = device

    def forward(self, log_probs, lengths):
        """
        :param log_probs: batch of log probabilities of each action (size S x B)
        :type log_probs: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        return negative_log_likelihood(self._device, log_probs, lengths)
