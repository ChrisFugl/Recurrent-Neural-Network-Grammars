from app.losses.loss import Loss
from app.losses.utils import get_nll_mask

class NegativeActionLogLikelihoodLoss(Loss):

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
        mask = get_nll_mask(self._device, log_probs, lengths)
        masked = log_probs * mask
        tree_ll = masked.sum(dim=0)
        action_ll = tree_ll / lengths.float()
        mean_action_ll = action_ll.mean()
        return - mean_action_ll
