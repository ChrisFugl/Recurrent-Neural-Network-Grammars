from app.losses.loss import Loss
import torch

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
        mask = self._get_mask(log_probs, lengths)
        masked = log_probs * mask
        sum = masked.sum(dim=0)
        normalized = sum / lengths.float()
        mean = normalized.mean()
        return - mean

    def _get_mask(self, log_probs, lengths):
        max_length, batch_size = log_probs.shape
        batch_size = len(lengths)
        mask = torch.ones((max_length, batch_size), device=self._device)
        for i, length in enumerate(lengths):
            mask[length:, i] = 0
        return mask
