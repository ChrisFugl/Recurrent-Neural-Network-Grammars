from app.losses.loss import Loss
import torch

class NegativeLikelihoodLoss(Loss):

    def __init__(self, device):
        """
        :type device: torch.device
        """
        super().__init__()
        self._device = device

    def forward(self, actions_probabilities, actions_lengths):
        """
        :param actions_probabilities: batch of probabilities of each action (size S x B)
        :type actions_probabilities: torch.Tensor
        :type actions_lengths: list of int
        :rtype: torch.Tensor
        """
        actions_probabilities_masked = actions_probabilities
        for i, length in enumerate(actions_lengths):
            actions_probabilities_masked[length - 1:, i] = 1
        actions_probabilities_product = actions_probabilities_masked.prod(dim=0)
        actions_probabilities_sum = actions_probabilities_product.sum()
        _, batch_size = actions_probabilities.shape
        return - actions_probabilities_sum / batch_size

    def _get_mask(self, actions_probabilities, actions_lengths):
        max_length = max(actions_lengths)
        batch_size = len(actions_lengths)
        mask = torch.zeros(max_length, batch_size, device=self._device)
        for i, length in enumerate(actions_lengths):
            mask[length:, i] = 1.0 / actions_probabilities[length:, i]
        return mask
