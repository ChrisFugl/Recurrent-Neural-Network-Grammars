from app.losses.loss import Loss
import torch
from torch import nn

class NegativeLogLikelihoodLoss(Loss):

    def __init__(self, device):
        """
        :type device: torch.device
        """
        super().__init__()
        self.device = device
        self.loss = nn.NLLLoss()

    def forward(self, predictions, groundtruths, lengths):
        """
        :param predictions: batch of log probabilities of each action (size S x B x A)
        :param groundtruths: batch of groundtruth actions (size S x B)
        :type predictions: torch.Tensor
        :type groundtruths: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        _, _, action_count = predictions.shape
        mask = torch.zeros(groundtruths.shape, device=self.device, dtype=torch.bool)
        for index, length in enumerate(lengths):
            mask[:length, index] = 1
        mask_flattened = mask.view(-1)
        mask_expanded = mask_flattened.unsqueeze(dim=1).expand(-1, action_count)
        groundtruths_flattened = groundtruths.view(-1)
        groundtruths_masked = groundtruths_flattened.masked_select(mask_flattened)
        predictions_flattened = predictions.view(-1, action_count)
        predictions_masked = predictions_flattened.masked_select(mask_expanded).view(-1, action_count)
        return self.loss(predictions_masked, groundtruths_masked)
