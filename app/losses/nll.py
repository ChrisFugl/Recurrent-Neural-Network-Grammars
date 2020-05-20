from app.losses.loss import Loss
import torch

class NLLLoss(Loss):

    def __init__(self, device):
        """
        :type device: torch.device
        """
        super().__init__()
        self.device = device

    def forward(self, predictions, groundtruths, lengths):
        """
        :param predictions: batch of log probabilities of each action (size S x B x A)
        :param groundtruths: batch of groundtruth actions (size S x B)
        :type predictions: torch.Tensor
        :type groundtruths: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        max_length, batch_size, action_count = predictions.shape
        mask = torch.zeros(groundtruths.shape, device=self.device, dtype=torch.bool)
        for index, length in enumerate(lengths):
            mask[:length, index] = 1

        indices = groundtruths.unsqueeze(dim=2)
        log_probs = torch.gather(predictions, 2, indices).view(max_length, batch_size)
        masked_log_probs = log_probs * mask

        log_likelihoods = masked_log_probs.sum(dim=0)
        loss = - log_likelihoods.mean()
        return loss
