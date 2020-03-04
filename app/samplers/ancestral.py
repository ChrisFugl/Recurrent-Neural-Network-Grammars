from app.samplers.sampler import Sampler
import torch

class AncestralSampler(Sampler):

    def __init__(self, device, model, posterior_scaling):
        """
        :type device: torch.device
        :type model: torch.model
        :type posterior_scaling: float
        """
        super().__init__()
        self._device = device
        self._model = model
        self._posterior_scaling = posterior_scaling

    def sample(self, tokens):
        """
        :param tokens: tokens length x 1 x hidden size
        :type tokens: torch.Tensor
        :returns: actions length x 1 x 1
        :rtype: torch.Tensor
        """
        # TODO
        return torch.tensor([[[0]]], dtype=torch.long, device=self._device)

    def __str__(self):
        return f'Ancestral(posterior_scaling={self._posterior_scaling})'
