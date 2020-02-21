import torch

class Action:
    """
    All actions should inherit from this class.
    """

    def __init__(self, device):
        """
        :type device: torch.device
        """
        self._device = device

    def index(self):
        """
        :type device: torch.device
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def index_as_tensor(self):
        """
        :rtype: torch.Tensor
        """
        index = self.index()
        return self._to_long_tensor(index)

    def type(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def _to_long_tensor(self, value):
        return torch.tensor(value, dtype=torch.long, device=self._device)
