import torch

class Action:
    """
    All actions should inherit from this class.
    """

    def index(self):
        """
        :type device: torch.device
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def type(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')
