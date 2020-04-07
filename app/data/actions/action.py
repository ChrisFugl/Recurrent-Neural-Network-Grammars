import torch

class Action:
    """
    All actions should inherit from this class.
    """

    def type(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')
