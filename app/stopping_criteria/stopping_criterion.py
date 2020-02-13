class StoppingCriterion:
    """
    Abstract class that all stopping criteria must inherit from.
    """

    def is_done(self, epoch, val_loss):
        """
        :type epoch: int
        :type val_loss: float
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')
