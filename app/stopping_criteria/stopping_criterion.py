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

    def state_dict(self):
        """
        :rtype: object
        """
        raise NotImplementedError('must be implemented by subclass')

    def load_state_dict(self, state_dict):
        """
        :type state_dict: object
        :rtype: object
        """
        raise NotImplementedError('must be implemented by subclass')
