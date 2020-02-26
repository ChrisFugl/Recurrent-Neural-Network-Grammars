class StoppingCriterion:
    """
    Abstract class that all stopping criteria must inherit from.
    """

    def is_done(self):
        """
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')

    def add_epoch(self, epoch):
        """
        :type epoch: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def add_val_loss(self, val_loss):
        """
        :type val_loss: float
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
