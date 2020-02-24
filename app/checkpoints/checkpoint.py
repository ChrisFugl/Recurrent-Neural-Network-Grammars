class Checkpoint:
    """
    Base class for when to save checkpoints.
    """

    def should_save_checkpoint(self, epoch, batch_count, start_of_epoch):
        """
        :type epoch: int
        :type batch_count: int
        :type start_of_epoch: bool
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')
