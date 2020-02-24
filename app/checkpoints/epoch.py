from app.checkpoints.checkpoint import Checkpoint

class EpochCheckpoint(Checkpoint):
    """
    Save checkpoints at a given epoch interval.
    """

    def __init__(self, epoch):
        """
        :type epoch: int
        """
        super().__init__()
        self._epoch = epoch

    def should_save_checkpoint(self, epoch, batch_count, start_of_epoch):
        """
        :type epoch: int
        :type batch_count: int
        :type start_of_epoch: bool
        :rtype: bool
        """
        return start_of_epoch and epoch != 0 and epoch % self._epoch == 0
