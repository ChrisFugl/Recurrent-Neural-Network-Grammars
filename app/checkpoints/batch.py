from app.checkpoints.checkpoint import Checkpoint

class BatchCheckpoint(Checkpoint):
    """
    Save checkpoints at a given batch interval.
    """

    def __init__(self, batch):
        """
        :type batch: int
        """
        super().__init__()
        self._batch = batch

    def should_save_checkpoint(self, epoch, batch_count, start_of_epoch):
        """
        :type epoch: int
        :type batch_count: int
        :type start_of_epoch: bool
        :rtype: bool
        """
        return batch_count != 0 and batch_count % self._batch == 0
