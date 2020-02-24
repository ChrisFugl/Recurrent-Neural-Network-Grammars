from app.checkpoints.checkpoint import Checkpoint
import time

class TimeCheckpoint(Checkpoint):
    """
    Save checkpoints at a given time interval.
    """

    def __init__(self, interval_s):
        """
        :type interval_s: int
        """
        super().__init__()
        self._interval_s = interval_s
        self._checkpoint_time = time.time()

    def should_save_checkpoint(self, epoch, batch_count, start_of_epoch):
        """
        :type epoch: int
        :type batch_count: int
        :type start_of_epoch: bool
        :rtype: bool
        """
        timestamp = time.time()
        if self._interval_s < timestamp - self._checkpoint_time:
            self._checkpoint_time = timestamp
            return True
        else:
            return False
