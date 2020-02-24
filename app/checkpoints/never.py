from app.checkpoints.checkpoint import Checkpoint

class NeverCheckpoint(Checkpoint):
    """
    Never save any checkpoints.
    """

    def should_save_checkpoint(self, epoch, batch_count, start_of_epoch):
        """
        :type epoch: int
        :type batch_count: int
        :type start_of_epoch: bool
        :rtype: bool
        """
        return False
