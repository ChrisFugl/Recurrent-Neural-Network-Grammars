from torch.optim import lr_scheduler
import warnings

class ConstantLearningRateScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        """
        :type optimizer: torch.optim.Optimizer
        :type last_epoch: int
        """
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use \`get_last_lr()\`.', DeprecationWarning)
        return self.base_lrs
