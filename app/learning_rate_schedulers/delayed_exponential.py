from torch.optim import lr_scheduler
import warnings

class DelayedExponentialLearningRateScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, delay, exponential_base, last_epoch=-1):
        """
        :type optimizer: torch.optim.Optimizer
        :type delay: int
        :type exponential_base: float
        :type last_epoch: int
        """
        self.delay = delay
        self.exponential_base = exponential_base
        super(DelayedExponentialLearningRateScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use \`get_last_lr()\`.', DeprecationWarning)
        return [self._get_delayed_exponential_lr(lr) for lr in self.base_lrs]

    def _get_delayed_exponential_lr(self, learning_rate):
        if self.last_epoch <= self.delay:
            return learning_rate
        else:
            exponential_factor = self.exponential_base ** (self.last_epoch - self.delay)
            return learning_rate * exponential_factor
