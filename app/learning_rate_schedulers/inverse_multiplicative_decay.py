from torch.optim import lr_scheduler
import warnings

class InverseAdditiveDecayLearningRateScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay, last_epoch=-1):
        """
        :type optimizer: torch.optim.Optimizer
        :type decay: float
        :type last_epoch: int
        """
        self._decay = decay
        super(InverseAdditiveDecayLearningRateScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use \`get_last_lr()\`.', DeprecationWarning)
        return [self._get_decayed_lr(lr) for lr in self.base_lrs]

    def _get_decayed_lr(self, learning_rate):
        return learning_rate / (1.0 + self.last_epoch * self._decay)
