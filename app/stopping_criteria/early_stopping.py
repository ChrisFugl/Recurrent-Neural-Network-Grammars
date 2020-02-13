from app.stopping_criteria.stopping_criterion import StoppingCriterion

class EarlyStoppingStoppingCriterion(StoppingCriterion):

    def __init__(self, epsilon):
        """
        :type epsilon: float
        """
        super().__init__()
        self._epsilon = epsilon
        self._prev_val_loss = None

    def is_done(self, epoch, val_loss):
        """
        :type epoch: int
        :type val_loss: float
        :rtype: bool
        """
        if self._prev_val_loss is None:
            self._prev_val_loss = val_loss
            return False

        if val_loss < self._prev_val_loss or self._equal(val_loss, self._prev_val_loss):
            return True

        self._prev_val_loss = val_loss
        return False

    def _equal(self, val_loss, prev_val_loss):
        return abs(val_loss - prev_val_loss) < self._epsilon
