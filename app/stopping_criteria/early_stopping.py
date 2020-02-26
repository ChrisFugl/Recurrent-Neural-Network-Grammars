from app.stopping_criteria.stopping_criterion import StoppingCriterion

class EarlyStoppingStoppingCriterion(StoppingCriterion):

    def __init__(self, epsilon):
        """
        :type epsilon: float
        """
        super().__init__()
        self._epsilon = epsilon
        self._done = False
        self._prev_val_loss = None

    def is_done(self):
        """
        :rtype: bool
        """
        return self._done

    def add_epoch(self, epoch):
        """
        :type epoch: int
        """
        pass

    def add_val_loss(self, val_loss):
        """
        :type val_loss: float
        """
        if self._prev_val_loss is None:
            self._prev_val_loss = val_loss
            return
        self._done = self._done or self._prev_val_loss < val_loss or self._equal(val_loss, self._prev_val_loss)
        self._prev_val_loss = val_loss

    def state_dict(self):
        """
        :rtype: object
        """
        return {
            'prev_val_loss': self._prev_val_loss,
        }

    def load_state_dict(self, state_dict):
        """
        :type state_dict: object
        :rtype: object
        """
        self._prev_val_loss = state_dict['prev_val_loss']

    def _equal(self, val_loss, prev_val_loss):
        return abs(val_loss - prev_val_loss) < self._epsilon
