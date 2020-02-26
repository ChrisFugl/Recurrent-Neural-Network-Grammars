from app.stopping_criteria.stopping_criterion import StoppingCriterion

class EpochsStoppingCriterion(StoppingCriterion):

    def __init__(self, epochs):
        """
        :type epochs: int
        """
        super().__init__()
        self._epochs = epochs
        self._done = False

    def is_done(self):
        """
        :rtype: bool
        """
        return self._done

    def add_epoch(self, epoch):
        """
        :type epoch: int
        """
        self._done = self._done or self._epochs == epoch

    def add_val_loss(self, val_loss):
        """
        :type val_loss: float
        """
        pass

    def state_dict(self):
        """
        :rtype: object
        """
        pass

    def load_state_dict(self, state_dict):
        """
        :type state_dict: object
        :rtype: object
        """
        pass
