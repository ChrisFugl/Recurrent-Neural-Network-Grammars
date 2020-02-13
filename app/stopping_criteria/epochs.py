from app.stopping_criteria.stopping_criterion import StoppingCriterion

class EpochsStoppingCriterion(StoppingCriterion):

    def __init__(self, epochs):
        """
        :type epochs: int
        """
        super().__init__()
        self._epochs = epochs

    def is_done(self, epoch, val_loss):
        """
        :type epoch: int
        :type val_loss: float
        :rtype: bool
        """
        return self._epochs < epoch
