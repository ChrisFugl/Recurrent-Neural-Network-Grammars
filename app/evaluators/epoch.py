from app.evaluators.evaluator import Evaluator

class EpochEvaluator(Evaluator):

    def __init__(self, epoch, pretraining, posttraining):
        """
        :type epoch: int
        :type pretraining: bool
        :type posttraining: bool
        """
        super().__init__()
        self._epoch = epoch
        self._pretraining = pretraining
        self._posttraining = posttraining

    def should_evaluate(self, epoch, batch, pretraining=False, posttraining=False, end_of_epoch=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type posttraining: bool
        :type end_of_epoch: bool
        """
        return (self._pretraining and pretraining) or (self._posttraining and posttraining) or (end_of_epoch and epoch % self._epoch == 0)
