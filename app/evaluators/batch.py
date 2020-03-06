from app.evaluators.evaluator import Evaluator

class BatchEvaluator(Evaluator):

    def __init__(self, batch, pretraining):
        """
        :type batch: int
        :type pretraining: bool
        """
        super().__init__()
        self._batch = batch
        self._pretraining = pretraining

    def should_evaluate(self, epoch, batch, pretraining=False, end_of_epoch=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type end_of_epoch: bool
        """
        return (self._pretraining and pretraining) or (not end_of_epoch and batch % self._batch == 0)
