from app.evaluators.evaluator import Evaluator

class BatchEvaluator(Evaluator):

    def __init__(self, batch, pretraining, posttraining):
        """
        :type batch: int
        :type pretraining: bool
        :type posttraining: bool
        """
        super().__init__()
        self._batch = batch
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
        return (self._pretraining and pretraining) or (self._posttraining and posttraining) or batch % self._batch == 0
