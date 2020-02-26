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

    def should_evaluate(self, epoch, batch, start_of_epoch=False, end_of_epoch=False, pretraining=False, posttraining=False):
        """
        :type epoch: int
        :type batch: int
        :type start_of_epoch: bool
        :type end_of_epoch: bool
        :type pretraining: bool
        :type posttraining: bool
        """
        return (self._pretraining and pretraining) or (self._posttraining and posttraining) or batch % self._batch == 0
