from app.evaluators.evaluator import Evaluator

class EpochEvaluator(Evaluator):

    def __init__(self, epoch, pretraining):
        """
        :type epoch: int
        :type pretraining: bool
        """
        super().__init__()
        self.epoch = epoch
        self.pretraining = pretraining

    def should_evaluate(self, epoch, batch, pretraining=False, end_of_epoch=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type end_of_epoch: bool
        """
        if pretraining:
            return self.pretraining
        return end_of_epoch and epoch % self.epoch == 0

    def evaluation_finished(self):
        pass
