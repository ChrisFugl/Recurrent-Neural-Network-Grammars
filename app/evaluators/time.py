from app.evaluators.evaluator import Evaluator
import time

class TimeEvaluator(Evaluator):

    def __init__(self, interval_s, pretraining):
        """
        :type interval_s: int
        :type pretraining: bool
        """
        super().__init__()
        self.interval_s = interval_s
        self.pretraining = pretraining
        self.timestamp = time.time()

    def should_evaluate(self, epoch, batch, pretraining=False, end_of_epoch=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type end_of_epoch: bool
        """
        if pretraining:
            return self.pretraining
        now = time.time()
        if self.interval_s < now - self.timestamp:
            self.timestamp = now
            return True
        else:
            return False
