class Evaluator:
    """
    Base class for all classes used to determine when to evaluate performance during training.
    """

    def should_evaluate(self, epoch, batch, pretraining=False, end_of_epoch=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type end_of_epoch: bool
        """
        raise NotImplementedError('must be implemented by subclass')

    def evaluation_finished(self):
        raise NotImplementedError('must be implemented by subclass')
