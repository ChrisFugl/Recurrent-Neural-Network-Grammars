class Evaluator:
    """
    Base class for all classes used to determine when to evaluate performance during training.
    """

    def should_evaluate(self, epoch, batch, pretraining=False, posttraining=False):
        """
        :type epoch: int
        :type batch: int
        :type pretraining: bool
        :type posttraining: bool
        """
        raise NotImplementedError('must be implemented by subclass')
