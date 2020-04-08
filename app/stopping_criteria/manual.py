from app.stopping_criteria.stopping_criterion import StoppingCriterion
import logging
import signal

class ManualStoppingCriterion(StoppingCriterion):

    def __init__(self):
        super().__init__()
        self._done = False
        self._logger = logging.getLogger('stopping_criterion')
        signal.signal(signal.SIGINT, self._exit_requested)
        signal.signal(signal.SIGTERM, self._exit_requested)

    def is_done(self):
        """
        :rtype: bool
        """
        return self._done

    def add_epoch(self, epoch):
        """
        :type epoch: int
        """
        pass

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

    def _exit_requested(self, signum, frame):
        self._logger.info('Exit requested')
        self._logger.info('Waiting for final evaluation before exiting')
        self._logger.info('ctrl-c again to force exit without final evaluation')
        self._done = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
