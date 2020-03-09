from app.tasks.task import Task
import logging
import os
import time

class EvaluateTask(Task):

    def __init__(self, device, data, sampler):
        """
        :type device: torch.device
        :type data: str
        :type sampler: app.samplers.sampler.Sampler
        """
        super().__init__()
        self._logger = logging.getLogger('evaluate')
        self._device = device
        self._data = data
        self._sampler = sampler

    def run(self):
        time_start = time.time()
        self._logger.info('Starting evaluation')
        self._logger.info(f'Saving output to {os.getcwd()}')
        self._logger.info(f'Using device: {self._device}')
        self._logger.info(f'Data: {self._data}')
        self._logger.info(f'Sampler: {self._sampler}')
        self._evaluate()
        time_stop = time.time()
        self._logger.info('Finished evaluation')
        self._logger.info(f'Time taken: {time_stop - time_start:0.2f} s')

    def _evaluate(self):
        gold_trees, predicted_trees, predicted_log_probs = self._sampler.evaluate()
        # bracket_scores = self._bracket_scores(tree_groundtruths, tree_predictions)

    def _bracket_scores(self, groundtruths, predictions):
        # TODO
        return None
