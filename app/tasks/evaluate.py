from app.scores import scores_from_samples
from app.tasks.task import Task
import logging
from math import exp, log
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
        samples = self._sampler.evaluate()
        scores, evalb_output, gold_path, predicted_path = scores_from_samples(samples)
        f1, precision, recall = scores
        self._logger.info(f'Saved gold trees at {gold_path}')
        self._logger.info(f'Saved predicted trees at {predicted_path}')
        self._log_stats('Gold trees', self._get_stats('gold', samples))
        self._log_stats('Predicted trees', self._get_stats('predicted', samples))
        self._log_stats('Tokens', self._get_stats('tokens', samples))
        self._logger.info(f'Recall    = {recall}')
        self._logger.info(f'Precision = {precision}')
        self._logger.info(f'F1        = {f1}')
        filename = 'evalb.txt'
        path = self._get_path(filename)
        with open(filename, 'w') as file:
            file.write(evalb_output)
        self._logger.info(f'Saved evalb output to {path}')

    def _get_path(self, filename):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, filename)
        return path

    def _get_stats(self, type, samples):
        """
        :type samples: list of app.samplers.sample.Sample
        """
        log_probs, probs, perplexities = [], [], []
        if type == 'gold' or type == 'predicted':
            if type == 'gold':
                items = [sample.gold.actions for sample in samples]
                log_probs = [sample.gold.log_prob for sample in samples]
            else:
                items = [sample.prediction.actions for sample in samples]
                log_probs = [sample.prediction.log_prob for sample in samples]
            for index, log_prob in enumerate(log_probs):
                try:
                    probs.append(exp(log_prob))
                    perplexities.append(exp(- log_prob / len(items[index])))
                except Exception:
                    self._logger.warning(f'Failed to compute probability/perplexity of {type} tree at index {index}')
        else:
            log_probs = []
            for index, sample in enumerate(samples):
                if sample.tokens_prob is not None:
                    try:
                        log_prob = log(sample.tokens_prob)
                        probs.append(sample.tokens_prob)
                        log_probs.append(log_prob)
                        perplexities.append(exp(- log_prob / len(sample.gold.tokens)))
                    except ValueError:
                        self._logger.warning(f'Failed to compute probability/perplexity of tokens at index {index}')
        if len(log_probs) == 0 or len(probs) == 0 or len(perplexities) == 0:
            return None
        log_likelihood = sum(log_probs) / len(log_probs)
        likelihood = sum(probs) / len(probs)
        perplexity = sum(perplexities) / len(perplexities)
        return log_likelihood, likelihood, perplexity

    def _log_stats(self, name, stats):
        if stats is not None:
            log_likelihood, likelihood, perplexity = stats
            self._logger.info(f'{name} mean log likelihood = {log_likelihood:0.8f}')
            self._logger.info(f'{name} mean likelihood     = {likelihood:0.8f}')
            self._logger.info(f'{name} mean perplexity     = {perplexity:0.8f}')
