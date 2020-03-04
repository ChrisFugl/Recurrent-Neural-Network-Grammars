from app.tasks.task import Task
import logging
import os

class EvaluateTask(Task):

    def __init__(self, device, inferer, sampler, iterator):
        """
        :type device: torch.device
        :type inferer: app.inferers.inferer.Inferer
        :type sampler: app.samplers.sampler.Sampler
        :type iterator: app.data.iterators.iterator.Iterator
        """
        super().__init__()
        self._logger = logging.getLogger('evaluate')
        self._device = device
        self._inferer = inferer
        self._sampler = sampler
        self._iterator = iterator

    def run(self):
        self._logger.info('Starting evaluation')
        self._logger.info(f'Saving output to {os.getcwd()}')
        self._logger.info(f'Using device: {self._device}')
        self._logger.info(f'Sampler: {self._sampler}')
        self._logger.info(f'Inferer: {self._inferer}')
        self._evaluate()
        self._logger.info('Finished evaluation')

    def _evaluate(self):
        tree_groundtruths, tree_predictions, scores = self._infer()
        bracket_scores = self._bracket_scores(tree_groundtruths, tree_predictions)
        print('groundtruths')
        print(tree_groundtruths)
        print('predictions')
        print(tree_predictions)

    def _infer(self):
        score_names = self._inferer.names()
        scores = {}
        for score_name in score_names:
            scores[score_name] = []
        tree_groundtruths = []
        tree_predictions = []
        for batch in self._iterator:
            self._infer_batch(tree_groundtruths, tree_predictions, scores, batch)
            # TODO: do not break
            break
        return tree_groundtruths, tree_predictions, scores

    def _infer_batch(self, tree_groundtruths, tree_predictions, scores, batch):
        """
        :type batch: app.data.batch.Batch
        """
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            tree, score = self._inferer.infer(element.tokens.tensor)
            for name, value in score.items():
                scores[name].append(value)
            tree_groundtruths.append(element.actions.tensor.squeeze())
            tree_predictions.append(tree)
            # TODO: do not break
            break

    def _bracket_scores(self, groundtruths, predictions):
        # TODO
        return None
