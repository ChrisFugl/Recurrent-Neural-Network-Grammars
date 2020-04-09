from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE
import logging
import torch

class Sampler:

    def __init__(self, log=True):
        """
        :type log: bool
        """
        self._log = log
        self._logger = logging.getLogger('sampler')

    def evaluate(self):
        """
        :rtype: list of app.samplers.sample.Sample
        """
        samples = []
        iterator, count = self.get_iterator()
        tree_counter = 0
        threshold_counter = 0
        threshold = count / 10 # log every 10% of total iterations
        for batch in iterator:
            batch_samples = self.evaluate_batch(batch)
            batch_size = len(batch_samples)
            samples.extend(batch_samples)
            tree_counter += batch_size
            threshold_counter += batch_size
            if self._log and threshold <= threshold_counter:
                self._logger.info(f'Sampling: {tree_counter:,} / {count:,} ({tree_counter/count:0.2%})')
                threshold_counter = 0
        return samples

    def evaluate_batch(self, batch):
        """
        :rtype: list of app.samplers.sample.Sample
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_iterator(self):
        raise NotImplementedError('must be implemented by subclass')

    def is_finished_sampling(self, actions, tokens_length):
        """
        :type actions: list of app.data.actions.action.Action
        :rtype: bool
        """
        return (
                len(actions) > 2
            and self.count_actions(actions, ACTION_NON_TERMINAL_TYPE) == self.count_actions(actions, ACTION_REDUCE_TYPE)
            and (
                   self.count_actions(actions, ACTION_SHIFT_TYPE) == tokens_length
                or self.count_actions(actions, ACTION_GENERATE_TYPE) == tokens_length
            )
        )

    def count_actions(self, actions, type):
        filtered = filter(lambda action: action.type() == type, actions)
        return len(list(filtered))

    def actions2tensor(self, action_converter, actions):
        action_integers = [action_converter.action2integer(action) for action in actions]
        actions_tensor = torch.tensor(action_integers, device=self._device, dtype=torch.long)
        return actions_tensor.reshape((len(actions), 1))
