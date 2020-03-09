from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE
from app.data.preprocessing.unknowns import fine_grained_unknownifier
import logging
import torch

class Sampler:

    def __init__(self):
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
            batch_size = self.get_batch_size(batch)
            for batch_index in range(batch_size):
                sample = self.evaluate_element(batch, batch_index)
                samples.append(sample)
                tree_counter += 1
                threshold_counter += 1
                if threshold <= threshold_counter:
                    self._logger.info(f'Sampling: {tree_counter:,} / {count:,} ({tree_counter/count:0.2%})')
                    threshold_counter = 0
        return samples

    def evaluate_element(self, batch, batch_index):
        """
        :rtype: app.samplers.sample.Sample
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_batch_size(self, batch):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_iterator(self):
        raise NotImplementedError('must be implemented by subclass')

    def sample(self, tokens):
        """
        :type tokens: list of str
        :rtype: list of app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def _is_finished_sampling(self, actions, tokens_length):
        """
        :type actions: list of app.data.actions.action.Action
        :rtype: bool
        """
        return (
                len(actions) > 2
            and self._count(actions, ACTION_NON_TERMINAL_TYPE) == self._count(actions, ACTION_REDUCE_TYPE)
            and (
                   self._count(actions, ACTION_SHIFT_TYPE) == tokens_length
                or self._count(actions, ACTION_GENERATE_TYPE) == tokens_length
            )
        )

    def _count(self, actions, type):
        filtered = filter(lambda action: action.type() == type, actions)
        return len(list(filtered))

    def _actions2tensor(self, action_converter, actions):
        action_integers = [action_converter.action2integer(action) for action in actions]
        actions_tensor = torch.tensor(action_integers, device=self._device, dtype=torch.long)
        return actions_tensor.reshape((len(actions), 1))

    def _tokens2tensor(self, token_converter, tokens):
        known_tokens = self.token_converter.tokens()
        unknownified_tokens = [fine_grained_unknownifier(known_tokens, (None, token)) for token in tokens]
        token_integers = [self._token_converter.token2integer(token) for token in unknownified_tokens]
        tokens_tensor = torch.tensor(token_integers, device=self._device, dtype=torch.long)
        return tokens_tensor.reshape((len(tokens), 1))
