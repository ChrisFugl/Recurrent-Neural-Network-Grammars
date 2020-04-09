from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE
from app.data.batch import Batch
from app.data.batch_utils import sequences2lengths, sequences2tensor
import logging

class Sampler:

    def __init__(self, device, action_converter, log=True):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type log: bool
        """
        self.device = device
        self.action_converter = action_converter
        self.log = log
        self.logger = logging.getLogger('sampler')

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
            if self.log and threshold <= threshold_counter:
                self.logger.info(f'Sampling: {tree_counter:,} / {count:,} ({tree_counter/count:0.2%})')
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

    def sample(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: app.data.batch.Batch
        """
        state = self.get_initial_state(batch)
        lengths_list = [length.cpu().item() for length in batch.tokens.lengths]
        predicted_actions = [[] for _ in range(batch.size)]
        finished_sampling = [False] * batch.size
        while not all(finished_sampling):
            log_probs = self.get_next_log_probs(state)
            samples = self.sample_actions(log_probs)
            actions = []
            for i, finished in enumerate(finished_sampling):
                if finished:
                    # None represents padding
                    actions.append(None)
                else:
                    sample = samples[i]
                    action = self.action_converter.integer2action(sample)
                    predicted_actions[i].append(action)
                    finished_sampling[i] = self.is_finished_sampling(predicted_actions[i], lengths_list[i])
                    actions.append(action)
            state = self.get_next_state(state, actions)
        # create batch from samples
        predicted_tensor = sequences2tensor(self.device, self.action_converter.action2integer, predicted_actions)
        predicted_lengths = sequences2lengths(self.device, predicted_actions)
        return Batch(
            predicted_tensor, predicted_lengths, predicted_actions,
            batch.tokens.tensor, batch.tokens.lengths, batch.tokens.tokens,
            batch.tags.tensor, batch.tags.lengths, batch.tags.tags,
        )

    def get_initial_state(self, batch):
        raise NotImplementedError('must be implemented by subclass')

    def get_next_log_probs(self, state):
        raise NotImplementedError('must be implemented by subclass')

    def get_next_state(self, state, actions):
        raise NotImplementedError('must be implemented by subclass')

    def sample_actions(self, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def batch_stats(self, batch_log_probs, lengths):
        """
        :param batch_log_probs: tensor, S x B x A
        :type batch_log_probs: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of float, list of list of float
        """
        summed = batch_log_probs.sum(dim=2)
        log_probs = [self.get_log_prob(summed[:length, i]) for i, length in enumerate(lengths)]
        probs = [self.get_probs(summed[:length, i]) for i, length in enumerate(lengths)]
        return log_probs, probs

    def get_log_prob(self, log_probs):
        return log_probs.sum().cpu().item()

    def get_probs(self, log_probs):
        return [prob.cpu().item() for prob in log_probs.exp()]

    def count_actions(self, actions, type):
        filtered = filter(lambda action: action.type() == type, actions)
        return len(list(filtered))
