from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE
from app.data.batch import Batch
from app.data.batch_utils import sequences2lengths, sequences2tensor
import logging
import torch

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
        self.action_count = action_converter.count()
        self.reduce_index = action_converter.string2integer('REDUCE')
        self.shift_index = action_converter.string2integer('SHIFT')
        nt_start = action_converter.get_non_terminal_offset()
        nt_count = action_converter.count_non_terminals()
        self.nt_indices = list(range(nt_start, nt_start + nt_count))

    def get_samples(self):
        """
        :rtype: list of app.samplers.sample.Sample
        """
        samples = []
        iterator, count = self.get_iterator()
        tree_counter = 0
        threshold_counter = 0
        threshold = count / 10 # log every 10% of total iterations
        for batch in iterator:
            batch_samples = self.sample_batch(batch)
            batch_size = len(batch_samples)
            samples.extend(batch_samples)
            tree_counter += batch_size
            threshold_counter += batch_size
            if self.log and threshold <= threshold_counter:
                self.logger.info(f'Sampling: {tree_counter:,} / {count:,} ({tree_counter/count:0.2%})')
                threshold_counter = 0
        return samples

    def sample_batch(self, batch):
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
            and self.count_actions(actions, ACTION_SHIFT_TYPE) == tokens_length
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
            samples = self.sample_actions(state, log_probs)
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
            batch.unknownified_tokens.tensor, batch.unknownified_tokens.tokens,
            batch.singletons,
            batch.tags.tensor, batch.tags.tags,
        )

    def get_initial_state(self, batch):
        raise NotImplementedError('must be implemented by subclass')

    def get_next_log_probs(self, state):
        raise NotImplementedError('must be implemented by subclass')

    def get_next_state(self, state, actions):
        raise NotImplementedError('must be implemented by subclass')

    def get_valid_actions(self, state, actions):
        raise NotImplementedError('must be implemented by subclass')

    def sample_action(self, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def sample_actions(self, state, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: list of int
        """
        samples = []
        batch_valid_actions = self.get_valid_actions(state)
        for i, valid_actions in enumerate(batch_valid_actions):
            valid_indices, index2action = self.get_valid_indices(valid_actions)
            valid_log_probs = log_probs[i, valid_indices]
            normalized_valid_log_probs = valid_log_probs - self.log_sum_exp(valid_log_probs)
            sample = self.sample_action(normalized_valid_log_probs)
            action_index = index2action[sample]
            samples.append(action_index)
        return samples

    def batch_log_probs(self, batch_log_probs, actions, lengths):
        """
        :param batch_log_probs: tensor, S x B x A
        :type batch_log_probs: torch.Tensor
        :param actions: tensor, S x B
        :type actions: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of torch.Tensor, list of float
        """
        max_length, batch_size = actions.shape
        indices = actions.unsqueeze(dim=2)
        selected_log_probs = torch.gather(batch_log_probs, 2, indices).view(max_length, batch_size)
        selected_log_probs = [selected_log_probs[:length, i] for i, length in enumerate(lengths)]
        log_likelihoods = [self.log_likelihood(log_probs) for log_probs in selected_log_probs]
        return selected_log_probs, log_likelihoods

    def log_likelihood(self, log_probs):
        return log_probs.sum().cpu().item()

    def count_actions(self, actions, type):
        filtered = filter(lambda action: action.type() == type, actions)
        return len(list(filtered))

    def get_valid_indices(self, valid_actions):
        valid_indices = []
        index2action = {}
        counter = 0
        if ACTION_REDUCE_TYPE in valid_actions:
            valid_indices.append(self.reduce_index)
            index2action[counter] = self.reduce_index
            counter += 1
        if ACTION_SHIFT_TYPE in valid_actions:
            valid_indices.append(self.shift_index)
            index2action[counter] = self.shift_index
            counter += 1
        if ACTION_NON_TERMINAL_TYPE in valid_actions:
            valid_indices.extend(self.nt_indices)
            for nt_index in self.nt_indices:
                index2action[counter] = nt_index
                counter += 1
        return valid_indices, index2action

    def log_sum_exp(self, values):
        max = values.max()
        shifted = torch.exp(values - max)
        summed = shifted.sum()
        return max + summed.log()
