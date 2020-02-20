from app.constants import PAD_INDEX
from app.data.iterators.iterator import Iterator
import random
import torch
from torch.nn.utils.rnn import pad_sequence

class UnorderedIterator(Iterator):

    def __init__(self, tokens, actions, token_converter, action_converter, device, batch_size, shuffle):
        """
        :type tokens: list of list of str
        :type actions: list of list of str
        :type token_converter: app.data.converters.token.TokenConverter
        :type action_converter: app.data.converters.action.ActionConverter
        :type device: torch.device
        :type batch_size: int
        :type shuffle: bool
        """
        super().__init__()
        self._tokens_strings = tokens
        self._actions_strings = actions
        self._tokens = self._tokens2integers(token_converter, tokens)
        self._actions = self._actions2integers(action_converter, actions)
        self._tokens_count = len(self._tokens)
        self._token_converter = token_converter
        self._device = device
        self._batch_size = batch_size
        self._shuffle = shuffle

    def _tokens2integers(self, token_converter, sentences):
        integers = []
        for sentence in sentences:
            sentence_integers = list(map(token_converter.token2integer, sentence))
            integers.append(sentence_integers)
        return integers

    def _actions2integers(self, action_converter, trees):
        integers = []
        for tree in trees:
            tree_integers = list(map(action_converter.action2integer, tree))
            integers.append(tree_integers)
        return integers

    def __iter__(self):
        if self._shuffle:
            actions, actions_strings, tokens, tokens_strings = self._shuffle_lists(
                self._actions,
                self._actions_strings,
                self._tokens,
                self._tokens_strings
            )
        else:
            actions = self._actions
            actions_strings = self._actions_strings
            tokens = self._tokens
            tokens_strings = self._tokens_strings
        return UnorderedIterable(tokens, tokens_strings, actions, actions_strings, self._device, self._batch_size)

    def _shuffle_lists(self, *lists):
        zipped = list(zip(*lists))
        random.shuffle(zipped)
        return zip(*zipped)

class UnorderedIterable:

    def __init__(self, tokens, tokens_strings, actions, actions_strings, device, batch_size):
        """
        :type tokens: list of list of int
        :type tokens_strings: list of list of str
        :type actions: list of list of int
        :type actions_strings: list of list of str
        :type device: torch.device
        :type batch_size: int
        """
        self._tokens = tokens
        self._tokens_strings = tokens_strings
        self._actions = actions
        self._actions_strings = actions_strings
        self._batch_size = batch_size
        self._device = device
        self._counter = 0
        self._total = len(tokens)

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter < self._total:
            start = self._counter
            end = min(self._counter + self._batch_size, self._total)
            tokens_strings = list(self._tokens_strings[start:end])
            actions_strings = list(self._actions_strings[start:end])
            tokens = list(map(self._to_tensor, self._tokens[start:end]))
            actions = list(map(self._to_tensor, self._actions[start:end]))
            # loop back to beginning if batch exceeds final observation
            # this ensures that batches always have a fixed length
            remaining = self._batch_size - (end - start)
            if 0 < remaining:
                tokens_strings_extension = self._tokens_strings[0:remaining]
                actions_strings_extension = self._actions_strings[0:remaining]
                tokens_extension = list(map(self._to_tensor, self._tokens[0:remaining]))
                actions_extension = list(map(self._to_tensor, self._actions[0:remaining]))
                tokens_strings.extend(tokens_strings_extension)
                actions_strings.extend(actions_strings_extension)
                tokens.extend(tokens_extension)
                actions.extend(actions_extension)
            self._counter += self._batch_size
            tokens_padded, tokens_lengths = self._pad(tokens)
            actions_padded, actions_lengths = self._pad(actions)
            output_tokens = (tokens_padded, tokens_lengths, tokens_strings)
            output_actions = (actions_padded, actions_lengths, actions_strings)
            return output_tokens, output_actions
        else:
            raise StopIteration()

    def _to_tensor(self, values):
        sequence_length = len(values)
        shape = (sequence_length,)
        return torch.tensor(values, device=self._device, dtype=torch.long).reshape(shape)

    def _pad(self, tensors):
        padded = pad_sequence(tensors, batch_first=False, padding_value=PAD_INDEX)
        lengths = list(map(len, tensors))
        return padded, lengths
