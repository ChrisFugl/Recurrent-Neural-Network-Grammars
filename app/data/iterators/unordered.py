from app.constants import PAD_INDEX
from app.data.iterators.iterator import Iterator
from functools import partial
import random
import torch
from torch.nn.utils.rnn import pad_sequence

class UnorderedIterator(Iterator):

    def __init__(self, device, action_converter, token_converter, batch_size, shuffle, tokens_strings, actions_strings):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type batch_size: int
        :type shuffle: bool
        :type tokens_strings: list of list of str
        :type actions_strings: list of list of str
        """
        super().__init__()
        self._actions = self._convert(partial(action_converter.string2action, device), actions_strings)
        self._actions_integers = self._convert(action_converter.string2integer, actions_strings)
        self._tokens_strings = tokens_strings
        self._tokens_integers = self._convert(token_converter.token2integer, tokens_strings)
        self._device = device
        self._batch_size = batch_size
        self._shuffle = shuffle

    def _convert(self, converter, sequences):
        converted_sequences = []
        for sequence in sequences:
            converted_sequence = list(map(converter, sequence))
            converted_sequences.append(converted_sequence)
        return converted_sequences

    def __iter__(self):
        if self._shuffle:
            actions_integers, actions, tokens_integers, tokens_strings = self._shuffle_lists(
                self._actions_integers,
                self._actions,
                self._tokens_integers,
                self._tokens_strings
            )
        else:
            actions_integers = self._actions_integers
            actions = self._actions
            tokens_integers = self._tokens_integers
            tokens_strings = self._tokens_strings
        return UnorderedIterable(tokens_integers, tokens_strings, actions_integers, actions, self._device, self._batch_size)

    def _shuffle_lists(self, *lists):
        zipped = list(zip(*lists))
        random.shuffle(zipped)
        return zip(*zipped)

class UnorderedIterable:

    def __init__(self, tokens_integers, tokens_strings, actions_integers, actions, device, batch_size):
        """
        :type tokens_integers: list of list of int
        :type tokens_strings: list of list of str
        :type actions_integers: list of list of int
        :type actions: list of list of app.actions.action.Action
        :type device: torch.device
        :type batch_size: int
        """
        self._tokens_integers = tokens_integers
        self._tokens_strings = tokens_strings
        self._actions_integers = actions_integers
        self._actions = actions
        self._batch_size = batch_size
        self._device = device
        self._counter = 0
        self._total = len(actions_integers)

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter < self._total:
            start = self._counter
            end = min(self._counter + self._batch_size, self._total)
            tokens_integers = list(map(self._to_tensor, self._tokens_integers[start:end]))
            tokens_strings = list(self._tokens_strings[start:end])
            actions_integers = list(map(self._to_tensor, self._actions_integers[start:end]))
            actions = list(self._actions[start:end])
            # loop back to beginning if batch exceeds final observation
            # this ensures that batches always have a fixed length
            remaining = self._batch_size - (end - start)
            if 0 < remaining:
                self._extend(tokens_integers, self._tokens_integers, remaining, self._to_tensor)
                self._extend(tokens_strings, self._tokens_strings, remaining)
                self._extend(actions_integers, self._actions_integers, remaining, self._to_tensor)
                self._extend(actions, self._actions, remaining)
            self._counter += self._batch_size
            tokens_integers_padded, tokens_lengths = self._pad(tokens_integers)
            actions_integers_padded, actions_lengths = self._pad(actions_integers)
            output_tokens = (tokens_integers_padded, tokens_lengths, tokens_strings)
            output_actions = (actions_integers_padded, actions_lengths, actions)
            return output_tokens, output_actions
        else:
            raise StopIteration()

    def _to_tensor(self, values):
        sequence_length = len(values)
        shape = (sequence_length,)
        return torch.tensor(values, device=self._device, dtype=torch.long).reshape(shape)

    def _extend(self, sequence, items, remaining, transform=None):
        extension = items[0:remaining]
        if transform is not None:
            extension = list(map(transform, extension))
        sequence.extend(extension)

    def _pad(self, tensors):
        padded = pad_sequence(tensors, batch_first=False, padding_value=PAD_INDEX)
        lengths = torch.tensor(list(map(len, tensors)), dtype=torch.long, device=self._device)
        return padded, lengths
