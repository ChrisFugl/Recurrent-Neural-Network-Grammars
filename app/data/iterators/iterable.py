from app.constants import PAD_INDEX
from app.data.batch import Batch
import torch
from torch.nn.utils.rnn import pad_sequence

class Iterable:

    def __init__(self, tokens_integers, tokens_strings, actions_integers, actions, tags_integers, tags_strings, device, batch_size):
        """
        :type tokens_integers: list of list of int
        :type tokens_strings: list of list of str
        :type actions_integers: list of list of int
        :type actions: list of list of app.actions.action.Action
        :type tags_integers: list of list of int
        :type tags_strings: list of list of str
        :type device: torch.device
        :type batch_size: int
        """
        self._tokens_integers = tokens_integers
        self._tokens_strings = tokens_strings
        self._actions_integers = actions_integers
        self._actions = actions
        self._tags_integers = tags_integers
        self._tags_strings = tags_strings
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
            tags_integers = list(map(self._to_tensor, self._tags_integers[start:end]))
            tags_strings = list(self._tags_strings[start:end])
            self._counter += end - start
            tokens_integers_padded, tokens_lengths = self._pad(tokens_integers)
            actions_integers_padded, actions_lengths = self._pad(actions_integers)
            tags_integers_padded, tags_lengths = self._pad(tags_integers)
            batch = Batch(
                actions_integers_padded, actions_lengths, actions,
                tokens_integers_padded, tokens_lengths, tokens_strings,
                tags_integers_padded, tags_lengths, tags_strings
            )
            return batch
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
