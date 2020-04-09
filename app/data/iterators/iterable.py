from app.data.batch import Batch
from app.data.batch_utils import int_sequences2tensor, sequences2lengths

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
            tokens_integers = self._tokens_integers[start:end]
            tokens_strings = list(self._tokens_strings[start:end])
            actions_integers = self._actions_integers[start:end]
            actions = list(self._actions[start:end])
            tags_integers = self._tags_integers[start:end]
            tags_strings = list(self._tags_strings[start:end])
            self._counter += end - start
            tokens_integers_padded = int_sequences2tensor(self._device, tokens_integers)
            tokens_lengths = sequences2lengths(self._device, tokens_integers)
            actions_integers_padded = int_sequences2tensor(self._device, actions_integers)
            actions_lengths = sequences2lengths(self._device, actions_integers)
            tags_integers_padded = int_sequences2tensor(self._device, tags_integers)
            tags_lengths = sequences2lengths(self._device, tags_integers)
            batch = Batch(
                actions_integers_padded, actions_lengths, actions,
                tokens_integers_padded, tokens_lengths, tokens_strings,
                tags_integers_padded, tags_lengths, tags_strings
            )
            return batch
        else:
            raise StopIteration()

    def _extend(self, sequence, items, remaining, transform=None):
        extension = items[0:remaining]
        if transform is not None:
            extension = list(map(transform, extension))
        sequence.extend(extension)
