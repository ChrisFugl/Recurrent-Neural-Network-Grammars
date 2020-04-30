from app.data.batch import Batch
from app.data.batch_utils import int_sequences2tensor, sequences2lengths
import torch
from torch.nn.utils.rnn import pad_sequence

class Iterable:

    def __init__(self,
        tokens_integers, tokens_strings,
        unknownified_tokens_integers, unknownified_tokens_strings,
        singletons,
        actions_integers, actions,
        tags_integers, tags_strings,
        device, batch_size
    ):
        """
        :type tokens_integers: list of list of int
        :type tokens_strings: list of list of str
        :type unknownified_tokens_integers: list of list of int
        :type unknownified_tokens_strings: list of list of str
        :type singletons: list of list bool
        :type actions_integers: list of list of int
        :type actions: list of list of app.actions.action.Action
        :type tags_integers: list of list of int
        :type tags_strings: list of list of str
        :type device: torch.device
        :type batch_size: int
        """
        self._tokens_integers = tokens_integers
        self._tokens_strings = tokens_strings
        self._unknownified_tokens_integers = unknownified_tokens_integers
        self._unknownified_tokens_strings = unknownified_tokens_strings
        self._singletons = singletons
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
            actions_integers = self._actions_integers[start:end]
            actions = list(self._actions[start:end])
            tokens_integers = self._tokens_integers[start:end]
            tokens_strings = list(self._tokens_strings[start:end])
            unknownified_tokens_integers = self._unknownified_tokens_integers[start:end]
            unknownified_tokens_strings = list(self._unknownified_tokens_strings[start:end])
            singletons = list(self._singletons[start:end])
            tags_integers = self._tags_integers[start:end]
            tags_strings = list(self._tags_strings[start:end])
            self._counter += end - start
            actions_integers_padded = int_sequences2tensor(self._device, actions_integers)
            actions_lengths = sequences2lengths(self._device, actions_integers)
            tokens_integers_padded = int_sequences2tensor(self._device, tokens_integers)
            tokens_lengths = sequences2lengths(self._device, tokens_integers)
            unknownified_tokens_integers_padded = int_sequences2tensor(self._device, unknownified_tokens_integers)
            singleton_tensors = [torch.tensor(singleton, device=self._device, dtype=torch.bool) for singleton in singletons]
            singletons_padded = pad_sequence(singleton_tensors, padding_value=0)
            tags_integers_padded = int_sequences2tensor(self._device, tags_integers)
            batch = Batch(
                actions_integers_padded, actions_lengths, actions,
                tokens_integers_padded, tokens_lengths, tokens_strings,
                unknownified_tokens_integers_padded, unknownified_tokens_strings,
                singletons_padded,
                tags_integers_padded, tags_strings
            )
            return batch
        else:
            raise StopIteration()
