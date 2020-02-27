from app.data.iterators.iterable import Iterable
from app.data.iterators.iterator import Iterator
from functools import partial

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
        return Iterable(tokens_integers, tokens_strings, actions_integers, actions, self._device, self._batch_size)

    def size(self):
        """
        :rtype: int
        """
        return len(self._actions)
