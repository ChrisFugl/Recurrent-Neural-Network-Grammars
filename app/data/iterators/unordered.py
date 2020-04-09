from app.data.batch_utils import map_sequences
from app.data.iterators.iterable import Iterable
from app.data.iterators.iterator import Iterator

class UnorderedIterator(Iterator):

    def __init__(self, device, action_converter, token_converter, tag_converter, batch_size, shuffle, tokens, actions_strings, tags):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TokenConverter
        :type batch_size: int
        :type shuffle: bool
        :type tokens: list of list of str
        :type actions_strings: list of list of str
        :type tags: list of list of str
        """
        super().__init__()
        self._actions = map_sequences(action_converter.string2action, actions_strings)
        self._actions_integers = map_sequences(action_converter.string2integer, actions_strings)
        self._tokens_strings = tokens
        self._tokens_integers = map_sequences(token_converter.token2integer, tokens)
        self._tags_strings = tags
        self._tags_integers = map_sequences(tag_converter.tag2integer, tags)
        self._device = device
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            actions_integers, actions, tokens_integers, tokens_strings, tags_integers, tags_strings = self._shuffle_lists(
                self._actions_integers, self._actions,
                self._tokens_integers, self._tokens_strings,
                self._tags_integers, self._tags_strings
            )
        else:
            actions_integers = self._actions_integers
            actions = self._actions
            tokens_integers = self._tokens_integers
            tokens_strings = self._tokens_strings
            tags_integers = self._tags_integers
            tags_strings = self._tags_strings
        return Iterable(tokens_integers, tokens_strings, actions_integers, actions, tags_integers, tags_strings, self._device, self._batch_size)

    def size(self):
        """
        :rtype: int
        """
        return len(self._actions)
