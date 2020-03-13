from app.data.iterators.iterable import Iterable
from app.data.iterators.iterator import Iterator
from functools import partial
from math import ceil
from operator import itemgetter

class OrderedIterator(Iterator):
    """
    Batches are ordered such that actions in a batch strive to have the same length.
    """

    def __init__(self, device, action_converter, token_converter, tag_converter, batch_size, shuffle, tokens_strings, actions_strings, tags):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type batch_size: int
        :type shuffle: bool
        :type tokens_strings: list of list of str
        :type actions_strings: list of list of str
        :type tags: list of list of str
        """
        super().__init__()
        self._device = device
        self._batch_size = batch_size
        self._shuffle = shuffle

        actions = self._convert(partial(action_converter.string2action, device), actions_strings)
        actions_integers = self._convert(action_converter.string2integer, actions_strings)
        tokens_integers = self._convert(token_converter.token2integer, tokens_strings)
        tags_integers = self._convert(tag_converter.tag2integer, tags)

        ordered = self._order_by_actions_count(actions_integers, actions, tokens_integers, tokens_strings, tags, tags_integers)
        self._actions_counts = ordered[0]
        if shuffle:
            self._actions_counts_set = set(self._actions_counts)
        self._actions_integers = ordered[1]
        self._actions = ordered[2]
        self._tokens_integers = ordered[3]
        self._tokens_strings = ordered[4]
        self._tags_strings = ordered[5]
        self._tags_integers = ordered[6]

    def __iter__(self):
        actions_integers, actions = self._actions_integers, self._actions
        tokens_integers, tokens_strings = self._tokens_integers, self._tokens_strings
        tags_strings, tags_integers = self._tags_strings, self._tags_integers
        if self._shuffle:
            self._shuffle_by_action_count(
                self._actions_integers, self._actions,
                self._tokens_integers, self._tokens_strings,
                self._tags_integers, self._tags_strings
            )
            batches = self._create_batches(actions_integers, actions, tokens_integers, tokens_strings, tags_strings, tags_integers)
            shuffled_batches = self._shuffle_lists(*batches)
            flattened = self._flatten_all_batches(*shuffled_batches)
            actions_integers, actions = flattened[0], flattened[1]
            tokens_integers, tokens_strings = flattened[2], flattened[3]
            tags_integers, tags_strings = flattened[4], flattened[5]
        return Iterable(tokens_integers, tokens_strings, actions_integers, actions, tags_integers, tags_strings, self._device, self._batch_size)

    def size(self):
        """
        :rtype: int
        """
        return len(self._actions)

    def _order_by_actions_count(self, action_ints, actions, token_ints, token_strs, tags):
        actions_counts = list(map(len, actions))
        zipped_lists = zip(actions_counts, action_ints, actions, token_ints, token_strs, tags)
        ordered_lists = sorted(zipped_lists, key=itemgetter(0))
        ordered_tuples = zip(*ordered_lists)
        return tuple(map(list, ordered_tuples))

    def _last_index(self, list, item):
        return max(index for index, value in enumerate(list) if value == item)

    def _shuffle_by_action_count(self, actions_integers, actions, tokens_integers, tokens_strings, tags):
        for action_count in self._actions_counts_set:
            start = self._actions_counts.index(action_count)
            end = self._last_index(self._actions_counts, action_count) + 1
            actions_ints_slice, actions_slice, tokens_ints_slice, tokens_strs_slice, tags_slice = self._shuffle_lists(
                actions_integers[start:end],
                actions[start:end],
                tokens_integers[start:end],
                tokens_strings[start:end],
                tags[start:end],
            )
            actions_integers[start:end] = actions_ints_slice
            actions[start:end] = actions_slice
            tokens_integers[start:end] = tokens_ints_slice
            tokens_strings[start:end] = tokens_strs_slice
            tags[start:end] = tags_slice

    def _create_batches(self, actions_integers, actions, tokens_integers, tokens_strings, tags):
        batches_ai, batches_a, batches_ti, batches_ts, batches_ta = [], [], [], [], []
        observations_count = len(actions_integers)
        batch_count = ceil(observations_count / self._batch_size)
        for batch_index in range(batch_count):
            start = batch_index * self._batch_size
            end = (batch_index + 1) * self._batch_size if batch_index < batch_count - 1 else observations_count
            batches_ai.append(actions_integers[start:end])
            batches_a.append(actions[start:end])
            batches_ti.append(tokens_integers[start:end])
            batches_ts.append(tokens_strings[start:end])
            batches_ta.append(tags[start:end])
        return batches_ai, batches_a, batches_ti, batches_ts, batches_ta

    def _flatten_all_batches(self, batches_ai, batches_a, batches_ti, batches_ts, batches_ta):
        return (
            self._flatten_batches(batches_ai),
            self._flatten_batches(batches_a),
            self._flatten_batches(batches_ti),
            self._flatten_batches(batches_ts),
            self._flatten_batches(batches_ta)
        )

    def _flatten_batches(self, batches):
        flattened = []
        for batch in batches:
            flattened.extend(batch)
        return flattened
