from app.data.batch_utils import map_sequences
from app.data.iterators.iterable import Iterable
from app.data.iterators.iterator import Iterator
from math import ceil
from operator import itemgetter

class OrderedIterator(Iterator):
    """
    Batches are ordered such that actions in a batch strive to have the same length.
    """

    def __init__(self,
        device, action_converter, token_converter, tag_converter, batch_size, shuffle,
        tokens, unknownified_tokens, actions_strings, tags
    ):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type batch_size: int
        :type shuffle: bool
        :type tokens: list of list of str
        :type unknownified_tokens: list of list of str
        :type actions_strings: list of list of str
        :type tags: list of list of str
        """
        super().__init__()
        self._device = device
        self._batch_size = batch_size
        self._shuffle = shuffle

        actions = map_sequences(action_converter.string2action, actions_strings)
        actions_integers = map_sequences(action_converter.string2integer, actions_strings)
        tokens_integers = map_sequences(token_converter.token2integer, tokens)
        unknownified_tokens_integers = map_sequences(token_converter.token2integer, unknownified_tokens)
        singletons = map_sequences(token_converter.is_singleton, tokens)
        tags_integers = map_sequences(tag_converter.tag2integer, tags)

        ordered = self._order_by_actions_count(
            actions_integers, actions,
            tokens_integers, tokens,
            unknownified_tokens_integers, unknownified_tokens,
            singletons,
            tags_integers, tags
        )
        self._actions_counts = ordered[0]
        if shuffle:
            self._actions_counts_set = set(self._actions_counts)
        self._actions_integers = ordered[1]
        self._actions = ordered[2]
        self._tokens_integers = ordered[3]
        self._tokens_strings = ordered[4]
        self._unknownified_tokens_integers = ordered[5]
        self._unknownified_tokens_strings = ordered[6]
        self._singletons = ordered[7]
        self._tags_integers = ordered[8]
        self._tags_strings = ordered[9]

    def __iter__(self):
        actions_integers, actions = self._actions_integers, self._actions
        tokens_integers, tokens_strings = self._tokens_integers, self._tokens_strings
        unknownified_tokens_integers, unknownified_tokens_strings = self._unknownified_tokens_integers, self._unknownified_tokens_strings
        singletons = self._singletons
        tags_strings, tags_integers = self._tags_strings, self._tags_integers
        if self._shuffle:
            self._shuffle_by_action_count(
                self._actions_integers, self._actions,
                self._tokens_integers, self._tokens_strings,
                self._unknownified_tokens_integers, self._unknownified_tokens_strings,
                self._singletons,
                self._tags_integers, self._tags_strings
            )
            batches = self._create_batches(
                actions_integers, actions,
                tokens_integers, tokens_strings,
                unknownified_tokens_integers, unknownified_tokens_strings,
                singletons,
                tags_integers, tags_strings
            )
            shuffled_batches = self._shuffle_lists(*batches)
            flattened = self._flatten_all_batches(*shuffled_batches)
            actions_integers, actions = flattened[0], flattened[1]
            tokens_integers, tokens_strings = flattened[2], flattened[3]
            unknownified_tokens_integers, unknownified_tokens_strings = flattened[4], flattened[5]
            singletons = flattened[6]
            tags_integers, tags_strings = flattened[7], flattened[8]
        return Iterable(
            tokens_integers, tokens_strings,
            unknownified_tokens_integers, unknownified_tokens_strings,
            singletons,
            actions_integers, actions,
            tags_integers, tags_strings,
            self._device, self._batch_size
        )

    def get_batch_size(self):
        """
        :rtype: int
        """
        return self._batch_size

    def size(self):
        """
        :rtype: int
        """
        return len(self._actions)

    def _order_by_actions_count(self, action_ints, actions, token_ints, token_strs, unk_token_ints, unk_token_strs, singletons, tags_integers, tags):
        actions_counts = list(map(len, actions))
        zipped_lists = zip(actions_counts, action_ints, actions, token_ints, token_strs, unk_token_ints, unk_token_strs, singletons, tags_integers, tags)
        ordered_lists = sorted(zipped_lists, key=itemgetter(0))
        ordered_tuples = zip(*ordered_lists)
        return tuple(map(list, ordered_tuples))

    def _last_index(self, list, item):
        return max(index for index, value in enumerate(list) if value == item)

    def _shuffle_by_action_count(self,
        actions_integers, actions,
        tokens_integers, tokens_strings,
        unk_tokens_integers, unk_tokens_strings,
        singletons,
        tags_integers, tags
    ):
        for action_count in self._actions_counts_set:
            start = self._actions_counts.index(action_count)
            end = self._last_index(self._actions_counts, action_count) + 1
            slices = self._shuffle_lists(
                actions_integers[start:end],
                actions[start:end],
                tokens_integers[start:end],
                tokens_strings[start:end],
                unk_tokens_integers[start:end],
                unk_tokens_strings[start:end],
                singletons[start:end],
                tags_integers[start:end],
                tags[start:end],
            )
            actions_integers[start:end] = slices[0]
            actions[start:end] = slices[1]
            tokens_integers[start:end] = slices[2]
            tokens_strings[start:end] = slices[3]
            unk_tokens_integers[start:end] = slices[4]
            unk_tokens_strings[start:end] = slices[5]
            singletons[start:end] = slices[6]
            tags_integers[start:end] = slices[7]
            tags[start:end] = slices[8]

    def _create_batches(self,
        actions_integers, actions,
        tokens_integers, tokens_strings,
        unk_tokens_integers, unk_tokens_strings,
        singletons,
        tags_integers, tags
    ):
        batches = [], [], [], [], [], [], [], [], []
        observations_count = len(actions_integers)
        batch_count = ceil(observations_count / self._batch_size)
        for batch_index in range(batch_count):
            start = batch_index * self._batch_size
            end = (batch_index + 1) * self._batch_size if batch_index < batch_count - 1 else observations_count
            batches[0].append(actions_integers[start:end])
            batches[1].append(actions[start:end])
            batches[2].append(tokens_integers[start:end])
            batches[3].append(tokens_strings[start:end])
            batches[4].append(unk_tokens_integers[start:end])
            batches[5].append(unk_tokens_strings[start:end])
            batches[6].append(singletons[start:end])
            batches[7].append(tags_integers[start:end])
            batches[8].append(tags[start:end])
        return batches

    def _flatten_all_batches(self, batches_ai, batches_a, batches_ti, batches_ts, batches_uti, batches_uts, batches_s, batches_tai, batches_tas):
        return (
            self._flatten_batches(batches_ai),
            self._flatten_batches(batches_a),
            self._flatten_batches(batches_ti),
            self._flatten_batches(batches_ts),
            self._flatten_batches(batches_uti),
            self._flatten_batches(batches_uts),
            self._flatten_batches(batches_s),
            self._flatten_batches(batches_tai),
            self._flatten_batches(batches_tas),
        )

    def _flatten_batches(self, batches):
        flattened = []
        for batch in batches:
            flattened.extend(batch)
        return flattened
