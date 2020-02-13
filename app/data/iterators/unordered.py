from app.data.iterators.iterator import Iterator
import random
import torch

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
            actions, tokens = self._shuffle_two_lists(self._actions, self._tokens)
        else:
            actions = self._actions
            tokens = self._tokens
        return UnorderedIterable(tokens, actions, self._device, self._batch_size)

    def _shuffle_two_lists(self, list1, list2):
        zipped = list(zip(list1, list2))
        random.shuffle(zipped)
        return zip(*zipped)

class UnorderedIterable:

    def __init__(self, tokens, actions, device, batch_size):
        """
        :type tokens: torch.tensor
        :type actions: torch.tensor
        :type device: torch.device
        :type batch_size: int
        """
        self._tokens = tokens
        self._actions = actions
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
            sentences = list(map(self._to_tensor, self._tokens[start:end]))
            trees = list(map(self._to_tensor, self._actions[start:end]))
            self._counter += self._batch_size
            return sentences, trees
        else:
            raise StopIteration()

    def _to_tensor(self, values):
        return torch.tensor(values, device=self._device, dtype=torch.int)
