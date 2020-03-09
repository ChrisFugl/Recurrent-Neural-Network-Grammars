from app.tasks.task import Task
from functools import reduce
from operator import iconcat
from tabulate import tabulate

class OracleStatsTask(Task):
    """
    Collect following stats for training, validation, and testset:
    * #sequences
    * #tokens
    * #unique tokens
    * #unknown types
    """

    def __init__(self, loader):
        """
        :type loader: app.data.loaders.loader.Loader
        """
        super().__init__()
        self._loader = loader

    def run(self):
        trees_train, actions_train, terms_train, unknownified_terms_train, _ = self._loader.load_train()
        trees_val, actions_val, terms_val, unknownified_terms_val, _ = self._loader.load_val()
        trees_test, actions_test, terms_test, unknownified_terms_test, _ = self._loader.load_test()

        trees = [trees_train, trees_val, trees_test]
        actions = [actions_train, actions_val, actions_test]
        terms = [terms_train, terms_val, terms_test]
        unknown_terms = [unknownified_terms_train, unknownified_terms_val, unknownified_terms_test]

        headers = ['Train', 'Val', 'Test']
        rows = [
            self._count_sequences(trees),
            self._count_tokens(unknown_terms),
            self._count_tokens_without_punctuation(unknown_terms),
            self._count_unique_tokens('Unique tokens', terms),
            self._count_unique_tokens_without_punctuation('Unique tokens without punctuation', terms),
            self._count_unique_tokens('Unique unknownified tokens', unknown_terms),
            self._count_unique_tokens_without_punctuation('Unique unknownified tokens without punctuation', unknown_terms),
            self._count_unknown_types(unknown_terms),
            self._count_non_terminals(actions),
            self._count_unique_non_terminals(actions),
            self._count_unknown_non_terminals_types(actions),
        ]

        colalign = ['left', 'right', 'right', 'right']
        print(tabulate(rows, headers, tablefmt='fancy_grid', colalign=colalign))

    def _count_sequences(self, data):
        counts = map(lambda sequences: f'{len(sequences):,}', data)
        return ['Sequences', *counts]

    def _count_tokens(self, data):
        counts = map(lambda tokens: f'{sum(map(len, tokens)):,}', data)
        return ['Tokens', *counts]

    def _count_tokens_without_punctuation(self, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_without_punctuation = list(filter(self._is_not_punctuation, tokens_flattened))
            counts[index] = f'{len(tokens_without_punctuation):,}'
        return ['Tokens without punctuation', *counts]

    def _count_unique_tokens(self, name, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_set = set(tokens_flattened)
            counts[index] = f'{len(tokens_set):,}'
        return [name, *counts]

    def _count_unique_tokens_without_punctuation(self, name, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_without_punctuation = list(filter(self._is_not_punctuation, tokens_flattened))
            tokens_set = set(tokens_without_punctuation)
            counts[index] = f'{len(tokens_set):,}'
        return [name, *counts]

    def _count_unknown_types(self, data):
        counts = [0, 0, 0]
        for index, unknown_tokens in enumerate(data):
            unknown_tokens_flattened = reduce(iconcat, unknown_tokens, [])
            unknown_tokens_filtered = filter(self._is_unknown_token, unknown_tokens_flattened)
            unknown_tokens_set = set(unknown_tokens_filtered)
            counts[index] = f'{len(unknown_tokens_set):,}'
        return ['Unknown types', *counts]

    def _count_non_terminals(self, data):
        counts = [0, 0, 0]
        for index, actions in enumerate(data):
            flattened = reduce(iconcat, actions, [])
            non_terminals = list(filter(self._is_non_terminal, flattened))
            counts[index] = f'{len(non_terminals):,}'
        return ['Non-terminals', *counts]

    def _count_unique_non_terminals(self, data):
        counts = [0, 0, 0]
        for index, actions in enumerate(data):
            flattened = reduce(iconcat, actions, [])
            non_terminals = set(filter(self._is_non_terminal, flattened))
            counts[index] = f'{len(non_terminals):,}'
        return ['Unique non-terminals', *counts]

    def _count_unknown_non_terminals_types(self, data):
        counts = [0, 0, 0]
        for index, actions in enumerate(data):
            flattened = reduce(iconcat, actions, [])
            unknown_non_terminals = set(filter(self._is_unknown_non_terminal, flattened))
            counts[index] = f'{len(unknown_non_terminals):,}'
        return ['Unknown non-terminal types', *counts]

    def _is_not_punctuation(self, token):
        return token not in ['.', '?', '!', ',', ';', ':', '--', '-', '(', ')', '[', ']', '{', '}', "'", '"', '...']

    def _is_unknown_token(self, token):
        return token.startswith('<UNK')

    def _is_non_terminal(self, action):
        return action.startswith('NT(')

    def _is_unknown_non_terminal(self, action):
        return action.startswith('NT(<UNK')
