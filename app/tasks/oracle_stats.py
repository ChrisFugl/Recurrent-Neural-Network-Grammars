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
        trees_train, actions_train, terms_train, unknownified_terms_train = self._loader.load_train()
        trees_val, actions_val, terms_val, unknownified_terms_val = self._loader.load_val()
        trees_test, actions_test, terms_test, unknownified_terms_test = self._loader.load_test()

        # get stats
        sequences_count = self._count_sequences([trees_train, trees_val, trees_test])
        tokens_count = self._count_tokens([terms_train, terms_val, terms_test])
        unique_tokens_count = self._count_unique_tokens([terms_train, terms_val, terms_test])
        unknown_types_count = self._count_unknown_types([unknownified_terms_train, unknownified_terms_val, unknownified_terms_test])

        # print table
        rows = [sequences_count, tokens_count, unique_tokens_count, unknown_types_count]
        headers = ['Train', 'Val', 'Test']
        table = tabulate(rows, headers)
        print(table)

    def _count_sequences(self, data):
        counts = map(len, data)
        return ['Sequences', *counts]

    def _count_tokens(self, data):
        counts = map(lambda tokens: sum(map(len, tokens)), data)
        return ['Tokens', *counts]

    def _count_unique_tokens(self, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_set = set(tokens_flattened)
            counts[index] = len(tokens_set)
        return ['Unique tokens', *counts]

    def _count_unknown_types(self, data):
        counts = [0, 0, 0]
        for index, unknown_tokens in enumerate(data):
            unknown_tokens_flattened = reduce(iconcat, unknown_tokens, [])
            unknown_tokens_filtered = filter(self._is_unknown_token, unknown_tokens_flattened)
            unknown_tokens_set = set(unknown_tokens_filtered)
            counts[index] = len(unknown_tokens_set)
        return ['Unknown types', *counts]

    def _is_unknown_token(self, token):
        return token.startswith('<UNK')
