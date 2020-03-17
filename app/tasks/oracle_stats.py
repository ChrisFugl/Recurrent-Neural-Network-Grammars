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
        trees_train, actions_train, terms_train, unknownified_terms_train, tags_train = self._loader.load_train()
        trees_val, actions_val, terms_val, unknownified_terms_val, tags_val = self._loader.load_val()
        trees_test, actions_test, terms_test, unknownified_terms_test, tags_test = self._loader.load_test()
        trees = [trees_train, trees_val, trees_test]
        actions = [actions_train, actions_val, actions_test]
        terms = [terms_train, terms_val, terms_test]
        unknown_terms = [unknownified_terms_train, unknownified_terms_val, unknownified_terms_test]
        tags = [tags_train, tags_val, tags_test]
        headers = ['Train', 'Val', 'Test']
        rows = [
            self._count_sequences(trees),
            self._count_strings('Tokens', unknown_terms),
            self._count_strings_without_punctuation('Tokens without punctuation', unknown_terms),
            self._count_unique_strings('Unique tokens', terms),
            self._count_unique_strings_without_punctuation('Unique tokens without punctuation', terms),
            self._count_unique_strings('Unique unknownified tokens', unknown_terms),
            self._count_unique_strings_without_punctuation('Unique unknownified tokens without punctuation', unknown_terms),
            self._count_unknown_types(unknown_terms),
            self._count_strings('Part-of-speech tags', tags),
            self._count_unique_strings('Unique part-of-speech tags', tags),
            self._count_non_terminals(actions),
            self._count_unique_non_terminals(actions),
            self._count_unknown_non_terminals_types(actions),
            self._max_children_nodes(actions),
            self._max_open_non_terminals(actions),
            self._min_length('Min actions', actions),
            self._mean_length('Mean actions', actions),
            self._max_length('Max actions', actions),
            self._min_length('Min tokens', unknown_terms),
            self._mean_length('Mean tokens', unknown_terms),
            self._max_length('Max tokens', unknown_terms),
        ]
        colalign = ['left', 'right', 'right', 'right']
        print(tabulate(rows, headers, tablefmt='github', colalign=colalign))

    def _count_sequences(self, data):
        counts = map(lambda sequences: f'{len(sequences):,}', data)
        return ['Sequences', *counts]

    def _count_strings(self, name, data):
        counts = map(lambda tokens: f'{sum(map(len, tokens)):,}', data)
        return [name, *counts]

    def _count_strings_without_punctuation(self, name, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_without_punctuation = list(filter(self._is_not_punctuation, tokens_flattened))
            counts[index] = f'{len(tokens_without_punctuation):,}'
        return [name, *counts]

    def _count_unique_strings(self, name, data):
        counts = [0, 0, 0]
        for index, tokens in enumerate(data):
            tokens_flattened = reduce(iconcat, tokens, [])
            tokens_set = set(tokens_flattened)
            counts[index] = f'{len(tokens_set):,}'
        return [name, *counts]

    def _count_unique_strings_without_punctuation(self, name, data):
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

    def _max_children_nodes(self, data):
        counts = [0, 0, 0]
        for index, sequences in enumerate(data):
            max_children_nodes = 0
            for actions in sequences:
                tree = self._actions2tree(actions)
                max_from_tree = self._max_children_nodes_from_tree(tree)
                max_children_nodes = max(max_children_nodes, max_from_tree)
            counts[index] = max_children_nodes
        return ['Max children nodes', *counts]

    def _max_children_nodes_from_tree(self, tree):
        if tree is None or tree.children is None:
            return 0
        children = map(self._max_children_nodes_from_tree, tree.children)
        return max(len(tree.children), *children)

    def _max_open_non_terminals(self, data):
        counts = [0, 0, 0]
        for index, sequences in enumerate(data):
            max_open_nt = 0
            for actions in sequences:
                open_nt = 0
                for action in actions:
                    if self._is_non_terminal(action):
                        open_nt += 1
                        max_open_nt = max(max_open_nt, open_nt)
                    elif self._is_reduce(action):
                        open_nt -= 1
            counts[index] = max_open_nt
        return ['Max open non-terminals', *counts]

    def _max_length(self, name, data):
        lengths = map(lambda sequences: map(len, sequences), data)
        max_lengths = map(lambda sequences: max(sequences), lengths)
        return [name, *max_lengths]

    def _min_length(self, name, data):
        lengths = map(lambda sequences: map(len, sequences), data)
        min_lengths = map(lambda sequences: min(sequences), lengths)
        return [name, *min_lengths]

    def _mean_length(self, name, data):
        lengths = map(lambda sequences: list(map(len, sequences)), data)
        mean_lengths = map(lambda sequences: f'{sum(sequences) / len(sequences):0.2f}', lengths)
        return [name, *mean_lengths]

    def _is_not_punctuation(self, token):
        return token not in ['.', '?', '!', ',', ';', ':', '--', '-', '(', ')', '[', ']', '{', '}', "'", '"', '...']

    def _is_unknown_token(self, token):
        return token.startswith('<UNK')

    def _is_non_terminal(self, action):
        return action.startswith('NT(')

    def _is_unknown_non_terminal(self, action):
        return action.startswith('NT(<UNK')

    def _is_reduce(self, action):
        return action == 'REDUCE'

    def _actions2tree(self, actions):
        # first action creates the root of the tree
        root = TreeNode(value=actions[0])
        parent = root
        for action in actions[1:]:
            if self._is_reduce(action):
                parent = parent.parent
            elif self._is_non_terminal(action):
                node = TreeNode(action, parent=parent)
                parent.add_child(node)
                parent = node
            else:
                node = TreeNode(action, parent=parent)
                parent.add_child(node)
        return root

class TreeNode:

    def __init__(self, value, parent=None, children=None):
        self.value = value
        self.parent = parent
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
