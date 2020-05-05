from app.constants import ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE

class StateFactory:

    def __init__(self, action_converter):
        """
        :type action_converter: app.data.converters.action.ActionConvter
        """
        self.action_converter = action_converter

    def initialize(self, rnn_state, tokens_lengths):
        """
        :type tokens_lengths: torch.Tensor
        :rtype: app.models.rnn_lm.state.State
        """
        batch_size = tokens_lengths.size(0)
        tokens_lengths_list = [length.cpu().item() for length in tokens_lengths]
        token_counter = [0 for _ in range(batch_size)]
        last_action = [None for _ in range(batch_size)]
        open_nt_count = [0 for _ in range(batch_size)]
        parent_node = [None for _ in range(batch_size)]
        return State(rnn_state, tokens_lengths_list, batch_size, token_counter, last_action, open_nt_count, parent_node)

    def next(self, state, rnn_state, actions):
        """
        :type state: app.models.rnn_lm.state.State
        :type actions: list of app.data.actions.action.Action
        :rtype: app.models.rnn_lm.state.State
        """
        for i, action in enumerate(actions):
            if action is not None:
                type = action.type()
                parent = state.parent_node[i]
                node = Tree(action, parent=parent)
                if type == ACTION_SHIFT_TYPE:
                    state.token_counter[i] = state.token_counter[i] + 1
                    state.last_action[i] = action
                    parent.add_child(node)
                elif type == ACTION_REDUCE_TYPE:
                    state.open_nt_count[i] = state.open_nt_count[i] - 1
                    state.parent_node[i] = parent.parent
                    state.last_action[i] = self.action_converter.get_cached_nt_action(parent.action.argument, False)
                elif type == ACTION_NON_TERMINAL_TYPE:
                    if parent is not None:
                        parent.add_child(node)
                    state.open_nt_count[i] = state.open_nt_count[i] + 1
                    state.parent_node[i] = node
                    state.last_action[i] = self.action_converter.get_cached_nt_action(action.argument, True)
                else:
                    raise Exception(f'Unknown action type: {type}')
        return State(rnn_state, state.tokens_lengths, state.batch_size, state.token_counter, state.last_action, state.open_nt_count, state.parent_node)

class State:

    def __init__(self, rnn_state, tokens_lengths, batch_size, token_counter, last_action, open_nt_count, parent_node):
        """
        :type tokens_lengths: list of int
        :type batch_size: int
        :type token_counter: list of int
        :type last_action: list of app.data.actions.action.Action
        :type open_nt_count: list of int
        :type parent_node: list of app.models.rnn_lm.state.Tree
        """
        self.rnn_state = rnn_state
        self.tokens_lengths = tokens_lengths
        self.batch_size = batch_size
        self.token_counter = token_counter
        self.last_action = last_action
        self.open_nt_count = open_nt_count
        self.parent_node = parent_node

class Tree:

    def __init__(self, action, parent=None, children=None):
        self.action = action
        self.parent = parent
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
