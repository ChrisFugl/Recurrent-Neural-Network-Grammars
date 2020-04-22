from app.constants import ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE, PAD_INDEX
import torch

class StateFactory:

    def __init__(self,
        device, action_converter, nt_converter, token_converter,
        action_set, generative, action_count, reduce_index, shift_index, gen_indices, nt_indices,
    ):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type nt_converter: app.data.converters.non_terminal.NonTerminalConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type action_set: app.data.action_sets.action_set.ActionSet
        :type generative: bool
        :type action_count: int
        :type reduce_index: int
        :type shift_index: int
        :type gen_indices: list of int
        :type nt_indices: list of int
        """
        self.device = device
        self.action_converter = action_converter
        self.nt_converter = nt_converter
        self.token_converter = token_converter
        self.action_set = action_set
        self.generative = generative
        self.action_count = action_count
        self.reduce_index = reduce_index
        self.shift_index = shift_index
        self.gen_indices = gen_indices
        self.nt_indices = nt_indices

    def initialize(self, batch_size, tokens_tensor, tags_tensor, tokens_lengths, make_invalid_mask=True):
        """
        :type batch_size: int
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type tokens_tensor_lengths: torch.Tensor
        :type make_invalid_mask: bool
        :rtype: app.models.parallel_rnng.state.State
        """
        parent_node = [None for _ in range(batch_size)]
        stack_size = [0 for _ in range(batch_size)]
        max_stack_size = 0
        shift_index = [0 for _ in range(batch_size)]
        token_counter = [0 for _ in range(batch_size)]
        last_action = [None for _ in range(batch_size)]
        open_nt_count = [0 for _ in range(batch_size)]
        invalid_mask = None
        if make_invalid_mask:
            invalid_mask = self.get_invalid_mask(tokens_lengths, token_counter, last_action, open_nt_count)
        state = State(
            tokens_tensor, tags_tensor,
            parent_node, stack_size, max_stack_size, shift_index,
            tokens_lengths, token_counter, last_action, open_nt_count,
            None, None, None, None, None, invalid_mask,
            None, None, None, None,
        )
        return state

    def next(self, state, actions, make_invalid_mask=True):
        """
        :type state: app.models.parallel_rnng.state.State
        :type actions: list of app.data.actions.action.Action
        :type make_invalid_mask: bool
        :rtype: app.models.parallel_rnng.state.State
        """
        batch_size = len(actions)
        max_stack_size = state.max_stack_size
        nt_index = [PAD_INDEX for _ in range(batch_size)]
        compose_nt_index = [PAD_INDEX for _ in range(batch_size)]
        token_index = [PAD_INDEX for _ in range(batch_size)]
        tag_index = [PAD_INDEX for _ in range(batch_size)]
        number_of_children = [0 for _ in range(batch_size)]
        nt_actions = []
        shift_actions = []
        gen_actions = []
        reduce_actions = []
        for i, action in enumerate(actions):
            if action is not None:
                type = action.type()
                parent = state.parent_node[i]
                node = Tree(action, parent=parent)
                if type == ACTION_SHIFT_TYPE:
                    shift_index = state.shift_index[i]
                    # token buffer processes terminals in reverse order
                    tokens_length = state.tokens_lengths[i]
                    shift_tensor_index = tokens_length - shift_index - 1
                    token_index[i] = state.tokens_tensor[shift_tensor_index, i]
                    tag_index[i] = state.tags_tensor[shift_tensor_index, i]
                    state.shift_index[i] = shift_index + 1
                    state.token_counter[i] = state.token_counter[i] + 1
                    state.stack_size[i] = state.stack_size[i] + 1
                    state.last_action[i] = action
                    parent.add_child(node)
                    shift_actions.append(i)
                elif type == ACTION_GENERATE_TYPE:
                    token_index[i] = self.token_converter.token2integer(action.argument)
                    state.token_counter[i] = state.token_counter[i] + 1
                    state.stack_size[i] = state.stack_size[i] + 1
                    parent.add_child(node)
                    state.last_action[i] = action
                    gen_actions.append(i)
                elif type == ACTION_NON_TERMINAL_TYPE:
                    if parent is not None:
                        parent.add_child(node)
                    nt_index[i] = self.nt_converter.non_terminal2integer(action.argument)
                    state.open_nt_count[i] = state.open_nt_count[i] + 1
                    state.stack_size[i] = state.stack_size[i] + 1
                    state.parent_node[i] = node
                    state.last_action[i] = self.action_converter.get_cached_nt_action(action.argument, True)
                    nt_actions.append(i)
                else:
                    compose_nt_index[i] = self.nt_converter.non_terminal2integer(parent.action.argument)
                    number_of_children[i] = len(parent.children)
                    state.open_nt_count[i] = state.open_nt_count[i] - 1
                    state.stack_size[i] = state.stack_size[i] - len(parent.children)
                    state.parent_node[i] = parent.parent
                    state.last_action[i] = self.action_converter.get_cached_nt_action(parent.action.argument, False)
                    reduce_actions.append(i)
                max_stack_size = max(max_stack_size, state.stack_size[i])
        nt_index_tensor = torch.tensor(nt_index, device=self.device, dtype=torch.long)
        compose_nt_index_tensor = torch.tensor(compose_nt_index, device=self.device, dtype=torch.long)
        token_index_tensor = torch.tensor(token_index, device=self.device, dtype=torch.long)
        tag_index_tensor = torch.tensor(tag_index, device=self.device, dtype=torch.long)
        number_of_children_tensor = torch.tensor(number_of_children, device=self.device, dtype=torch.long)
        invalid_mask = None
        if make_invalid_mask:
            invalid_mask = self.get_invalid_mask(state.tokens_lengths, state.token_counter, state.last_action, state.open_nt_count)
        next_state = State(
            state.tokens_tensor, state.tags_tensor,
            state.parent_node, state.stack_size, max_stack_size, state.shift_index,
            state.tokens_lengths, state.token_counter, state.last_action, state.open_nt_count,
            nt_index_tensor, compose_nt_index_tensor, number_of_children_tensor, token_index_tensor, tag_index_tensor, invalid_mask,
            nt_actions, shift_actions, gen_actions, reduce_actions,
        )
        return next_state

    def get_invalid_mask(self, tokens_lengths, token_counter, last_action, open_nt_count):
        batch_size = len(token_counter)
        mask = torch.ones((batch_size, self.action_count), device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            valid_actions = self.action_set.valid_actions(tokens_lengths[i], token_counter[i], last_action[i], open_nt_count[i])
            if ACTION_REDUCE_TYPE in valid_actions:
                mask[i, self.reduce_index] = 0
            if self.generative and ACTION_GENERATE_TYPE in valid_actions:
                mask[i, self.gen_indices] = 0
            if not self.generative and ACTION_SHIFT_TYPE in valid_actions:
                mask[i, self.shift_index] = 0
            if ACTION_NON_TERMINAL_TYPE in valid_actions:
                mask[i, self.nt_indices] = 0
        return mask.unsqueeze(dim=0)

class State:

    def __init__(self,
        tokens_tensor, tags_tensor,
        parent_node, stack_size, max_stack_size, shift_index,
        tokens_lengths, token_counter, last_action, open_nt_count,
        nt_index, compose_nt_index, number_of_children, token_index, tag_index, invalid_mask,
        nt_actions, shift_actions, gen_actions, reduce_actions,
    ):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type parent_node: list of app.models.parallel_rnng.state.Tree
        :type stack_size: list of int
        :type max_stack_size: int
        :type shift_index: list of int
        :type tokens_lengths: torch.Tensor
        :type token_counter: list of int
        :type last_action: list of app.data.actions.action.Action
        :type open_nt_count: list of int
        :type nt_index: torch.Tensor
        :type compose_nt_index: torch.Tensor
        :type number_of_children: torch.Tensor
        :type token_index: torch.Tensor
        :type tag_index: torch.Tensor
        :type invalid_mask: torch.Tensor
        :type nt_actions: list of int
        :type shift_actions: list of int
        :type gen_actions: list of int
        :type reduce_actions: list of int
        """
        self.tokens_tensor = tokens_tensor
        self.tags_tensor = tags_tensor
        self.parent_node = parent_node
        self.stack_size = stack_size
        self.max_stack_size = max_stack_size
        self.shift_index = shift_index
        self.tokens_lengths = tokens_lengths
        self.token_counter = token_counter
        self.last_action = last_action
        self.open_nt_count = open_nt_count
        self.nt_index = nt_index
        self.compose_nt_index = compose_nt_index
        self.number_of_children = number_of_children
        self.token_index = token_index
        self.tag_index = tag_index
        self.invalid_mask = invalid_mask
        self.nt_actions = nt_actions
        self.shift_actions = shift_actions
        self.gen_actions = gen_actions
        self.reduce_actions = reduce_actions
        if nt_actions is None:
            self.non_pad_actions = None
        else:
            non_pad_actions = []
            non_pad_actions.extend(nt_actions)
            non_pad_actions.extend(shift_actions)
            non_pad_actions.extend(gen_actions)
            non_pad_actions.extend(reduce_actions)
            non_pad_actions.sort()
            self.non_pad_actions = non_pad_actions

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
