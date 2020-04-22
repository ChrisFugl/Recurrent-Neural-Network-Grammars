from app.constants import ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE

class RNNGState:

    def __init__(self, stack_top, action_state, buffer_state, tokens, tags, tokens_length, open_non_terminals_count, token_counter, last_action=None, parent_node=None):
        """
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_state: object
        :type buffer_state: app.models.rnng.buffer.BufferState
        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :type tokens_length: int
        :type open_non_terminals_count: int
        :type token_counter: int
        :type last_action: app.data.actions.action.Action
        :type parent_node: Tree
        """
        self.stack_top = stack_top
        self.action_state = action_state
        self.buffer_state = buffer_state
        self.tokens = tokens
        self.tags = tags
        self.tokens_length = tokens_length
        self.open_non_terminals_count = open_non_terminals_count
        self.token_counter = token_counter
        self.last_action = last_action
        self.parent_node = parent_node

    def next(self, action_converter, action, stack_top, action_state, buffer_state, open_non_terminals_count, token_counter):
        """
        :type action_converter: app.data.converters.action.ActionConverter
        :type action: app.data.actions.action.Action
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_state: object
        :type buffer_state: app.models.rnng.buffer.BufferState
        :type open_non_terminals_count: int
        :type token_counter: int
        :rtype: RNNGState
        """
        parent_node = self.parent_node
        node = Tree(action, parent=parent_node)
        type = action.type()
        last_action = action
        if type == ACTION_SHIFT_TYPE:
            parent_node.add_child(node)
        elif type == ACTION_GENERATE_TYPE:
            parent_node.add_child(node)
        elif type == ACTION_NON_TERMINAL_TYPE:
            if parent_node is not None:
                parent_node.add_child(node)
            parent_node = node
        else: # reduce
            last_action = action_converter.get_cached_nt_action(parent_node.action.argument, False)
            parent_node = parent_node.parent
        return RNNGState(
            stack_top,
            action_state,
            buffer_state,
            self.tokens,
            self.tags,
            self.tokens_length,
            open_non_terminals_count,
            token_counter,
            parent_node=parent_node,
            last_action=last_action,
        )

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
