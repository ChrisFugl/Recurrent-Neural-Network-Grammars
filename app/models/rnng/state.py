from app.constants import ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.data.actions.non_terminal import NonTerminalAction

class RNNGState:

    def __init__(self, stack_top, action_top, token_top, tokens, tags, tokens_length, open_non_terminals_count, token_counter, last_action=None, parent_node=None):
        """
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_top: app.models.rnng.stack.StackNode
        :type token_top: app.models.rnng.stack.StackNode
        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :type tokens_length: int
        :type open_non_terminals_count: int
        :type token_counter: int
        :type last_action: app.data.actions.action.Action
        :type parent_node: Tree
        """
        self.stack_top = stack_top
        self.action_top = action_top
        self.token_top = token_top
        self.tokens = tokens
        self.tags = tags
        self.tokens_length = tokens_length
        self.open_non_terminals_count = open_non_terminals_count
        self.token_counter = token_counter
        self.last_action = last_action
        self.parent_node = parent_node

    def next(self, action, stack_top, action_top, token_top, open_non_terminals_count, token_counter):
        """
        :type action: app.data.actions.action.Action
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_top: app.models.rnng.stack.StackNode
        :type token_top: app.models.rnng.stack.StackNode
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
            last_action = NonTerminalAction(parent_node.action.argument, open=False)
            parent_node = parent_node.parent
        return RNNGState(
            stack_top,
            action_top,
            token_top,
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
