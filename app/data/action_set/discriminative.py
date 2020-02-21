from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_SHIFT_INDEX, ACTION_NON_TERMINAL_INDEX,
    ACTION_NON_TERMINAL_TYPE,
    MAX_OPEN_NON_TERMINALS
)
from app.data.action_set.action_set import ActionSet

class Discriminative(ActionSet):

    def valid_actions(self, token_buffer, token_index, stack, open_non_terminals_count):
        """
        :type token_buffer: app.memories.memory.Memory
        :type token_index: int
        :type stack: app.stacks.stack.Stack
        :type open_non_terminals_count: int
        :rtype: list of int, dict
        """
        valid_actions = []
        action2index = {}
        counter = 0
        buffer_is_empty = token_index == token_buffer.count()
        _, top_action = stack.top()
        top_action_type = top_action.type()
        is_non_terminal = top_action_type == ACTION_NON_TERMINAL_TYPE
        if buffer_is_empty or open_non_terminals_count >= 2 or (not is_non_terminal or (is_non_terminal and not top_action.open)):
            valid_actions.append(ACTION_REDUCE_INDEX)
            action2index[ACTION_REDUCE_INDEX] = counter
            counter += 1
        if not buffer_is_empty and open_non_terminals_count >= 1:
            valid_actions.append(ACTION_SHIFT_INDEX)
            action2index[ACTION_SHIFT_INDEX] = counter
            counter += 1
        if not buffer_is_empty and open_non_terminals_count <= MAX_OPEN_NON_TERMINALS:
            valid_actions.append(ACTION_NON_TERMINAL_INDEX)
            action2index[ACTION_NON_TERMINAL_INDEX] = counter
        return valid_actions, action2index
