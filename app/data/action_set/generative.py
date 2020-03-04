from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_GENERATE_INDEX, ACTION_NON_TERMINAL_INDEX,
    ACTION_NON_TERMINAL_TYPE,
    MAX_OPEN_NON_TERMINALS
)
from app.data.action_set.action_set import ActionSet

class Generative(ActionSet):

    def valid_actions(self, token_buffer, token_counter, stack, open_non_terminals_count):
        """
        :type token_buffer: app.memories.memory.Memory
        :type token_counter: int
        :type stack: app.stacks.stack.Stack
        :type open_non_terminals_count: int
        :rtype: list of int, dict
        """
        valid_actions = []
        action2index = {}
        counter = 0
        _, top_action = stack.top()
        top_action_type = top_action.type()
        is_non_terminal = top_action_type == ACTION_NON_TERMINAL_TYPE
        if (not is_non_terminal or (is_non_terminal and not top_action.open)) and open_non_terminals_count >= 1:
            valid_actions.append(ACTION_REDUCE_INDEX)
            action2index[ACTION_REDUCE_INDEX] = counter
            counter += 1
        if open_non_terminals_count >= 1:
            valid_actions.append(ACTION_GENERATE_INDEX)
            action2index[ACTION_GENERATE_INDEX] = counter
            counter += 1
        if open_non_terminals_count <= MAX_OPEN_NON_TERMINALS:
            valid_actions.append(ACTION_NON_TERMINAL_INDEX)
            action2index[ACTION_NON_TERMINAL_INDEX] = counter
        return valid_actions, action2index
