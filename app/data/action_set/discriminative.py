from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_SHIFT_INDEX, ACTION_NON_TERMINAL_INDEX,
    ACTION_NON_TERMINAL_TYPE,
    MAX_OPEN_NON_TERMINALS
)
from app.data.action_set.action_set import ActionSet

class Discriminative(ActionSet):

    def valid_actions(self, tokens_length, token_counter, last_action, open_non_terminals_count):
        """
        :type tokens_length: int
        :type token_counter: int
        :type last_action: app.actions.action.Action
        :type open_non_terminals_count: int
        :rtype: list of int, dict
        """
        valid_actions = []
        action2index = {}
        if last_action is None:
            # last action was the start-action, only non-terminals should be allowed
            valid_actions.append(ACTION_NON_TERMINAL_INDEX)
            action2index[ACTION_NON_TERMINAL_INDEX] = 0
        else:
            counter = 0
            buffer_is_empty = tokens_length <= token_counter
            last_action_type = last_action.type()
            is_non_terminal = last_action_type == ACTION_NON_TERMINAL_TYPE
            if (not is_non_terminal or (is_non_terminal and not last_action.open)) and (buffer_is_empty or open_non_terminals_count >= 2):
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
