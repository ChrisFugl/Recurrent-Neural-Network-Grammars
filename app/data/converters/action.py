from app.constants import (
    ACTION_REDUCE_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE,
    ACTION_EMBEDDING_OFFSET, PAD_INDEX, PAD_SYMBOL, START_ACTION_INDEX, START_ACTION_SYMBOL
)
from app.data.actions.generate import GenerateAction
from app.data.actions.non_terminal import NonTerminalAction
from app.data.actions.reduce import ReduceAction
from app.data.actions.shift import ShiftAction
from app.data.actions.parse import parse_action

class ActionConverter:
    """
    Converts actions to integers and back again in a deterministic manner.

    Actions come in the following order:
    * padding
    * singletons (REDUCE, SHIFT)
    * terminals (GEN)
    * non-terminals (NT)
    """

    def __init__(self, token_converter, is_generative, trees):
        """
        :type token_converter: app.data.converters.token.TokenConverter
        :type is_generative: bool
        :type trees: list of list of str
        """
        self._generative = is_generative
        self._singleton2index, self._index2singleton = self._get_singleton_actions(self._generative)
        self._singleton_count = len(self._index2singleton)
        self._terminal2index, self._index2terminal = self._get_terminal_actions(self._generative, token_converter)
        self._terminal_count = len(self._index2terminal)
        self._non_terminal2index, self._index2non_terminal = self._get_non_terminal_actions(self._generative, trees)
        self._non_terminal_count = len(self._index2non_terminal)
        self._actions_count = ACTION_EMBEDDING_OFFSET + self._singleton_count + self._terminal_count + self._non_terminal_count

    def count(self):
        """
        Count number of actions.

        :rtype: int
        """
        return self._actions_count

    def count_singleton_actions(self):
        """
        :rtype: int
        """
        return self._singleton_count

    def count_non_terminals(self):
        """
        :rtype: int
        """
        return self._non_terminal_count

    def count_terminals(self):
        """
        :rtype: int
        """
        return self._terminal_count

    def integer2action(self, index):
        """
        :type integer: int
        :rtype: app.data.actions.action.Action
        """
        string = self.integer2string(index)
        action = self.string2action(string)
        return action

    def integer2string(self, index):
        """
        :type integer: int
        :rtype: str
        """
        if index == PAD_INDEX:
            return PAD_SYMBOL
        elif index == START_ACTION_INDEX:
            return START_ACTION_SYMBOL
        elif index < ACTION_EMBEDDING_OFFSET + self._singleton_count:
            return self._index2singleton[index]
        elif index < ACTION_EMBEDDING_OFFSET + self._singleton_count + self._terminal_count:
            terminal = self._index2terminal[index - self._singleton_count]
            return f'GEN({terminal})'
        else:
            non_terminal = self._index2non_terminal[index - self._singleton_count - self._terminal_count]
            return f'NT({non_terminal})'

    def string2action(self, device, action_string):
        """
        :type device: torch.device
        :type action_string: str
        :rtype: app.actions.action.Action
        """
        if action_string == PAD_SYMBOL:
            raise Exception('Cannot convert padding symbol to action.')
        if action_string == START_ACTION_SYMBOL:
            raise Exception('Cannot convert start symbol to action.')
        type, argument = parse_action(action_string)
        if self._generative:
            if type == ACTION_REDUCE_TYPE:
                return ReduceAction(device)
            elif type == ACTION_GENERATE_TYPE:
                argument_index = self._terminal2index[argument]
                return GenerateAction(device, argument, argument_index)
            else:
                argument_index = self._non_terminal2index[argument]
                return NonTerminalAction(device, argument, argument_index)
        else:
            if type == ACTION_REDUCE_TYPE:
                return ReduceAction(device)
            elif type == ACTION_SHIFT_TYPE:
                return ShiftAction(device)
            else:
                argument_index = self._non_terminal2index[argument]
                return NonTerminalAction(device, argument, argument_index)

    def string2integer(self, action_string):
        """
        :type action_string: str
        :rtype: int
        """
        if action_string == PAD_SYMBOL:
            return PAD_INDEX
        if action_string == START_ACTION_SYMBOL:
            return START_ACTION_INDEX
        type, argument = parse_action(action_string)
        if self._generative:
            if type == ACTION_REDUCE_TYPE:
                return self._singleton2index['REDUCE']
            elif type == ACTION_GENERATE_TYPE:
                return self._singleton_count + self._terminal2index[argument]
            else:
                return self._singleton_count + self._terminal_count + self._non_terminal2index[argument]
        else:
            if type == ACTION_REDUCE_TYPE:
                return self._singleton2index['REDUCE']
            elif type == ACTION_SHIFT_TYPE:
                return self._singleton2index['SHIFT']
            else:
                return self._singleton_count + self._non_terminal2index[argument]

    def _get_singleton_actions(self, generative):
        if generative:
            index2action = ['REDUCE']
            action2index = {'REDUCE': 0}
        else:
            index2action = ['REDUCE', 'SHIFT']
            action2index = {'REDUCE': 0, 'SHIFT': 1}
        return action2index, index2action

    def _get_terminal_actions(self, generative, token_converter):
        index2action = []
        action2index = {}
        if generative:
            tokens = token_converter.tokens()
            for index, token in enumerate(tokens):
                index2action.append(token)
                action2index[token] = index
        return action2index, index2action

    def _get_non_terminal_actions(self, generative, trees):
        index2action = []
        action2index = {}
        counter = 0
        for tree in trees:
            for action_string in tree:
                action_type, action_argument = parse_action(action_string)
                if action_type == ACTION_NON_TERMINAL_TYPE:
                    non_terminal = action_argument
                    if not non_terminal in action2index:
                        index2action.append(non_terminal)
                        action2index[non_terminal] = counter
                        counter += 1
        return action2index, index2action
