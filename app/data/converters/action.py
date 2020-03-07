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

    def get_singleton_offset(self):
        """
        :rtype: int
        """
        return ACTION_EMBEDDING_OFFSET

    def get_terminal_offset(self):
        """
        :rtype: int
        """
        return ACTION_EMBEDDING_OFFSET + self._singleton_count

    def get_non_terminal_offset(self):
        """
        :rtype: int
        """
        return ACTION_EMBEDDING_OFFSET + self._singleton_count + self._terminal_count

    def action2integer(self, action):
        """
        :type action: app.data.actions.action.Action
        :rtype: int
        """
        if hasattr(action, 'argument'):
            argument = action.argument
        else:
            argument = None
        return self._action_args2integer(action.type(), argument)

    def integer2action(self, device, index):
        """
        :type device: torch.device
        :type integer: int
        :rtype: app.data.actions.action.Action
        """
        string = self.integer2string(index)
        action = self.string2action(device, string)
        return action

    def integer2string(self, index):
        """
        :type integer: int
        :rtype: str
        """
        if index == PAD_INDEX:
            return PAD_SYMBOL
        if index == START_ACTION_INDEX:
            return START_ACTION_SYMBOL
        singleton_offset = self.get_singleton_offset()
        if index < singleton_offset + self._singleton_count:
            return self._index2singleton[index - singleton_offset]
        terminal_offset = self.get_terminal_offset()
        if index < terminal_offset + self._terminal_count:
            terminal = self._index2terminal[index - terminal_offset]
            return f'GEN({terminal})'
        non_terminal_offset = self.get_non_terminal_offset()
        non_terminal = self._index2non_terminal[index - non_terminal_offset]
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
        return self._action_args2integer(type, argument)

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

    def _action_args2integer(self, type, argument):
        if self._generative:
            if type == ACTION_REDUCE_TYPE:
                singleton_offset = self.get_singleton_offset()
                return singleton_offset + self._singleton2index['REDUCE']
            elif type == ACTION_GENERATE_TYPE:
                terminal_offset = self.get_terminal_offset()
                return terminal_offset + self._terminal2index[argument]
            else:
                non_terminal_offset = self.get_non_terminal_offset()
                return non_terminal_offset + self._non_terminal2index[argument]
        else:
            if type == ACTION_REDUCE_TYPE:
                singleton_offset = self.get_singleton_offset()
                return singleton_offset + self._singleton2index['REDUCE']
            elif type == ACTION_SHIFT_TYPE:
                singleton_offset = self.get_singleton_offset()
                return singleton_offset + self._singleton2index['SHIFT']
            else:
                non_terminal_offset = self.get_non_terminal_offset()
                return non_terminal_offset + self._non_terminal2index[argument]
