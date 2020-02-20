from app.constants import PAD_INDEX, PAD_SYMBOL
from app.data.actions.discriminative import Discriminative
from app.data.actions.generative import Generative

class ActionConverter:
    """
    Converts actions to integers and back again in a deterministic manner.

    Actions come in the following order:
    * singletons (REDUCE, SHIFT)
    * terminals (GEN)
    * non-terminals (NT)
    """

    def __init__(self, token_converter, trees):
        """
        :type token_converter: app.data.converters.token.TokenConverter
        :type trees: list of list of str
        """
        self._generative = self._is_generative(trees)
        self._action_set = self._get_action_set()
        self._singleton2index, self._index2singleton = self._get_singleton_actions(self._generative)
        self._singleton_count = len(self._index2singleton)
        self._terminal2index, self._index2terminal = self._get_terminal_actions(self._generative, token_converter)
        self._terminal_count = len(self._index2terminal)
        self._non_terminal2index, self._index2non_terminal = self._get_non_terminal_actions(self._generative, self._action_set, trees)
        self._non_terminal_count = len(self._index2non_terminal)
        self._actions_count = 1 + self._singleton_count + self._terminal_count + self._non_terminal_count

    def count(self):
        """
        Count number of actions.

        :rtype: int
        """
        return self._actions_count

    def integer2action(self, index):
        """
        :type integer: int
        :rtype: str
        """
        if index == PAD_INDEX:
            return PAD_SYMBOL
        elif index < 1 + self._singleton_count:
            return self._index2singleton[index]
        elif index < 1 + self._singleton_count + self._terminal_count:
            terminal = self._index2terminal[index - self._singleton_count]
            return f'GEN({terminal})'
        else:
            non_terminal = self._index2non_terminal[index - self._singleton_count - self._terminal_count]
            return f'NT({non_terminal})'

    def action2integer(self, action_string):
        """
        :type action: str
        :rtype: int
        """
        if action_string == PAD_SYMBOL:
            return PAD_INDEX
        action = self._action_set.string2action(action_string)
        if self._generative:
            if action.type == Generative.REDUCE:
                return self._singleton2index['REDUCE']
            elif action.type == Generative.GEN:
                return self._singleton_count + self._terminal2index[action.argument]
            else:
                return self._singleton_count + self._terminal_count + self._non_terminal2index[action.argument]
        else:
            if action.type == Generative.REDUCE:
                return self._singleton2index['REDUCE']
            elif action.type == Generative.SHIFT:
                return self._singleton2index['SHIFT']
            else:
                return self._singleton_count + self._non_terminal2index[action.argument]

    def get_action_set(self):
        """
        :rtype: app.data.actions.action_set.ActionSet
        """
        return self._action_set

    def is_discriminative(self):
        """
        :rtype: bool
        """
        return not self._generative

    def is_generative(self):
        """
        :rtype: bool
        """
        return self._generative

    def _is_generative(self, trees):
        """
        :type trees: list of list of str
        """
        first_tree = trees[0]
        return not 'SHIFT' in first_tree

    def _get_action_set(self):
        if self._generative:
            return Generative()
        else:
            return Discriminative()

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

    def _get_non_terminal_actions(self, generative, action_set, trees):
        index2action = []
        action2index = {}
        counter = 0
        for tree in trees:
            for action_string in tree:
                action = action_set.string2action(action_string)
                if action.type == action_set.NT:
                    non_terminal = action.argument
                    if not non_terminal in action2index:
                        index2action.append(non_terminal)
                        action2index[non_terminal] = counter
                        counter += 1
        return action2index, index2action
