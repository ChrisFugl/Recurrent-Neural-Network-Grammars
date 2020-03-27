from app.constants import ACTION_NON_TERMINAL_TYPE, PAD_INDEX, PAD_SYMBOL, NON_TERMINAL_EMBEDDING_OFFSET
from app.data.actions.parse import parse_action

class NonTerminalConverter:

    def __init__(self, trees):
        """
        :type trees: list of list of str
        """
        self._non_terminal2integer, self._integer2non_terminal = self._get_non_terminal_converters(trees)

    def _get_non_terminal_converters(self, trees):
        non_terminal2integer = {PAD_SYMBOL: PAD_INDEX}
        integer2non_terminal = [PAD_SYMBOL]
        counter = NON_TERMINAL_EMBEDDING_OFFSET
        for tree in trees:
            for action_string in tree:
                action_type, action_argument = parse_action(action_string)
                if action_type == ACTION_NON_TERMINAL_TYPE:
                    non_terminal = action_argument
                    if not non_terminal in non_terminal2integer:
                        integer2non_terminal.append(non_terminal)
                        non_terminal2integer[non_terminal] = counter
                        counter += 1
        return non_terminal2integer, integer2non_terminal

    def count(self):
        """
        Count number of unique non_terminals.

        :rtype: int
        """
        return len(self._integer2non_terminal)

    def integer2non_terminal(self, integer):
        """
        :type integer: int
        :rtype: str
        """
        return self._integer2non_terminal[integer]

    def non_terminal2integer(self, non_terminal):
        """
        :type non_terminal: str
        :rtype: int
        """
        return self._non_terminal2integer[non_terminal]

    def non_terminals(self):
        """
        :rtype: list of str
        """
        return self._integer2non_terminal
