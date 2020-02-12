from app.data.preprocessing.terminals import get_terminal_node, find_next_terminal_start_index

class ActionSet:

    def action2string(self, action):
        """
        :type action: app.data.actions.action.Action
        :rtype: str
        """
        raise NotImplementedError('must be implemented by subclass')

    def string2action(self, value):
        """
        :type value: str
        :rtype: app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def line2actions(self, unknownified_tokens, line):
        """
        :type unknownified_tokens: list of str
        :type line: str
        :rtype: list of app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def line2tokens(self, line):
        """
        :type line: str
        :rtype: list of str
        """
        tokens = []
        line_index = find_next_terminal_start_index(line, 0)
        while line_index != -1:
            _, token, terminal_end_index = get_terminal_node(line, line_index)
            line_index = find_next_terminal_start_index(line, terminal_end_index + 1)
            tokens.append(token)
        return tokens

    def get_next_action_start_index(self, line, line_index):
        """
        :type line: str
        :type line_index: int
        :rtype: str
        """
        next_open_bracket_index = line.find('(', line_index)
        next_close_bracket_index = line.find(')', line_index)
        if next_open_bracket_index == -1 or next_close_bracket_index == -1:
            return max(next_open_bracket_index, next_close_bracket_index)
        else:
            return min(next_open_bracket_index, next_close_bracket_index)
