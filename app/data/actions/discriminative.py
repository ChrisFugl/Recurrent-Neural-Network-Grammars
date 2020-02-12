from app.data.actions.action import Action
from app.data.actions.action_set import ActionSet
from app.data.preprocessing.non_terminals import get_non_terminal_identifier
from app.data.preprocessing.terminals import get_terminal_node, is_start_of_terminal_node
import re

class Discriminative(ActionSet):

    REDUCE = 0
    SHIFT = 1
    NT = 2

    _nt_pattern = re.compile(r'^NT\((\S+)\)$')
    _reduce_pattern = re.compile(r'^REDUCE$')
    _shift_pattern = re.compile(r'^SHIFT$')

    def action2string(self, action):
        """
        :type action: app.data.actions.action.Action
        :rtype: str
        """
        if action.type == Discriminative.NT:
            return f'NT({action.argument})'
        elif action.type == Discriminative.REDUCE:
            return 'REDUCE'
        elif action.type == Discriminative.SHIFT:
            return f'SHIFT'
        raise Exception(f'Unknown action: {action.type}')

    def string2action(self, value):
        """
        :type value: str
        :rtype: app.data.actions.action.Action
        """
        nt_match = self._nt_pattern.match(value)
        if nt_match is not None:
            return Action(Discriminative.NT, nt_match.group(1))
        reduce_match = self._reduce_pattern.match(value)
        if reduce_match is not None:
            return Action(Discriminative.REDUCE, None)
        shift_match = self._shift_pattern.match(value)
        if shift_match is not None:
            return Action(Discriminative.SHIFT, None)
        raise Exception(f'Unknown action: {value}')

    def line2actions(self, unknownified_tokens, line):
        """
        :type unknownified_tokens: list of str
        :type line: str
        :rtype: list of app.data.actions.action.Action
        """
        assert line[0] == '(', 'Tree must start with "(".'
        assert line[len(line) - 1] == ')', 'Tree must end with ")".'
        actions = []
        line_index = 0
        while line_index != -1:
            if line[line_index] == ')':
                action = Action(Discriminative.REDUCE, None)
                search_index = line_index + 1
            elif is_start_of_terminal_node(line, line_index):
                _, word, terminal_end_index = get_terminal_node(line, line_index)
                action = Action(Discriminative.SHIFT, None)
                search_index = terminal_end_index + 1
            else:
                non_terminal = get_non_terminal_identifier(line, line_index)
                action = Action(Discriminative.NT, non_terminal)
                search_index = line_index + 1
            line_index = self.get_next_action_start_index(line, search_index)
            actions.append(action)
        return actions
