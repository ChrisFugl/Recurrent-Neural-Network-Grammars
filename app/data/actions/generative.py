from app.data.actions.action import Action
from app.data.actions.action_set import ActionSet
from app.data.preprocessing.non_terminals import get_non_terminal_identifier
from app.data.preprocessing.terminals import get_terminal_node, is_start_of_terminal_node
import re

class Generative(ActionSet):

    NT = 0
    REDUCE = 1
    GEN = 2

    _nt_pattern = re.compile(r'^NT\((\S+)\)$')
    _reduce_pattern = re.compile(r'^REDUCE$')
    _gen_pattern = re.compile(r'^GEN\((\S+)\)$')

    def action2string(self, action):
        """
        :type action: app.data.actions.action.Action
        :rtype: str
        """
        if action.type == Generative.NT:
            return f'NT({action.argument})'
        elif action.type == Generative.REDUCE:
            return 'REDUCE'
        elif action.type == Generative.GEN:
            return f'GEN({action.argument})'
        raise Exception(f'Unknown action: {action.type}')

    def string2action(self, value):
        """
        :type value: str
        :rtype: app.data.actions.action.Action
        """
        nt_match = self._nt_pattern.match(value)
        if nt_match is not None:
            return Action(Generative.NT, nt_match.group(1))
        reduce_match = self._reduce_pattern.match(value)
        if reduce_match is not None:
            return Action(Generative.REDUCE, None)
        gen_match = self._gen_pattern.match(value)
        if gen_match is not None:
            return Action(Generative.GEN, gen_match.group(1))
        raise Exception(f'Unknown action: {value}')

    def line2actions(self, line):
        """
        :type line: str
        :rtype: list of app.data.actions.action.Action
        """
        assert line[0] == '(', 'Tree must start with "(".'
        assert line[len(line) - 1] == ')', 'Tree must end with ")".'
        actions = []
        line_index = 0
        while line_index != -1:
            if line[line_index] == ')':
                action = Action(Generative.REDUCE, None)
                search_index = line_index + 1
            elif is_start_of_terminal_node(line, line_index):
                _, word, terminal_end_index = get_terminal_node(line, line_index)
                action = Action(Generative.GEN, word)
                search_index = terminal_end_index + 1
            else:
                non_terminal = get_non_terminal_identifier(line, line_index)
                action = Action(Generative.NT, non_terminal)
                search_index = line_index + 1
            line_index = self.get_next_action_start_index(line, search_index)
            actions.append(action)
        return actions
