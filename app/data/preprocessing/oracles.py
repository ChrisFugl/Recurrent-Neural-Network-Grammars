from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_GENERATE_TYPE, ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, UNKNOWN_IDENTIFIER
from app.data.preprocessing.non_terminals import get_non_terminal_identifier
from app.data.preprocessing.terminals import get_terminal_node, find_next_terminal_start_index, is_start_of_terminal_node
from app.data.preprocessing.unknowns import constant_unknownifier, fine_grained_unknownifier
from functools import partial

def brackets2oracle(known_non_terminals, known_terminals, generative, fine_grained_unknowns, unknown_non_terminals, brackets):
    """
    :type known_non_terminals: list of str
    :type known_terminals: list of str
    :type generative: bool
    :type fine_grained_unknowns: bool
    :type unknown_non_terminals: bool
    :type brackets: list of str
    """
    token_type = _get_token_type(generative)
    unknownifier = _get_unknownifier(known_terminals, fine_grained_unknowns)
    brackets_stripped = []
    actions = []
    tokens = []
    tokens_unknownified = []
    for line in brackets:
        line_stripped = line.strip()
        line_tags, line_tokens = _line2tokens(line_stripped)
        line_tokens_unknownified = list(map(unknownifier, zip(line_tags, line_tokens)))
        line_actions = _line2actions(token_type, line_tokens_unknownified, line_stripped)
        action2string = partial(_action2string, known_non_terminals, unknown_non_terminals)
        line_actions_strings = list(map(action2string, line_actions))
        brackets_stripped.append(line_stripped)
        actions.append(line_actions_strings)
        tokens.append(line_tokens)
        tokens_unknownified.append(line_tokens_unknownified)
    return brackets_stripped, actions, tokens, tokens_unknownified

def _get_token_type(generative):
    if generative:
        return ACTION_GENERATE_TYPE
    else:
        return ACTION_SHIFT_TYPE

def _line2tokens(line):
    """
    :type line: str
    :rtype: list of str, list of str
    """
    tags = []
    tokens = []
    line_index = find_next_terminal_start_index(line, 0)
    while line_index != -1:
        tag, token, terminal_end_index = get_terminal_node(line, line_index)
        line_index = find_next_terminal_start_index(line, terminal_end_index + 1)
        tags.append(tag)
        tokens.append(token)
    return tags, tokens

def _line2actions(token_type, unknownified_tokens, line):
    """
    :type token_type: str
    :type unknownified_tokens: list of str
    :type line: str
    :rtype: list of (int, str)
    """
    assert line[0] == '(', 'Tree must start with "(".'
    assert line[len(line) - 1] == ')', 'Tree must end with ")".'
    actions = []
    line_index = 0
    token_index = 0
    while line_index != -1:
        if line[line_index] == ')':
            action = ACTION_REDUCE_TYPE, None
            search_index = line_index + 1
        elif is_start_of_terminal_node(line, line_index):
            _, _, terminal_end_index = get_terminal_node(line, line_index)
            word = unknownified_tokens[token_index]
            action = token_type, word
            token_index += 1
            search_index = terminal_end_index + 1
        else:
            non_terminal = get_non_terminal_identifier(line, line_index)
            action = ACTION_NON_TERMINAL_TYPE, non_terminal
            search_index = line_index + 1
        line_index = _get_next_action_start_index(line, search_index)
        actions.append(action)
    return actions

def _action2string(known_non_terminals, unknown_non_terminals, action):
    type, argument = action
    if type == ACTION_REDUCE_TYPE:
        return 'REDUCE'
    elif type == ACTION_GENERATE_TYPE:
        return f'GEN({argument})'
    elif type == ACTION_SHIFT_TYPE:
        return 'SHIFT'
    elif type == ACTION_NON_TERMINAL_TYPE:
        if unknown_non_terminals and argument not in known_non_terminals:
            non_terminal = UNKNOWN_IDENTIFIER
        else:
            non_terminal = argument
        return f'NT({non_terminal})'
    else:
        raise Exception(f'Unknown action type: {type}')

def _get_next_action_start_index(line, line_index):
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

def _get_unknownifier(known_terminals, fine_grained_unknowns):
    if fine_grained_unknowns:
        unknownifier = fine_grained_unknownifier
    else:
        unknownifier = constant_unknownifier
    return partial(unknownifier, known_terminals)

def get_trees_from_oracle(oracle):
    """
    :type oracle: list of str
    :rtype: list of str
    """
    lines_count = len(oracle)
    assert lines_count % 4 == 0, 'Oracle files must have a multiple of four lines.'
    trees = oracle[0:lines_count:4]
    return trees

def get_actions_from_oracle(oracle):
    """
    :type oracle: list of str
    :rtype: list of list of str
    """
    return _get_from_oracle(oracle, 1)

def get_terms_from_oracle(oracle):
    """
    :type oracle: list of str
    :rtype: list of list of str
    """
    return _get_from_oracle(oracle, 2)

def get_unknownified_terms_from_oracle(oracle):
    """
    :type oracle: list of str
    :rtype: list of list of str
    """
    return _get_from_oracle(oracle, 3)

def get_tags_from_oracle(oracle):
    """
    :type oracle: list of str
    :rtype: list of list of str
    """
    lines_count = len(oracle)
    assert lines_count % 4 == 0, 'Oracle files must have a multiple of four lines.'
    tags = []
    for line_index in range(0, lines_count, 4):
        line_tags, _ = _line2tokens(oracle[line_index])
        tags.append(line_tags)
    return tags

def _get_from_oracle(oracle, start_index):
    lines_count = len(oracle)
    assert lines_count % 4 == 0, 'Oracle files must have a multiple of four lines.'
    sentences = oracle[start_index:lines_count:4]
    sentences_tokenized = list(map(lambda sentence: sentence.split(' '), sentences))
    return sentences_tokenized

def read_oracle(path):
    """
    :type path: str
    :rtype: list of str
    """
    with open(path, 'r') as input_file:
        oracle = input_file.read().split('\n')
    stripped = map(lambda line: line.strip(), oracle)
    non_empty = filter(lambda line: len(line) != 0, stripped)
    return list(non_empty)
