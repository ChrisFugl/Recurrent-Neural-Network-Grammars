from app.data.actions.discriminative import Discriminative
from app.data.actions.generative import Generative
from app.data.preprocessing.unknowns import constant_unknownifier, fine_grained_unknownifier
from functools import partial

def brackets2oracle(brackets, known_terminals, generative, fine_grained_unknowns):
    """
    :type brackets: list of str
    :type known_terminals: list of str
    :type generative: bool
    :type fine_grained_unknowns: bool
    """
    action_set = _get_action_set(generative)
    unknownifier = _get_unknownifier(known_terminals, fine_grained_unknowns)
    brackets_stripped = []
    actions = []
    tokens = []
    tokens_unknownified = []
    for line in brackets:
        line_stripped = line.strip()
        line_tokens = action_set.line2tokens(line_stripped)
        line_tokens_unknownified = list(map(unknownifier, line_tokens))
        line_actions = action_set.line2actions(line_tokens_unknownified, line_stripped)
        line_actions = list(map(action_set.action2string, line_actions))
        brackets_stripped.append(line_stripped)
        actions.append(line_actions)
        tokens.append(line_tokens)
        tokens_unknownified.append(line_tokens_unknownified)
    return brackets_stripped, actions, tokens, tokens_unknownified

def _get_action_set(generative):
    if generative:
        return Generative()
    else:
        return Discriminative()

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
        return input_file.read().split('\n')
