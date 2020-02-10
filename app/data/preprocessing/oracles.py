from app.data.actions import Discriminative, Generative
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
        line_actions = action_set.line2actions(line_stripped)
        line_actions = list(map(action_set.action2string, line_actions))
        line_tokens = action_set.line2tokens(line_stripped)
        line_tokens_unknownified = list(map(unknownifier, line_tokens))
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
