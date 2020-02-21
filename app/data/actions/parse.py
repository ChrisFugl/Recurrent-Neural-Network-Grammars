from app.constants import (
    ACTION_GENERATE_TYPE, ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE,
    PATTERN_GEN, PATTERN_SHIFT, PATTERN_REDUCE, PATTERN_NON_TERMINAL
)

def parse_action(action_string):
    """
    :type action_string: str
    :rtype: int, str
    """
    nt_match = PATTERN_NON_TERMINAL.match(action_string)
    if nt_match is not None:
        return ACTION_NON_TERMINAL_TYPE, nt_match.group(1)
    reduce_match = PATTERN_REDUCE.match(action_string)
    if reduce_match is not None:
        return ACTION_REDUCE_TYPE, None
    shift_match = PATTERN_SHIFT.match(action_string)
    if shift_match is not None:
        return ACTION_SHIFT_TYPE, None
    gen_match = PATTERN_GEN.match(action_string)
    if gen_match is not None:
        return ACTION_GENERATE_TYPE, gen_match.group(1)
    raise Exception(f'Unknown action: {action_string}')
