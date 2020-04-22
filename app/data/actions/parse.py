from app.constants import ACTION_GENERATE_TYPE, ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE

def parse_action(action_string):
    """
    :type action_string: str
    :rtype: int, str
    """
    first_character = action_string[0]
    if first_character == 'N':
        argument = action_string[3:-1]
        return ACTION_NON_TERMINAL_TYPE, argument
    elif first_character == 'S':
        return ACTION_SHIFT_TYPE, None
    elif first_character == 'G':
        argument = action_string[4:-1]
        return ACTION_GENERATE_TYPE, argument
    elif first_character == 'R':
        return ACTION_REDUCE_TYPE, None
    else:
        raise Exception(f'Unknown action: {action_string}')
