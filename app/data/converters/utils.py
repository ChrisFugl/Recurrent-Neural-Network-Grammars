def action_string_dis2gen(gen_action_converter, tokens, action_strings):
    """
    :type gen_action_converter: app.data.converters.action.ActionConverter
    :type tokens: list of str
    :type action_strings: list of str
    :rtype: list of app.data.actions.action.Action
    """
    shift_index = 0
    actions = []
    for action_string in action_strings:
        if action_string == 'SHIFT':
            action = gen_action_converter.get_cached_gen_action(tokens[shift_index])
            shift_index += 1
        else:
            action = gen_action_converter.string2action(action_string)
        actions.append(action)
    return actions
