def get_action_set(model_type):
    """
    :type model_type: str
    :rtype: app.data.action_sets.action_set.ActionSet
    """
    if model_type == 'generative':
        from app.data.action_sets.generative import GenerativeActionSet
        return GenerativeActionSet()
    elif model_type == 'discriminative':
        from app.data.action_sets.discriminative import DiscriminativeActionSet
        return DiscriminativeActionSet()
    else:
        raise Exception(f'Unknown model type: {model_type}')
