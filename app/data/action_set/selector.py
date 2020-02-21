from app.data.action_set.discriminative import Discriminative
from app.data.action_set.generative import Generative

def get_action_set(model_type):
    """
    :type model_type: str
    :rtype: app.data.action_set.action_set.ActionSet
    """
    if model_type == 'generative':
        return Generative()
    elif model_type == 'discriminative':
        return Discriminative()
    else:
        raise Exception(f'Unknown model type: {model_type}')
