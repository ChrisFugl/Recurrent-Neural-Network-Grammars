from app.data.action_set import get_action_set
from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.data.loaders import get_loader
from app.models import get_model
from app.utils import get_training_config, is_generative
import hydra
import os

def load_saved_model(device, load_dir):
    """
    :type device: torch.device
    :type load_dir: str
    :rtype: app.models.model.Model, app.data.converters.action.ActionConverter
    """
    training_config = get_training_config(load_dir)
    generative = is_generative(training_config.type)
    action_set = get_action_set(training_config.type)
    loader = get_loader(training_config.loader, name=f'loader_{training_config.name}')
    _, actions, _, unknownified_tokens = loader.load()
    token_converter = TokenConverter(unknownified_tokens)
    action_converter = ActionConverter(token_converter, generative, actions)
    model = get_model(device, generative, token_converter, action_converter, action_set, training_config.model)
    _load_model_params(model, load_dir)
    return model, action_converter

def _load_model_params(model, load_dir):
    """
    :type model: app.models.model.Model
    :type load_dir: str
    """
    absolute_load_dir = hydra.utils.to_absolute_path(load_dir)
    model_params_path = os.path.join(absolute_load_dir, 'model.pt')
    model.load(model_params_path)
