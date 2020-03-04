from app.data.action_set import get_action_set
from app.models import get_model
from app.utils import get_training_config
import hydra
import os

def load_saved_model(device, generative, token_count, action_count, non_terminal_count, load_dir):
    """
    :type device: torch.device
    :type generative: bool
    :type token_count: int
    :type action_count: int
    :type non_terminal_count: int
    :type load_dir: str
    :rtype: app.models.model.Model
    """
    training_config = get_training_config(load_dir)
    action_set = get_action_set(training_config.type)
    model = get_model(device, generative, token_count, action_count, non_terminal_count, action_set, training_config.model)
    _load_model_params(model, load_dir)
    return model

def _load_model_params(model, load_dir):
    """
    :type model: app.models.model.Model
    :type load_dir: str
    """
    absolute_load_dir = hydra.utils.to_absolute_path(load_dir)
    model_params_path = os.path.join(absolute_load_dir, 'model.pt')
    model.load(model_params_path)
