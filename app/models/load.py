from app.data.converters.action import ActionConverter
from app.data.converters.non_terminal import NonTerminalConverter
from app.data.converters.tag import TagConverter
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
    """
    training_config = get_training_config(load_dir)
    generative = is_generative(training_config.type)
    loader = get_loader(training_config.loader, name=f'loader_{training_config.name}')
    _, actions, tokens_train, unknownified_tokens, tags = loader.load_train()
    token_converter = TokenConverter(tokens_train, unknownified_tokens)
    tag_converter = TagConverter(tags)
    action_converter = ActionConverter(generative, actions)
    non_terminal_converter = NonTerminalConverter(actions)
    model = get_model(device, generative, action_converter, token_converter, tag_converter, non_terminal_converter, training_config.model)
    load_model_params(model, load_dir)
    return model, generative, action_converter, token_converter, tag_converter, non_terminal_converter

def load_model_params(model, load_dir):
    """
    :type model: app.models.model.Model
    :type load_dir: str
    """
    absolute_load_dir = hydra.utils.to_absolute_path(load_dir)
    model_params_path = os.path.join(absolute_load_dir, 'model.pt')
    model.load(model_params_path)
