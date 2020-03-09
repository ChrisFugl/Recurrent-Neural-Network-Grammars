from app.data.action_set import get_action_set
from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.models import get_model
from app.models.load import load_model_params
from app.utils import get_training_config, is_generative

def get_sampler(device, data, iterator_config, config):
    """
    :type device: torch.device
    :type data: str
    :param data: enumerated value: train, val, test
    :type iterator_config: object
    :type config: object
    :rtype: app.samplers.sampler.Sampler
    """
    if config.type == 'ancestral':
        from app.samplers.ancestral import AncestralSampler
        assert config.load_dir is not None, 'Ancestral sampling requires a discriminative model.'
        model, iterator, action_converter, token_converter = _load_from_dir(device, data, iterator_config, config.load_dir)
        return AncestralSampler(device, model, iterator, action_converter, token_converter, config.posterior_scaling, config.samples)
    else:
        raise Exception(f'Unknown sampler: {config.type}')

def _load_from_dir(device, data, iterator_config, load_dir):
    training_config = get_training_config(load_dir)
    generative = is_generative(training_config.type)
    loader = get_loader(training_config.loader, name=training_config.name)
    _, actions_train, _, unknownified_tokens_train, _ = loader.load_train()
    _, actions_eval, _, unknownified_tokens_eval, tags_eval = _load_evaluation_data(loader, data)
    token_converter = TokenConverter(unknownified_tokens_train)
    action_converter = ActionConverter(token_converter, generative, actions_train)
    action_set = get_action_set(training_config.type)
    model = get_model(device, generative, token_converter, action_converter, action_set, training_config.model)
    load_model_params(model, load_dir)
    iterator = get_iterator(device, action_converter, token_converter, unknownified_tokens_eval, actions_eval, tags_eval, iterator_config)
    return model, iterator, action_converter, token_converter

def _load_evaluation_data(loader, data):
    if data == 'train':
        return loader.load_train()
    elif data == 'val':
        return loader.load_val()
    elif data == 'test':
        return loader.load_test()
    else:
        raise Exception(f'Unknown data: {data}')
