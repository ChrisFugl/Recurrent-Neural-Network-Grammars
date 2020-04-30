from app.data.converters.action import ActionConverter
from app.data.converters.non_terminal import NonTerminalConverter
from app.data.converters.tag import TagConverter
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
        model, iterator, action_converter = _load_from_dir(device, data, iterator_config, config.load_dir)
        return AncestralSampler(device, model, iterator, action_converter, config.posterior_scaling, config.samples)
    elif config.type == 'greedy':
        from app.samplers.greedy import GreedySampler
        assert config.load_dir is not None, 'Greedy sampling requires a discriminative model.'
        model, iterator, action_converter = _load_from_dir(device, data, iterator_config, config.load_dir)
        return GreedySampler(device, model, iterator, action_converter, config.posterior_scaling)
    elif config.type == 'importance':
        from app.samplers.importance import ImportanceSampler
        assert config.load_dir_dis is not None, 'Importance sampling requires a discriminative model.'
        assert config.load_dir_gen is not None, 'Importance sampling requires a generative model.'
        discriminative = _load_from_dir(device, data, iterator_config, config.load_dir_dis)
        generative = _load_from_dir(device, data, iterator_config, config.load_dir_gen)
        return ImportanceSampler(device, config.posterior_scaling, config.samples, *discriminative, *generative)
    else:
        raise Exception(f'Unknown sampler: {config.type}')

def _load_from_dir(device, data, iterator_config, load_dir):
    training_config = get_training_config(load_dir)
    generative = is_generative(training_config.type)
    loader = get_loader(training_config.loader, name=f'loader_{training_config.name}')
    _, actions_train, tokens_train, unknownified_tokens_train, tags_train = loader.load_train()
    _, actions_eval, tokens_eval, unknownified_tokens_eval, tags_eval = _load_evaluation_data(loader, data)
    token_converter = TokenConverter(tokens_train, unknownified_tokens_train)
    tag_converter = TagConverter(tags_train)
    action_converter = ActionConverter(generative, actions_train)
    non_terminal_converter = NonTerminalConverter(actions_train)
    model = get_model(device, generative, action_converter, token_converter, tag_converter, non_terminal_converter, training_config.model)
    load_model_params(model, load_dir)
    iterator = get_iterator(device, action_converter, token_converter, tag_converter, unknownified_tokens_eval, unknownified_tokens_eval, actions_eval, tags_eval, iterator_config)
    return model, iterator, action_converter

def _load_evaluation_data(loader, data):
    if data == 'train':
        return loader.load_train()
    elif data == 'val':
        return loader.load_val()
    elif data == 'test':
        return loader.load_test()
    else:
        raise Exception(f'Unknown data: {data}')
