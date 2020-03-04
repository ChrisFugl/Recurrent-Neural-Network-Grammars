from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.inferers import get_inferer
from app.models.load import load_saved_model
from app.samplers import get_sampler
from app.tasks.evaluate import EvaluateTask
from app.utils import get_device, get_training_config, is_generative, set_seed
import hydra

@hydra.main(config_path='configs/evaluate.yaml')
def _main(config):
    assert config.load_dir is not None, 'load_dir must be specified.'

    set_seed(config.seed)
    device = get_device(config.gpu)

    # load training data in order to count tokens, actions, and non-terminals
    training_config = get_training_config(config.load_dir)
    generative = is_generative(training_config.type)
    loader_training = get_loader(training_config.loader)
    _, actions_evaluate, _, unknownified_tokens_evaluate = loader_training.load_train()
    token_converter = TokenConverter(unknownified_tokens_evaluate)
    action_converter = ActionConverter(token_converter, generative, actions_evaluate)
    token_count = token_converter.count()
    action_count = action_converter.count()
    non_terminal_count = action_converter.count_non_terminals()

    # load evaluation data
    loader_evaluate = get_loader(config.loader)
    _, actions_evaluate, _, unknownified_tokens_evaluate = _load_evaluation_data(loader_evaluate, config.data)
    iterator = get_iterator(device, action_converter, token_converter, unknownified_tokens_evaluate, actions_evaluate, config.iterator)

    model = load_saved_model(device, token_count, action_count, non_terminal_count, config.load_dir)
    sampler = get_sampler(device, token_count, action_count, non_terminal_count, config)
    inferer = get_inferer(device, model, sampler, config.inferer)
    task = EvaluateTask(device, inferer, sampler, iterator)
    task.run()

def _load_evaluation_data(loader, data):
    if data == 'train':
        return loader.load_train()
    elif data == 'val':
        return loader.load_val()
    elif data == 'test':
        return loader.load_test()
    else:
        raise Exception(f'Unknown data: {data}')

if __name__ == '__main__':
    _main()
