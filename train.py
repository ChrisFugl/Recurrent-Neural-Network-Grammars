from app.data.action_set import get_action_set
from app.checkpoints import get_checkpoint
from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.evaluators import get_evaluator
from app.losses import get_loss
from app.models import get_model
from app.optimizers import get_optimizer
from app.stopping_criteria import get_stopping_criterion
from app.tasks.train import TrainTask
import numpy as np
import hydra
import random
import torch

@hydra.main(config_path='configs/train.yaml')
def _main(config):
    _set_seed(config.seed)
    loader = get_loader(config.loader)
    _, actions_train, _, unknownified_tokens_train = loader.load_train()
    _, actions_val, _, unknownified_tokens_val = loader.load_val()
    is_generative = config.type == 'generative'
    device = _get_device(config.gpu)
    token_converter = TokenConverter(unknownified_tokens_train)
    action_converter = ActionConverter(token_converter, is_generative, actions_train)
    action_set = get_action_set(config.type)
    iterator_train = get_iterator(device, action_converter, token_converter, unknownified_tokens_train, actions_train, config.iterator)
    iterator_val = get_iterator(device, action_converter, token_converter, unknownified_tokens_val, actions_val, config.iterator)
    token_count = token_converter.count()
    action_count = action_converter.count()
    non_terminal_count = action_converter.count_non_terminals()
    model = get_model(device, token_count, action_count, non_terminal_count, action_set, config)
    loss = get_loss(device, config.loss)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    stopping_criterion = get_stopping_criterion(config.stopping_criterion)
    checkpoint = get_checkpoint(config.checkpoint)
    evaluator = get_evaluator(config.evaluator)
    task = TrainTask(
        device,
        iterator_train, iterator_val,
        model, loss, optimizer,
        stopping_criterion, checkpoint, evaluator,
        config.load_checkpoint, token_count, non_terminal_count, action_count,
    )
    task.run()

def _get_device(gpu):
    if torch.cuda.is_available() and gpu is not None:
        return torch.device(gpu)
    else:
        return torch.device('cpu')

def _set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

if __name__ == '__main__':
    _main()
