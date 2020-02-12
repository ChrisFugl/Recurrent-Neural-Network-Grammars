from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.models import get_model
from app.optimizers import get_optimizer
from app.tasks.train import TrainTask
import hydra
import torch

@hydra.main(config_path='configs/train.yaml')
def _main(config):
    loader = get_loader(config.loader)
    _, actions_train, _, unknownified_tokens_train = loader.load_train()
    _, actions_val, _, unknownified_tokens_val = loader.load_val()
    token_converter = TokenConverter(unknownified_tokens_train)
    action_converter = ActionConverter(token_converter, actions_train)
    device = _get_device()
    iterator_train = get_iterator(config.iterator, unknownified_tokens_train, actions_train, token_converter, action_converter, device)
    iterator_val = get_iterator(config.iterator, unknownified_tokens_val, actions_val, token_converter, action_converter, device)
    model = get_model(config.model, device)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    task = TrainTask(iterator_train, iterator_val, model, optimizer, token_converter, action_converter)
    task.run()

def _get_device():
    if torch.cuda.is_available():
        return torch.device(0)
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    _main()
