from app.data.action_set import get_action_set
from app.checkpoints import get_checkpoint
from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.data.converters.action import ActionConverter
from app.data.converters.token import TokenConverter
from app.evaluators import get_evaluator
from app.learning_rate_schedulers import get_learning_rate_scheduler
from app.losses import get_loss
from app.models import get_model
from app.optimizers import get_optimizer
from app.stopping_criteria import get_stopping_criterion
from app.tasks.train import TrainTask
from app.utils import get_device, is_generative, set_seed
import hydra

@hydra.main(config_path='configs/train.yaml')
def _main(config):
    set_seed(config.seed)
    loader = get_loader(config.loader)
    _, actions_train, _, unknownified_tokens_train = loader.load_train()
    _, actions_val, _, unknownified_tokens_val = loader.load_val()
    generative = is_generative(config.type)
    device = get_device(config.gpu)
    token_converter = TokenConverter(unknownified_tokens_train)
    action_converter = ActionConverter(token_converter, generative, actions_train)
    action_set = get_action_set(config.type)
    iterator_train = get_iterator(device, action_converter, token_converter, unknownified_tokens_train, actions_train, config.iterator)
    iterator_val = get_iterator(device, action_converter, token_converter, unknownified_tokens_val, actions_val, config.iterator)
    token_count = token_converter.count()
    action_count = action_converter.count()
    non_terminal_count = action_converter.count_non_terminals()
    model = get_model(device, generative, token_count, action_count, non_terminal_count, action_set, config.model)
    loss = get_loss(device, config.loss)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    learning_rate_scheduler = get_learning_rate_scheduler(optimizer, config.lr_scheduler)
    stopping_criterion = get_stopping_criterion(config.stopping_criterion)
    checkpoint = get_checkpoint(config.checkpoint)
    evaluator = get_evaluator(config.evaluator)
    task = TrainTask(
        device,
        iterator_train, iterator_val,
        model, loss, optimizer, learning_rate_scheduler,
        stopping_criterion, checkpoint, evaluator,
        config.load_checkpoint, token_count, non_terminal_count, action_count,
    )
    task.run()

if __name__ == '__main__':
    _main()
