from app.constants import ACTION_EMBEDDING_OFFSET, NON_TERMINAL_EMBEDDING_OFFSET, TAG_EMBEDDING_OFFSET, TOKEN_EMBEDDING_OFFSET
from app.checkpoints import get_checkpoint
from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.data.converters.action import ActionConverter
from app.data.converters.non_terminal import NonTerminalConverter
from app.data.converters.tag import TagConverter
from app.data.converters.token import TokenConverter
from app.evaluators import get_evaluator
from app.learning_rate_schedulers import get_learning_rate_scheduler
from app.losses import get_loss
from app.models import get_model
from app.optimizers import get_optimizer
from app.samplers.greedy import GreedySampler
from app.stopping_criteria import get_stopping_criterion
from app.tasks.train import TrainTask
from app.utils import get_device, is_generative, set_seed
import hydra

@hydra.main(config_path='configs/train.yaml')
def _main(config):
    set_seed(config.seed)
    loader = get_loader(config.loader)
    _, actions_train, _, unknownified_tokens_train, tags_train = loader.load_train()
    _, actions_val, _, unknownified_tokens_val, tags_val = loader.load_val()
    generative = is_generative(config.type)
    device = get_device(config.gpu)
    token_converter = TokenConverter(unknownified_tokens_train)
    tag_converter = TagConverter(tags_train)
    action_converter = ActionConverter(generative, actions_train)
    non_terminal_converter = NonTerminalConverter(actions_train)
    iterator_converters = (action_converter, token_converter, tag_converter)
    model_converters = (action_converter, token_converter, tag_converter, non_terminal_converter)
    iterator_train = get_iterator(device, *iterator_converters, unknownified_tokens_train, actions_train, tags_train, config.iterator)
    iterator_val = get_iterator(device, *iterator_converters, unknownified_tokens_val, actions_val, tags_val, config.iterator)
    model = get_model(device, generative, *model_converters, config.model)
    loss = get_loss(device, config.loss)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    learning_rate_scheduler = get_learning_rate_scheduler(optimizer, config.lr_scheduler)
    stopping_criterion = get_stopping_criterion(config.stopping_criterion)
    if generative:
        sampler = None
    else:
        sampler = GreedySampler(device, model, iterator_val, action_converter, 1.0, log=False)
    checkpoint = get_checkpoint(config.checkpoint)
    evaluator = get_evaluator(config.evaluator)
    task = TrainTask(
        device,
        iterator_train,
        iterator_val,
        model,
        loss,
        optimizer,
        learning_rate_scheduler,
        stopping_criterion,
        checkpoint,
        evaluator,
        sampler,
        config.log_train_every,
        config.load_checkpoint,
        token_converter.count() - TOKEN_EMBEDDING_OFFSET,
        tag_converter.count() - TAG_EMBEDDING_OFFSET,
        non_terminal_converter.count() - NON_TERMINAL_EMBEDDING_OFFSET,
        action_converter.count() - ACTION_EMBEDDING_OFFSET,
    )
    task.run()

if __name__ == '__main__':
    _main()
