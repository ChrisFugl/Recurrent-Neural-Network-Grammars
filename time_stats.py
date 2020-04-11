from app.data.iterators import get_iterator
from app.data.loaders import get_loader
from app.data.converters.action import ActionConverter
from app.data.converters.non_terminal import NonTerminalConverter
from app.data.converters.tag import TagConverter
from app.data.converters.token import TokenConverter
from app.models import get_model
from app.tasks.time_stats import TimeStatsTask
from app.utils import get_device, is_generative
import hydra

@hydra.main(config_path='configs/time_stats.yaml')
def main(config):
    loader = get_loader(config.loader)
    _, actions, _, unknownified_tokens, tags = loader.load_train()
    generative = is_generative(config.type)
    device = get_device(config.gpu)
    token_converter = TokenConverter(unknownified_tokens)
    tag_converter = TagConverter(tags)
    action_converter = ActionConverter(generative, actions)
    non_terminal_converter = NonTerminalConverter(actions)
    iterator_converters = (action_converter, token_converter, tag_converter)
    model_converters = (action_converter, token_converter, tag_converter, non_terminal_converter)
    iterator = get_iterator(device, *iterator_converters, unknownified_tokens, actions, tags, config.iterator)
    model = get_model(device, generative, *model_converters, config.model)
    task = TimeStatsTask(model, iterator)
    task.run()

if __name__ == '__main__':
    main()
