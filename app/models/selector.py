from app.composers import get_composer
from app.constants import ACTION_EMBEDDING_OFFSET
from app.distributions import get_distribution
from app.embeddings import get_embedding
from app.memories import get_memory
from app.representations import get_representation
from app.stacks import get_stack

def get_model(device, generative, token_converter, action_converter, action_set, config):
    """
    :type device: torch.device
    :type generative: bool
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.token.TokenConverter
    :type action_set: app.data.action_set.ActionSet
    :type config: object
    :rtype: app.models.model.Model
    """
    if config.type == 'rnng':
        from app.models.rnng import RNNG
        token_count = token_converter.count()
        action_count = action_converter.count()
        non_terminal_count = action_converter.count_non_terminals()
        action_embedding = get_embedding(action_count, config.embedding)
        non_terminal_embedding = get_embedding(non_terminal_count, config.embedding)
        non_terminal_compose_embedding = get_embedding(non_terminal_count, config.embedding)
        token_embedding = get_embedding(token_count, config.embedding)
        rnn_args = [device, config.embedding.size, config.rnn]
        action_history = get_memory(config.memory, rnn_args=rnn_args)
        token_buffer = get_memory(config.memory, rnn_args=rnn_args)
        stack_rnn_args = [device, config.embedding.size, config.rnn]
        stack = get_stack(config.stack, rnn_args=stack_rnn_args)
        representation = get_representation(config.embedding.size, config.representation)
        composer = get_composer(device, config)
        token_distribution = None if not generative else get_distribution(device, config)
        return RNNG(
            device,
            generative,
            action_embedding,
            token_embedding,
            non_terminal_embedding,
            non_terminal_compose_embedding,
            action_history,
            token_buffer,
            stack,
            representation,
            config.representation.size,
            composer,
            token_distribution,
            non_terminal_count,
            action_set,
            config.threads,
            config.reverse_tokens,
        ).to(device)
    else:
        raise Exception(f'Unknown model: {config.type}')
