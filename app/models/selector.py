from app.composers import get_composer
from app.distributions import get_distribution
from app.embeddings import get_embedding
from app.memories import get_memory
from app.representations import get_representation
from app.stacks import get_stack

def get_model(device, token_count, action_count, non_terminal_count, action_set, config):
    """
    :type device: torch.device
    :type token_count: int
    :type action_count: int
    :type non_terminal_count: int
    :type action_set: app.data.action_set.ActionSet
    :type config: object
    :rtype: app.models.model.Model
    """
    model_config = config.model
    if model_config.type == 'rnng':
        from app.models.rnng import RNNG
        batch_size = config.iterator.batch_size
        action_embedding = get_embedding(action_count, model_config.embedding)
        non_terminal_embedding = get_embedding(non_terminal_count, model_config.embedding)
        non_terminal_compose_embedding = get_embedding(non_terminal_count, model_config.embedding)
        token_embedding = get_embedding(token_count, model_config.embedding)
        rnn_args = [device, model_config.embedding.size, batch_size, model_config.rnn]
        action_history = get_memory(model_config.memory, rnn_args=rnn_args)
        token_buffer = get_memory(model_config.memory, rnn_args=rnn_args)
        stack_rnn_args = [device, model_config.embedding.size, 1, model_config.rnn]
        stack = get_stack(model_config.stack, rnn_args=stack_rnn_args)
        representation = get_representation(model_config.embedding.size, model_config.representation)
        composer = get_composer(device, config)
        token_distribution = get_distribution(device, config)
        return RNNG(
            device,
            action_embedding,
            token_embedding,
            non_terminal_embedding,
            non_terminal_compose_embedding,
            action_history,
            token_buffer,
            stack,
            representation,
            model_config.representation.size,
            composer,
            token_distribution,
            non_terminal_count,
            action_set,
            model_config.threads
        ).to(device)
    else:
        raise Exception(f'Unknown model: {model_config.type}')
