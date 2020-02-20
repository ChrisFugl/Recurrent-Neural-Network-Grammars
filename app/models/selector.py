from app.embeddings import get_embedding
from app.memories import get_memory
from app.representations import get_representation
from app.stacks import get_stack

def get_model(device, token_count, action_count, config):
    """
    :type device: torch.device
    :type token_count: int
    :type action_count: int
    :type config: object
    :rtype: app.models.model.Model
    """
    config_model = config.model
    if config_model.type == 'rnng':
        from app.models.rnng import RNNG
        batch_size = config.iterator.batch_size
        action_embedding = get_embedding(action_count, config.embedding)
        token_embedding = get_embedding(token_count, config.embedding)
        rnn_args = [device, config.embedding.size, batch_size, config.rnn]
        action_history = get_memory(config.memory, rnn_args=rnn_args)
        token_buffer = get_memory(config.memory, rnn_args=rnn_args)
        stack_rnn_args = [device, config.embedding.size, 1, config.rnn]
        stack = get_stack(config.stack, rnn_args=stack_rnn_args)
        representation = get_representation(config.embedding.size, config.representation)
        return RNNG(
            device,
            action_embedding,
            token_embedding,
            action_history,
            token_buffer,
            stack,
            representation,
            config.representation.size
        ).to(device)
    else:
        raise Exception(f'Unknown model: {config_model.type}')
