from app.composers import get_composer
from app.distributions import get_distribution
from app.embeddings import get_embedding
from app.representations import get_representation
from app.rnn import get_rnn

def get_model(device, generative, action_converter, token_converter, tag_converter, config):
    """
    :type device: torch.device
    :type generative: bool
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.token.TokenConverter
    :type tag_converter: app.data.converters.tag.TagConverter
    :type config: object
    :rtype: app.models.model.Model
    """
    if config.type == 'rnng':
        from app.models.rnng.stack import Stack
        if generative:
            token_size = config.size.rnn
        else:
            token_size = config.size.token
        non_terminal_count = action_converter.count_non_terminals()
        action_embedding = get_embedding(action_converter.count(), config.size.action, config.embedding)
        non_terminal_embedding = get_embedding(non_terminal_count, config.size.rnn, config.embedding)
        non_terminal_compose_embedding = get_embedding(non_terminal_count, config.size.rnn, config.embedding)
        token_embedding = get_embedding(token_converter.count(), token_size, config.embedding)
        embeddings = (action_embedding, token_embedding, non_terminal_embedding, non_terminal_compose_embedding)
        action_history = Stack(get_rnn(device, config.size.action, config.rnn))
        token_buffer = Stack(get_rnn(device, config.size.rnn, config.rnn))
        stack = Stack(get_rnn(device, config.size.rnn, config.rnn))
        structures = (action_history, token_buffer, stack)
        converters = (action_converter, token_converter, tag_converter)
        representation = get_representation(config.size.rnn, config.rnn.hidden_size, config.representation)
        composer = get_composer(device, config)
        sizes = (config.size.action, token_size, config.size.rnn, config.rnn.hidden_size)
        base_args = (device, embeddings, structures, converters, representation, composer, sizes, config.threads)
        if generative:
            from app.models.rnng.generative import GenerativeRNNG
            token_distribution = get_distribution(device, action_converter, config)
            model = GenerativeRNNG(*base_args, token_distribution)
        else:
            from app.models.rnng.discriminative import DiscriminativeRNNG
            pos_embedding = get_embedding(token_converter.count(), config.size.pos, config.embedding)
            model = DiscriminativeRNNG(*base_args, config.size.token, config.size.pos, pos_embedding)
    else:
        raise Exception(f'Unknown model: {config.type}')
    return model.to(device)
