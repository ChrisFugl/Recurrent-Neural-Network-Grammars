from app.composers import get_composer
from app.distributions import get_distribution
from app.embeddings import get_embedding
from app.embeddings.pretrained import PretrainedEmbedding
from app.representations import get_representation
from app.rnn import get_rnn

def get_model(device, generative, action_converter, token_converter, tag_converter, nt_converter, config):
    """
    :type device: torch.device
    :type generative: bool
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.token.TokenConverter
    :type tag_converter: app.data.converters.tag.TagConverter
    :type nt_converter: app.data.converters.non_terminal.NonTerminalConverter
    :type config: object
    :rtype: app.models.model.Model
    """
    if config.type == 'rnng' or config.type == 'parallel_rnng':
        token_size = config.size.rnn if generative else config.size.token
        nt_count = nt_converter.count()
        action_embedding = get_embedding(action_converter.count(), config.size.action, config.action_emb_drop, config.embedding)
        nt_embedding = get_embedding(nt_count, config.size.rnn, config.nt_emb_drop, config.embedding)
        nt_compose_embedding = get_embedding(nt_count, config.rnn.hidden_size, config.nt_com_emb_drop, config.embedding)
        token_embedding = get_embedding(token_converter.count(), token_size, config.token_emb_drop, config.embedding)
        embeddings = (action_embedding, token_embedding, nt_embedding, nt_compose_embedding)
        converters = (action_converter, token_converter, tag_converter, nt_converter)
        representation = get_representation(config.size.rnn, config.rnn.hidden_size, config.representation)
        composer = get_composer(device, config)
        sizes = (config.size.action, token_size, config.size.rnn, config.rnn.hidden_size)
        pretrained = None
        if not generative and config.pretrained_word_vectors is not None:
            pretrained = PretrainedEmbedding(device, config.pretrained_word_vectors, config.pretrained_emb_drop)
        if config.type == 'rnng':
            from app.models.rnng.buffer import Buffer
            from app.models.rnng.history import History
            from app.models.rnng.stack import Stack
            action_history = History(device, get_rnn(device, config.size.action, config.rnn))
            token_buffer = Buffer(device, get_rnn(device, config.size.rnn, config.rnn))
            stack = Stack(get_rnn(device, config.size.rnn, config.rnn))
            structures = (action_history, token_buffer, stack)
            base_args = (device, embeddings, structures, converters, representation, composer, sizes, config.threads)
            if generative:
                from app.models.rnng.generative import GenerativeRNNG
                token_distribution = get_distribution(device, action_converter, config)
                model = GenerativeRNNG(*base_args, token_distribution)
            else:
                from app.models.rnng.discriminative import DiscriminativeRNNG
                pos_embedding = get_embedding(tag_converter.count(), config.size.pos, config.pos_emb_drop, config.embedding)
                model = DiscriminativeRNNG(*base_args, config.size.pos, pos_embedding, config.unk_token_prob, pretrained)
        else:
            from app.models.parallel_rnng.history_lstm import HistoryLSTM
            from app.models.parallel_rnng.stack_lstm import StackLSTM
            rnn_args = [config.rnn.hidden_size, config.rnn.num_layers, config.rnn.bias, config.rnn.dropout]
            action_history = HistoryLSTM(device, config.size.action, *rnn_args, config.rnn.weight_drop)
            stack = StackLSTM(device, config.size.rnn, *rnn_args, config.rnn.weight_drop)
            structures = [action_history, None, stack]
            base_args = (device, embeddings, structures, converters, representation, composer, sizes, config.sample_stack_size)
            if generative:
                from app.models.parallel_rnng.generative import GenerativeParallelRNNG
                from app.models.parallel_rnng.output_buffer_lstm import OutputBufferLSTM
                token_buffer = OutputBufferLSTM(device, config.size.rnn, *rnn_args, config.rnn.weight_drop)
                structures[1] = token_buffer
                model = GenerativeParallelRNNG(*base_args)
            else:
                from app.models.parallel_rnng.discriminative import DiscriminativeParallelRNNG
                from app.models.parallel_rnng.input_buffer_lstm import InputBufferLSTM
                pos_embedding = get_embedding(tag_converter.count(), config.size.pos, config.pos_emb_drop, config.embedding)
                token_buffer = InputBufferLSTM(device, config.size.rnn, *rnn_args, config.rnn.weight_drop)
                structures[1] = token_buffer
                model = DiscriminativeParallelRNNG(*base_args, config.size.pos, pos_embedding, config.unk_token_prob, pretrained)
    elif config.type == 'rnn_lm':
        assert generative, 'RNN LM is only defined for generative parsing.'
        from app.models.rnn_lm.rnn_lm import RNNLM
        action_embedding_size = config.size.rnn
        action_count = action_converter.count()
        rnn = get_rnn(device, config.size.rnn, config.rnn)
        action_embedding = get_embedding(action_count, action_embedding_size, config.action_emb_drop, config.embedding)
        model = RNNLM(device, rnn, action_embedding, action_embedding_size, action_converter)
    elif config.type == 'rnn_parser':
        assert not generative, 'RNN parser is only defined for discriminative parsing.'
        from app.models.rnn_parser.rnn_parser import RNNParser
        action_embedding_size = config.size.rnn
        encoder_rnn = get_rnn(device, config.size.rnn, config.rnn)
        encoder_output_size = encoder_rnn.get_output_size()
        decoder_input_size = action_embedding_size + encoder_output_size
        decoder_rnn = get_rnn(device, decoder_input_size, config.rnn)
        action_count = action_converter.count()
        token_count = token_converter.count()
        action_embedding = get_embedding(action_count, action_embedding_size, config.action_emb_drop, config.embedding)
        token_embedding = get_embedding(token_count, config.size.rnn, config.token_emb_drop, config.embedding)
        model = RNNParser(device, encoder_rnn, decoder_rnn, action_embedding, config.size.rnn, token_embedding, action_converter)
    else:
        raise Exception(f'Unknown model: {config.type}')
    return model.to(device)
