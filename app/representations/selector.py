def get_representation(rnn_input_size, rnn_hidden_size, config):
    """
    :type rnn_input_size: int
    :type rnn_hidden_size: int
    :type config: object
    """
    if config.type == 'attentive':
        from app.representations.attentive import AttentiveRepresentation
        return AttentiveRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    elif config.type == 'attentive_no_history':
        from app.representations.attentive_no_history import AttentiveNoHistoryRepresentation
        return AttentiveNoHistoryRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    elif config.type == 'attentive_stack_only':
        from app.representations.attentive_stack_only import AttentiveStackOnlyRepresentation
        return AttentiveStackOnlyRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    elif config.type == 'buffer_only':
        from app.representations.buffer_only import BufferOnlyRepresentation
        return BufferOnlyRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    elif config.type == 'history_only':
        from app.representations.history_only import HistoryOnlyRepresentation
        return HistoryOnlyRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    if config.type == 'stack_only':
        from app.representations.stack_only import StackOnlyRepresentation
        return StackOnlyRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    elif config.type == 'vanilla':
        from app.representations.vanilla import VanillaRepresentation
        return VanillaRepresentation(rnn_hidden_size, rnn_input_size, config.dropout)
    else:
        raise Exception(f'Unknown representation: {config.type}')
