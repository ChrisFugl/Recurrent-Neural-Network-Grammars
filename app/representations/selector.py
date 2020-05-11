def get_representation(device, rnn_input_size, rnn_hidden_size, dropout_type, config):
    """
    :type device: torch.device
    :type rnn_input_size: int
    :type rnn_hidden_size: int
    :type dropout_type: str
    :type config: object
    """
    if config.type == 'attentive':
        from app.representations.attentive import AttentiveRepresentation
        return AttentiveRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'attentive_buffer':
        from app.representations.attentive_buffer import AttentiveBufferRepresentation
        return AttentiveBufferRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'attentive_no_history':
        from app.representations.attentive_no_history import AttentiveNoHistoryRepresentation
        return AttentiveNoHistoryRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'attentive_stack_only':
        from app.representations.attentive_stack_only import AttentiveStackOnlyRepresentation
        return AttentiveStackOnlyRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'buffer_only':
        from app.representations.buffer_only import BufferOnlyRepresentation
        return BufferOnlyRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'history_only':
        from app.representations.history_only import HistoryOnlyRepresentation
        return HistoryOnlyRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    if config.type == 'stack_only':
        from app.representations.stack_only import StackOnlyRepresentation
        return StackOnlyRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    elif config.type == 'vanilla':
        from app.representations.vanilla import VanillaRepresentation
        return VanillaRepresentation(device, rnn_hidden_size, rnn_input_size, dropout_type, config.dropout)
    else:
        raise Exception(f'Unknown representation: {config.type}')
