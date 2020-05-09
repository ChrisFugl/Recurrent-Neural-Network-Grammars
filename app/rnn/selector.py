def get_rnn(device, input_size, dropout_type, config):
    """
    :type device: torch.device
    :type input_size: int
    :type dropout_type: str
    :type config: object
    :rtype: app.rnn.rnn.RNN
    """
    if config.type == 'lstm':
        from app.rnn.lstm import LSTM
        return LSTM(
            device, input_size, config.hidden_size, config.num_layers, config.bias,
            config.dropout, config.bidirectional, config.weight_drop, dropout_type,
        ).to(device)
    elif config.type == 'layer_norm_lstm':
        from app.rnn.layer_norm_lstm import LayerNormLSTM
        return LayerNormLSTM(
            device, input_size, config.hidden_size, config.num_layers,
            config.dropout, config.weight_drop, dropout_type).to(device)
    else:
        raise Exception(f'Unknown RNN: {config.type}')
