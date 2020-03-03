def get_rnn(device, input_size, config):
    """
    :type device: torch.device
    :type input_size: int
    :type config: object
    :rtype: app.rnn.rnn.RNN
    """
    if config.type == 'lstm':
        from app.rnn.lstm import LSTM
        return LSTM(
            device,
            input_size,
            config.hidden_size,
            config.num_layers,
            config.bias,
            config.dropout,
            config.bidirectional
        )
    else:
        raise Exception(f'Unknown RNN: {config.type}')
