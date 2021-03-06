from app.rnn import get_rnn
from copy import deepcopy

def get_composer(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.composers.composer.Composer
    """
    type = config.composer.type
    if type == 'birnn':
        from app.composers.birnn import BiRNNComposer
        rnn_config = deepcopy(config.rnn)
        rnn_config['bidirectional'] = False
        if config.composer.num_layers == 1 and config.dropout_type == 'normal':
            rnn_config['dropout'] = None
        rnn_config['num_layers'] = config.composer.num_layers
        rnn_forward = get_rnn(device, config.rnn.hidden_size, config.dropout_type, rnn_config)
        rnn_backward = get_rnn(device, config.rnn.hidden_size, config.dropout_type, rnn_config)
        return BiRNNComposer(device, rnn_forward, rnn_backward, config.rnn.hidden_size, config.size.rnn, config.composer.dropout, config.dropout_type)
    else:
        raise Exception(f'Unknown composer: {type}')
