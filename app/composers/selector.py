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
        rnn_config['bidirectional'] = True
        rnn_config['num_layers'] = config.composer.num_layers
        birnn = get_rnn(device, config.embedding.size, rnn_config)
        return BiRNNComposer(birnn, config.embedding.size)
    else:
        raise Exception(f'Unknown composer: {type}')
