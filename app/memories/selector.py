from app.rnn import get_rnn

def get_memory(config, **kwargs):
    """
    :type config: object
    :rtype: app.memories.memory.Memory
    """
    if config.type == 'rnn':
        from app.memories.rnn import RNNMemory
        rnn_args = kwargs.get('rnn_args')
        rnn = get_rnn(*rnn_args)
        return RNNMemory(rnn)
    else:
        raise Exception(f'Unknown memory: {config.type}')
