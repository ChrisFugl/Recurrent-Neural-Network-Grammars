from app.rnn import get_rnn

def get_stack(config, **kwargs):
    """
    :type config: object
    :rtype: app.stacks.stack.Stack
    """
    if config.type == 'rnn':
        from app.stacks.rnn import StackRNN
        rnn_args = kwargs.get('rnn_args')
        rnn = get_rnn(*rnn_args)
        return StackRNN(rnn)
    else:
        raise Exception(f'Unknown stack: {config.type}')
