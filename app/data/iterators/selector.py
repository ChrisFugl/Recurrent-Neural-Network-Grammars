def get_iterator(config, tokens, actions, token_converter, action_converter, device):
    """
    :type config: object
    :type tokens: list of list of str
    :type actions: list of list of str
    :type token_converter: app.data.converters.TokenConverter
    :type action_converter: app.data.converters.action.ActionConverter
    :type device: torch.device
    :rtype: app.iterators.iterator.Iterator
    """
    if config.type == 'unordered':
        from app.data.iterators.unordered import UnorderedIterator
        return UnorderedIterator(tokens, actions, token_converter, action_converter, device, config.batch_size, config.shuffle)
    else:
        raise Exception(f'Unknown iterator: {config.type}')
