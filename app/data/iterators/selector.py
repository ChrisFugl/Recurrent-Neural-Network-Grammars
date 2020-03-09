def get_iterator(device, action_converter, token_converter, tokens, actions_strings, tags, config):
    """
    :type device: torch.device
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.TokenConverter
    :type tokens: list of list of str
    :type actions_strings: list of list of str
    :type tags: list of list of str
    :type config: object
    :rtype: app.iterators.iterator.Iterator
    """
    if config.type == 'ordered':
        from app.data.iterators.ordered import OrderedIterator
        return OrderedIterator(device, action_converter, token_converter, config.batch_size, config.shuffle, tokens, actions_strings, tags)
    elif config.type == 'unordered':
        from app.data.iterators.unordered import UnorderedIterator
        return UnorderedIterator(device, action_converter, token_converter, config.batch_size, config.shuffle, tokens, actions_strings, tags)
    else:
        raise Exception(f'Unknown iterator: {config.type}')
