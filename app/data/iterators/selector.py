def get_iterator(device, action_converter, token_converter, tokens, actions_strings, config):
    """
    :type device: torch.device
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.TokenConverter
    :type tokens: list of list of str
    :type actions_strings: list of list of str
    :type config: object
    :rtype: app.iterators.iterator.Iterator
    """
    if config.type == 'unordered':
        from app.data.iterators.unordered import UnorderedIterator
        return UnorderedIterator(
            device,
            action_converter,
            token_converter,
            config.batch_size,
            config.shuffle,
            tokens,
            actions_strings
        )
    else:
        raise Exception(f'Unknown iterator: {config.type}')
