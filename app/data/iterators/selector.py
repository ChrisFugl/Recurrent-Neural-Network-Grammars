def get_iterator(device, action_converter, token_converter, tag_converter, tokens, unknownified_tokens, actions_strings, tags, config):
    """
    :type device: torch.device
    :type action_converter: app.data.converters.action.ActionConverter
    :type token_converter: app.data.converters.token.TokenConverter
    :type tag_converter: app.data.converters.tag.TagConverter
    :type tokens: list of list of str
    :type unknownified_tokens: list of list of str
    :type actions_strings: list of list of str
    :type tags: list of list of str
    :type config: object
    :rtype: app.iterators.iterator.Iterator
    """
    iterator_args = (
        device, action_converter, token_converter, tag_converter, config.batch_size, config.shuffle,
        tokens, unknownified_tokens, actions_strings, tags,
    )
    if config.type == 'ordered':
        from app.data.iterators.ordered import OrderedIterator
        return OrderedIterator(*iterator_args)
    elif config.type == 'unordered':
        from app.data.iterators.unordered import UnorderedIterator
        return UnorderedIterator(*iterator_args)
    else:
        raise Exception(f'Unknown iterator: {config.type}')
