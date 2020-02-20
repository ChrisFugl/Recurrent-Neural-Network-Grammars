def get_representation(embedding_size, config):
    """
    :type embedding_size: int
    :type config: object
    """
    if config.type == 'stack_only':
        from app.representations.stack_only import StackOnlyRepresentation
        return StackOnlyRepresentation(config.size)
    elif config.type == 'vanilla':
        from app.representations.vanilla import VanillaRepresentation
        return VanillaRepresentation(embedding_size, config.size)
    else:
        raise Exception(f'Unknown representation: {config.type}')
