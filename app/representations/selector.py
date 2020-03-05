def get_representation(embedding_size, config):
    """
    :type embedding_size: int
    :type config: object
    """
    if config.type == 'stack_only':
        from app.representations.stack_only import StackOnlyRepresentation
        return StackOnlyRepresentation(embedding_size, config.size, config.dropout)
    elif config.type == 'vanilla':
        from app.representations.vanilla import VanillaRepresentation
        return VanillaRepresentation(embedding_size, config.size, config.dropout)
    else:
        raise Exception(f'Unknown representation: {config.type}')
