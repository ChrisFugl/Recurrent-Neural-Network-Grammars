def get_loader(config, name=None):
    """
    :type config: object
    :type name: str
    """
    if config.type == 'oracle':
        from app.data.loaders.oracle import OracleLoader
        return OracleLoader(config.data_dir, name=name)
    elif config.type == 'penn':
        from app.data.loaders.penn import PennLoader
        return PennLoader(config.data_dir, config.train_sections, config.val_sections, config.test_sections)
    elif config.type == 'raw':
        from app.data.loaders.raw import RawLoader
        return RawLoader(config.data_dir)
    else:
        raise Exception(f'Unknown loader: {config.type}')
