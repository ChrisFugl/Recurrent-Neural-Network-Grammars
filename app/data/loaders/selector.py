def get_loader(config):
    if config.type == 'oracle':
        from app.data.loaders.oracle import OracleLoader
        return OracleLoader(config.data_dir)
    elif config.type == 'raw':
        from app.data.loaders.raw import RawLoader
        return RawLoader(config.data_dir)
    else:
        raise Exception(f'Unknown loader: {config.type}')
