def get_model(config, device):
    """
    :rtype: app.models.model.Model
    """
    if config.type == 'rnng':
        from app.models.rnng import RNNG
        return RNNG().to(device)
    else:
        raise Exception(f'Unknown model: {config.type}')
