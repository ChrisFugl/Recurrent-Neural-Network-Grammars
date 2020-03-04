def get_inferer(device, model, sampler, config):
    """
    :type device: torch.device
    :type model: app.models.model.Model
    :type sampler: app.samplers.sampler.Sampler
    :type config: object
    :rtype: app.inferers.inferer.Inferer
    """
    if config.type == 'importance':
        raise NotImplementedError('importance sampling has not yet been implemented')
    elif config.type == 'parser':
        from app.inferers.parser import ParserInferer
        return ParserInferer(device, model, sampler, config.samples)
    else:
        pass
