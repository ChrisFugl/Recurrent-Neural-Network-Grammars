from app.models.load import load_saved_model

def get_sampler(device, generative, token_count, action_count, non_terminal_count, config):
    """
    :type device: torch.device
    :type generative: bool
    :type token_count: int
    :type action_count: int
    :type non_terminal_count: int
    :type config: object
    :rtype: app.samplers.sampler.Sampler
    """
    if config.type == 'ancestral':
        from app.samplers.ancestral import AncestralSampler
        assert config.load_dir is not None, 'Ancestral sampling requires a loading directory containing a saved model.'
        model = load_saved_model(device, generative, token_count, action_count, non_terminal_count, config.load_dir)
        return AncestralSampler(device, model, config.posterior_scaling)
    else:
        raise Exception(f'Unknown sampler: {config.type}')
