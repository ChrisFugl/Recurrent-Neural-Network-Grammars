def get_loss(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.losses.loss.Loss
    """
    if config.type == 'nll':
        from app.losses.nll import NLLLoss
        return NLLLoss(device)
    elif config.type == 'nll_reduced':
        from app.losses.nll_reduced import NLLReducedLoss
        return NLLReducedLoss(device)
    else:
        raise Exception(f'Unknown loss: {config.type}')
