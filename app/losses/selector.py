def get_loss(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.losses.loss.Loss
    """
    if config.type == 'negative_likelihood':
        from app.losses.negative_likelihood import NegativeLikelihoodLoss
        return NegativeLikelihoodLoss(device)
    else:
        raise Exception(f'Unknown loss: {config.type}')
