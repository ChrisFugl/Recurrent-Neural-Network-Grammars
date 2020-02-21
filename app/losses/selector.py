def get_loss(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.losses.loss.Loss
    """
    if config.type == 'negative_log_likelihood':
        from app.losses.negative_log_likelihood import NegativeLogLikelihoodLoss
        return NegativeLogLikelihoodLoss(device)
    else:
        raise Exception(f'Unknown loss: {config.type}')
