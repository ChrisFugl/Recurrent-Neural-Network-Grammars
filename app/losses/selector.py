def get_loss(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.losses.loss.Loss
    """
    if config.type == 'negative_action_log_likelihood':
        from app.losses.negative_action_log_likelihood import NegativeActionLogLikelihoodLoss
        return NegativeActionLogLikelihoodLoss(device)
    elif config.type == 'negative_tree_log_likelihood':
        from app.losses.negative_tree_log_likelihood import NegativeTreeLogLikelihoodLoss
        return NegativeTreeLogLikelihoodLoss(device)
    else:
        raise Exception(f'Unknown loss: {config.type}')
