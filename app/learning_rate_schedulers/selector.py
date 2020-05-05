def get_learning_rate_scheduler(optimizer, config):
    """
    :type optimizer: torch.optim.Optimizer
    :type config: object
    :rtype: torch.optim.lr_scheduler._LRScheduler
    """
    if config.type == 'constant':
        from app.learning_rate_schedulers.constant import ConstantLearningRateScheduler
        return ConstantLearningRateScheduler(optimizer)
    elif config.type == 'delayed_exponential':
        from app.learning_rate_schedulers.delayed_exponential import DelayedExponentialLearningRateScheduler
        return DelayedExponentialLearningRateScheduler(optimizer, config.delay, config.decay)
    elif config.type == 'inverse_additive_decay':
        from app.learning_rate_schedulers.inverse_multiplicative_decay import InverseAdditiveDecayLearningRateScheduler
        return InverseAdditiveDecayLearningRateScheduler(optimizer, config.decay)
    else:
        raise Exception(f'Unknown learning rate scheduler: {config.type}')
