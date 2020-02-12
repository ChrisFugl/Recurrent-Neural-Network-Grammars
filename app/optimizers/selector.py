from torch import optim

def get_optimizer(config, parameters):
    """
    :rtype: torch.optim.Optimizer
    """
    if config.type == 'adam':
        return optim.Adam(
            parameters,
            lr=config.learning_rate,
            betas=[config.beta1, config.beta2],
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    else:
        raise Exception(f'Unknown optimizer: {config.type}')
