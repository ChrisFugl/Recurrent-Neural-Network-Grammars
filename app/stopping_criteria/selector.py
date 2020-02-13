def get_stopping_criterion(config):
    if config.type == 'epochs':
        from app.stopping_criteria.epochs import EpochsStoppingCriterion
        return EpochsStoppingCriterion(config.epochs)
    elif config.type == 'early_stopping':
        from app.stopping_criteria.early_stopping import EarlyStoppingStoppingCriterion
        return EarlyStoppingStoppingCriterion(config.epsilon)
    else:
        raise Exception(f'Unknown stopping criterion: {config.type}')
