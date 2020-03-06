def get_evaluator(config):
    """
    :type config: object
    :rtype: app.evaluators.evaluator.Evaluator
    """
    if config.type == 'epoch':
        from app.evaluators.epoch import EpochEvaluator
        return EpochEvaluator(config.epoch, config.pretraining)
    elif config.type == 'batch':
        from app.evaluators.batch import BatchEvaluator
        return BatchEvaluator(config.batch, config.pretraining)
    else:
        raise Exception(f'Unknown evaluator: {config.type}')
