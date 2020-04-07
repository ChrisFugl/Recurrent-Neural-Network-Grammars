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
    elif config.type == 'time':
        from app.evaluators.time import TimeEvaluator
        return TimeEvaluator(config.interval_s, config.pretraining)
    else:
        raise Exception(f'Unknown evaluator: {config.type}')
