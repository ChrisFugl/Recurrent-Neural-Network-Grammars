def get_checkpoint(config):
    """
    :type config: object
    :rtype: app.checkpoints.checkpoint.Checkpoint
    """
    if config.type == 'never':
        from app.checkpoints.never import NeverCheckpoint
        return NeverCheckpoint()
    elif config.type == 'epoch':
        from app.checkpoints.epoch import EpochCheckpoint
        return EpochCheckpoint(config.epoch)
    elif config.type == 'batch':
        from app.checkpoints.batch import BatchCheckpoint
        return BatchCheckpoint(config.batch)
    elif config.type == 'time':
        from app.checkpoints.time import TimeCheckpoint
        return TimeCheckpoint(config.interval_s)
    else:
        raise Exception(f'Unknown checkpoint: {config.type}')
