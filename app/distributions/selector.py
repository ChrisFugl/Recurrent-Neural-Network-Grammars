from app.data.converters.cluster import ClusterConverter

def get_distribution(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.distributions.distribution.Distribution
    """
    if config.distribution.type == 'class_based_softmax':
        from app.distributions.class_based_softmax import ClassBasedSoftmax
        cluster_converter = ClusterConverter(config.distribution.cluster)
        return ClassBasedSoftmax(device, cluster_converter, config.representation.size).to(device)
    else:
        raise Exception(f'Unknown distribution: {config.distribution.type}')
