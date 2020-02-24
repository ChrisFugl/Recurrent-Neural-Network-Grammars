from app.data.converters.cluster import ClusterConverter

def get_distribution(config):
    """
    :type config: object
    :rtype: app.distributions.distribution.Distribution
    """
    if config.distribution.type == 'class_based_softmax':
        from app.distributions.class_based_softmax import ClassBasedSoftmax
        cluster_converter = ClusterConverter(config.distribution.cluster)
        return ClassBasedSoftmax(cluster_converter, config.representation.size)
    else:
        raise Exception(f'Unknown distribution: {config.distribution.type}')
