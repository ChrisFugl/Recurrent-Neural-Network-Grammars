from app.data.converters.cluster import ClusterConverter

def get_distribution(device, config):
    """
    :type device: torch.device
    :type config: object
    :rtype: app.distributions.distribution.Distribution
    """
    model_config = config.model
    distribution_config = model_config.distribution
    distribution_type = distribution_config.type
    if distribution_type == 'class_based_softmax':
        from app.distributions.class_based_softmax import ClassBasedSoftmax
        cluster_converter = ClusterConverter(distribution_config.cluster)
        return ClassBasedSoftmax(device, cluster_converter, model_config.representation.size).to(device)
    else:
        raise Exception(f'Unknown distribution: {distribution_type}')
