from app.data.converters.cluster import ClusterConverter

def get_distribution(device, action_converter, config):
    """
    :type device: torch.device
    :type action_converter: app.data.converters.action.ActionConverter
    :type config: object
    :rtype: app.distributions.distribution.Distribution
    """
    distribution_config = config.distribution
    distribution_type = distribution_config.type
    if distribution_type == 'class_based_softmax':
        from app.distributions.class_based_softmax import ClassBasedSoftmax
        assert distribution_config.cluster is not None, 'Please specify a path to a cluster file.'
        cluster_converter = ClusterConverter(distribution_config.cluster)
        return ClassBasedSoftmax(device, cluster_converter, action_converter, config.size.rnn).to(device)
    else:
        raise Exception(f'Unknown distribution: {distribution_type}')
