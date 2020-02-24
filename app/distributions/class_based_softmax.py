from app.distributions.distribution import Distribution
from torch import nn

class ClassBasedSoftmax(Distribution):

    def __init__(self, device, cluster_converter, representation_size):
        """
        :type device: torch.device
        :type cluster_converter: app.data.converters.cluster.ClusterConverter
        :type representation_size: int
        """
        super().__init__()
        self._cluster_converter = cluster_converter
        self._logits2log_prob = nn.LogSoftmax(dim=2)
        clusters_count = cluster_converter.count()
        self._representation2clusters = nn.Linear(in_features=representation_size, out_features=clusters_count, bias=True)
        cluster2logits = []
        for cluster_index in range(clusters_count):
            tokens_count = cluster_converter.count_tokens(cluster_index)
            cluster = nn.Linear(in_features=representation_size, out_features=tokens_count, bias=True).to(device)
            cluster2logits.append(cluster)
        self._cluster2logits = nn.ModuleList(cluster2logits)

    def log_prob(self, representation, token):
        """
        Compute log probability of given value.

        :type representation: torch.Tensor
        :type token: str
        :rtype: torch.Tensor
        """
        cluster_index, token_index = self._cluster_converter.token2cluster(token)
        cluster_logits = self._representation2clusters(representation)
        cluster_log_probs = self._logits2log_prob(cluster_logits)
        cluster_log_prob = cluster_log_probs[:, :, cluster_index]
        cluster2logits = self._cluster2logits[cluster_index]
        token_logits = cluster2logits(representation)
        token_log_probs = self._logits2log_prob(token_logits)
        token_log_prob = token_log_probs[:, :, token_index]
        return cluster_log_prob + token_log_prob

    def sample(self, representation, n):
        """
        Sample n values from distribution.

        :type representation: torch.Tensor
        :type n: int
        :rtype: torch.Tensor
        """
        # TODO
        raise NotImplementedError('not implemented yet')
