from app.distributions.distribution import Distribution
import torch
from torch import nn

class ClassBasedSoftmax(Distribution):

    def __init__(self, device, cluster_converter, action_converter, representation_size):
        """
        :type device: torch.device
        :type cluster_converter: app.data.converters.cluster.ClusterConverter
        :type action_converter: app.data.converters.action.ActionConverter
        :type representation_size: int
        """
        super().__init__()
        self._device = device
        self._cluster_converter = cluster_converter
        self._action_converter = action_converter
        self._logits2log_prob = nn.LogSoftmax(dim=2)
        clusters_count = cluster_converter.count()
        self._representation2clusters = nn.Linear(in_features=representation_size, out_features=clusters_count, bias=True)
        cluster2logits = []
        for cluster_index in range(clusters_count):
            tokens_count = cluster_converter.count_tokens_in_cluster(cluster_index)
            cluster = nn.Linear(in_features=representation_size, out_features=tokens_count, bias=True).to(device)
            cluster2logits.append(cluster)
        self._cluster2logits = nn.ModuleList(cluster2logits)

    def log_prob(self, representation, token, posterior_scaling=1.0):
        """
        Compute log probability of given value.

        :type representation: torch.Tensor
        :type posterior_scaling: float
        :type token: str
        :rtype: torch.Tensor
        """
        cluster_index, token_index = self._cluster_converter.token2cluster(token)
        cluster_logits = self._representation2clusters(representation)
        cluster_log_probs = self._logits2log_prob(posterior_scaling * cluster_logits)
        cluster_log_prob = cluster_log_probs[:, :, cluster_index]
        cluster2logits = self._cluster2logits[cluster_index]
        token_logits = cluster2logits(representation)
        token_log_probs = self._logits2log_prob(token_logits)
        token_log_prob = token_log_probs[:, :, token_index]
        return cluster_log_prob + token_log_prob

    def log_probs(self, representation, posterior_scaling=1.0):
        """
        Log probabilities of all elements in distribution.

        :type representation: torch.Tensor
        :type posterior_scaling: float
        :rtype: torch.Tensor
        """
        cluster_logits = self._representation2clusters(representation)
        cluster_log_probs = self._logits2log_prob(posterior_scaling * cluster_logits).squeeze()
        clusters_count = self._cluster_converter.count()
        tokens_count = self._cluster_converter.count_tokens()
        terminal_offset = self._action_converter.get_terminal_offset()
        log_probs = torch.empty((tokens_count,), device=self._device, dtype=torch.float)
        for cluster_index in range(clusters_count):
            cluster_log_prob = cluster_log_probs[cluster_index]
            cluster2logits = self._cluster2logits[cluster_index]
            token_logits = cluster2logits(representation)
            token_log_probs = self._logits2log_prob(posterior_scaling * token_logits).squeeze()
            cluster_tokens_count = self._cluster_converter.count_tokens_in_cluster(cluster_index)
            for token_index in range(cluster_tokens_count):
                token_log_prob = token_log_probs[token_index]
                token = self._cluster_converter.cluster2token((cluster_index, token_index))
                action_index = self._action_converter.token2integer(token)
                log_prob_index = action_index - terminal_offset
                log_probs[log_prob_index] = cluster_log_prob + token_log_prob
        return log_probs

    def __str__(self):
        return f'ClassBasedSoftmax(n_clusters={self._cluster_converter.count()})'
