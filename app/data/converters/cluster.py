import csv
import hydra
import pandas as pd

class ClusterConverter:
    """
    Responsible for:
    * converting between tokens and clusters
    * doing so in a deterministic way given a list of tokens
    """

    def __init__(self, cluster_file_path):
        """
        :type cluster_file_path: str
        """
        self._token2cluster, self._cluster2token = self._get_cluster_converters(cluster_file_path)

    def _get_cluster_converters(self, cluster_file_path):
        data = self._read_cluster(cluster_file_path)
        clusters = list(data.cluster.unique())
        cluster2index = {}
        cluster2token = [[] for _ in range(len(clusters))]
        token2cluster = {}
        for cluster_index, cluster in enumerate(clusters):
            cluster2index[cluster] = cluster_index
        for _, row in data.iterrows():
            cluster_index = cluster2index[row.cluster]
            cluster = cluster2token[cluster_index]
            cluster.append(row.token)
            token_index = len(cluster) - 1
            token2cluster[row.token] = cluster_index, token_index
        return token2cluster, cluster2token

    def _read_cluster(self, cluster_file_path):
        headers = ['cluster', 'token', 'count']
        dtypes = {'cluster': str, 'token': str, 'count': int}
        absolute_file_path = hydra.utils.to_absolute_path(cluster_file_path)
        data = pd.read_csv(absolute_file_path, sep='\t', header=None, names=headers, dtype=dtypes, quoting=csv.QUOTE_NONE)
        return data

    def count(self):
        """
        Count number of clusters.

        :rtype: int
        """
        return len(self._cluster2token)

    def count_tokens(self, cluster_index):
        """
        Count number of tokens in specific cluster.

        :type cluster_index: int
        :rtype: int
        """
        return len(self._cluster2token[cluster_index])

    def cluster2token(self, indices):
        """
        :type indices: int, int
        :rtype: str
        """
        cluster_index, token_index = indices
        return self._cluster2token[cluster_index][token_index]

    def token2cluster(self, token):
        """
        :type token: str
        :rtype: int, int
        """
        return self._token2cluster[token]
