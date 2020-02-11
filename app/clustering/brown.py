from app.clustering.clustering import Clustering
from app.constants import BROWN_CLUSTERING_TOOL_PATH
from app.data.preprocessing import get_unknownified_terms_from_oracle
import hydra
import shutil
import subprocess

class BrownClustering(Clustering):

    def __init__(self, input_path, output_path, clusters):
        """
        :type input_path: str
        :type output_path: str
        :type clusters: int
        """
        super().__init__()
        self._input_path = hydra.utils.to_absolute_path(input_path)
        self._output_path = hydra.utils.to_absolute_path(output_path)
        self._clusters = clusters
        self._tool_path = hydra.utils.to_absolute_path(BROWN_CLUSTERING_TOOL_PATH)
        self._terms_name = f'terms'
        self._terms_path = f'{self._terms_name}.txt'

    def cluster(self):
        # get terms from input oracle file
        with open(self._input_path, 'r') as input_file:
            oracle = input_file.read().split('\n')
        terms = get_unknownified_terms_from_oracle(oracle)
        terms_content = '\n'.join(terms)
        with open(self._terms_path, 'w') as terms_file:
            terms_file.write(terms_content)
        # create clusters
        subprocess.run([self._tool_path, '--text', self._terms_path, '--c', str(self._clusters)], capture_output=True)
        # move resulting cluster file to output destination
        cluster_path = f'{self._terms_name}-c{self._clusters}-p1.out/paths'
        shutil.copy(cluster_path, self._output_path)
