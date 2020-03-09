from app.clustering.clustering import Clustering
from app.constants import BROWN_CLUSTERING_TOOL_PATH
import hydra
import os
import shutil
import subprocess

class BrownClustering(Clustering):

    def __init__(self, loader, output_dir, clusters):
        """
        :type loader: app.data.loaders.loader.Loader
        :type output_dir: str
        :type clusters: int
        """
        super().__init__()
        absolute_output_dir = hydra.utils.to_absolute_path(output_dir)
        output_filename = f'brown_c{clusters}.cluster'
        self._output_path = os.path.join(absolute_output_dir, output_filename)
        self._loader = loader
        self._clusters = clusters
        self._tool_path = hydra.utils.to_absolute_path(BROWN_CLUSTERING_TOOL_PATH)
        self._terms_name = f'terms'
        self._terms_path = f'{self._terms_name}.txt'

    def cluster(self):
        # get terms from input oracle file
        _, _, _, terms_tokenized, _ = self._loader.load_train()
        terms = list(map(lambda tokens: ' '.join(tokens), terms_tokenized))
        terms_content = '\n'.join(terms)
        with open(self._terms_path, 'w') as terms_file:
            terms_file.write(terms_content)
        # create clusters
        subprocess.run([self._tool_path, '--text', self._terms_path, '--c', str(self._clusters)], capture_output=True)
        # move resulting cluster file to output destination
        cluster_path = f'{self._terms_name}-c{self._clusters}-p1.out/paths'
        shutil.copy(cluster_path, self._output_path)
