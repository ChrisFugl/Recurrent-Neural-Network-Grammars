from app.data.loaders.loader import Loader
import hydra
import os

class RawLoader(Loader):
    """
    Assumes that data directory contains the following files:
    * train.txt
    * val.txt
    * test.txt
    """

    def __init__(self, data_dir):
        """
        :type data_dir: str
        """
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        self._train_path = os.path.join(absolute_data_dir, 'train.txt')
        self._val_path = os.path.join(absolute_data_dir, 'val.txt')
        self._test_path = os.path.join(absolute_data_dir, 'test.txt')

    def load_train(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._train_path)

    def load_val(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._val_path)

    def load_test(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._test_path)

    def _load(self, path):
        with open(path, 'r') as file:
            trees = file.read().split('\n')
        return trees
