from app.data.loaders.loader import Loader
from app.data.preprocessing.oracles import (
    get_actions_from_oracle,
    get_terms_from_oracle,
    get_trees_from_oracle,
    get_unknownified_terms_from_oracle,
    read_oracle
)
import hydra
import os

class OracleLoader(Loader):
    """
    Assumes that data directory contains the following files:
    * train.oracle
    * val.oracle
    * test.oracle
    """

    def __init__(self, data_dir):
        """
        :type data_dir: str
        """
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        self._train_path = os.path.join(absolute_data_dir, 'train.oracle')
        self._val_path = os.path.join(absolute_data_dir, 'val.oracle')
        self._test_path = os.path.join(absolute_data_dir, 'test.oracle')

    def load_train(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens
        """
        return self._load(self._train_path)

    def load_val(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens
        """
        return self._load(self._val_path)

    def load_test(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens
        """
        return self._load(self._test_path)

    def _load(self, path):
        oracle = read_oracle(path)
        trees = get_trees_from_oracle(oracle)
        actions = get_actions_from_oracle(oracle)
        terms = get_terms_from_oracle(oracle)
        unknownified_terms = get_unknownified_terms_from_oracle(oracle)
        return trees, actions, terms, unknownified_terms
