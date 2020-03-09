from app.data.loaders.loader import Loader
from app.data.preprocessing.oracles import (
    get_actions_from_oracle,
    get_tags_from_oracle,
    get_terms_from_oracle,
    get_trees_from_oracle,
    get_unknownified_terms_from_oracle,
    read_oracle
)
import hydra
import logging
import os

class OracleLoader(Loader):
    """
    Assumes that data directory contains the following files:
    * train.oracle
    * val.oracle
    * test.oracle
    """

    def __init__(self, data_dir, name=None):
        """
        :type data_dir: str
        :type name: str
        """
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        logger_name = 'loader' if name is None else name
        self._logger = logging.getLogger(logger_name)
        self._logger.info(f'Loading data from {absolute_data_dir}')
        self._train_path = os.path.join(absolute_data_dir, 'train.oracle')
        self._val_path = os.path.join(absolute_data_dir, 'val.oracle')
        self._test_path = os.path.join(absolute_data_dir, 'test.oracle')

    def load_train(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens
        """
        data = self._load(self._train_path)
        self._logger.info('Finished loading training data')
        return data

    def load_val(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens
        """
        data = self._load(self._val_path)
        self._logger.info('Finished loading validation data')
        return data

    def load_test(self):
        """
        :rtype: list of str, list of list of str, list of list of str, list of list of str, list of list of str
        :returns: trees, actions, tokens, unknownified tokens, tags
        """
        data = self._load(self._test_path)
        self._logger.info('Finished loading test data')
        return data

    def _load(self, path):
        oracle = read_oracle(path)
        trees = get_trees_from_oracle(oracle)
        actions = get_actions_from_oracle(oracle)
        terms = get_terms_from_oracle(oracle)
        unknownified_terms = get_unknownified_terms_from_oracle(oracle)
        tags = get_tags_from_oracle(oracle)
        return trees, actions, terms, unknownified_terms, tags
