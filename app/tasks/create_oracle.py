from app.data.preprocessing.oracles import brackets2oracle
from app.data.preprocessing.terminals import get_terminals
from app.tasks.task import Task
import hydra
import logging
import os

class CreateOracleTask(Task):

    def __init__(self, loader, data_dir, oracle_type, fine_grained_unknowns):
        """
        :type loader: app.data.loaders.loader.Loader
        :type data_dir: str
        :type oracle_type: str
        :type fine_grained_unknowns: bool
        """
        super().__init__()
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        save_dir_name = oracle_type
        self._save_dir_path = os.path.join(absolute_data_dir, save_dir_name)
        self._train_save_path = os.path.join(self._save_dir_path, 'train.oracle')
        self._val_save_path = os.path.join(self._save_dir_path, 'val.oracle')
        self._test_save_path = os.path.join(self._save_dir_path, 'test.oracle')
        self._loader = loader
        self._generative = oracle_type == 'generative'
        self._fine_grained_unknowns = fine_grained_unknowns
        self._logger = logging.getLogger('create_oracle')

    def run(self):
        self._logger.info(f'Saving output to {self._save_dir_path}')
        os.makedirs(self._save_dir_path, exist_ok=True)
        train_bracket_lines = self._loader.load_train()
        self._logger.info('Finished loading training data')
        val_bracket_lines = self._loader.load_val()
        self._logger.info('Finished loading validation data')
        test_bracket_lines = self._loader.load_test()
        self._logger.info('Finished loading test data')
        terminals, terminals_counter = get_terminals(train_bracket_lines)
        train_oracle = brackets2oracle(train_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        self._logger.info('Finished converting training data')
        val_oracle = brackets2oracle(val_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        self._logger.info('Finished converting validation data')
        test_oracle = brackets2oracle(test_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        self._logger.info('Finished converting test data')
        self._save(train_oracle, self._train_save_path)
        self._save(val_oracle, self._val_save_path)
        self._save(test_oracle, self._test_save_path)

    def _save(self, oracle, path):
        self._create_parent_directories(path)
        lines = []
        brackets, actions, tokens, tokens_unknownified = oracle
        n_examples = len(brackets)
        for example_index in range(n_examples):
            lines.append(brackets[example_index])
            lines.append(' '.join(actions[example_index]))
            lines.append(' '.join(tokens[example_index]))
            lines.append(' '.join(tokens_unknownified[example_index]))
        content = '\n'.join(lines)
        with open(path, 'w') as file:
            file.write(content)

    def _create_parent_directories(self, path):
        parent_directory_path = os.path.abspath(os.path.join(path, '..'))
        os.makedirs(parent_directory_path, exist_ok=True)
