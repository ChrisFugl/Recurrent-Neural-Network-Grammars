from app.data.preprocessing.oracles import brackets2oracle
from app.data.preprocessing.terminals import get_terminals
from app.tasks.task import Task
import hydra
import os

class CreateOracleTask(Task):

    def __init__(self, loader, data_dir, generative, fine_grained_unknowns):
        """
        :type loader: app.data.loaders.loader.Loader
        :type data_dir: str
        :type generative: bool
        :type fine_grained_unknowns: bool
        """
        super().__init__()
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        save_dir_name = 'generative' if generative else 'discriminative'
        self._save_dir_path = os.path.join(absolute_data_dir, save_dir_name)
        self._train_save_path = os.path.join(self._save_dir_path, 'train.oracle')
        self._val_save_path = os.path.join(self._save_dir_path, 'val.oracle')
        self._test_save_path = os.path.join(self._save_dir_path, 'test.oracle')
        self._loader = loader
        self._generative = generative
        self._fine_grained_unknowns = fine_grained_unknowns

    def run(self):
        os.makedirs(self._save_dir_path, exist_ok=True)
        train_bracket_lines = self._loader.load_train()
        val_bracket_lines = self._loader.load_val()
        test_bracket_lines = self._loader.load_test()
        terminals, terminals_counter = get_terminals(train_bracket_lines)
        train_oracle = brackets2oracle(train_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        val_oracle = brackets2oracle(val_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        test_oracle = brackets2oracle(test_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
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
