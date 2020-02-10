from app.data.preprocessing import brackets2oracle, get_terminals
from app.tasks.task import Task
import hydra
import os

class CreateOracleTask(Task):

    def __init__(self, train_path, train_save_path, val_path, val_save_path, test_path, test_save_path, generative, fine_grained_unknowns):
        """
        :type train_path: str
        :type train_save_path: str
        :type val_path: str
        :type val_save_path: str
        :type test_path: str
        :type test_save_path: str
        :type generative: bool
        :type fine_grained_unknowns: bool
        """
        super().__init__()
        self._train_path = hydra.utils.to_absolute_path(train_path)
        self._train_save_path = hydra.utils.to_absolute_path(train_save_path)
        self._val_path = hydra.utils.to_absolute_path(val_path)
        self._val_save_path = hydra.utils.to_absolute_path(val_save_path)
        self._test_path = hydra.utils.to_absolute_path(test_path)
        self._test_save_path = hydra.utils.to_absolute_path(test_save_path)
        self._generative = generative
        self._fine_grained_unknowns = fine_grained_unknowns

    def run(self):
        train_bracket_lines = self._load(self._train_path)
        val_bracket_lines = self._load(self._val_path)
        test_bracket_lines = self._load(self._test_path)
        terminals, terminals_counter = get_terminals(train_bracket_lines)
        train_oracle = brackets2oracle(train_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        val_oracle = brackets2oracle(val_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        test_oracle = brackets2oracle(test_bracket_lines, terminals, self._generative, self._fine_grained_unknowns)
        self._save(train_oracle, self._train_save_path)
        self._save(val_oracle, self._val_save_path)
        self._save(test_oracle, self._test_save_path)

    def _load(self, path):
        with open(path, 'r') as file:
            return file.readlines()

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
