from app.constants import (
    ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE,
    EVALB_TOOL_PATH, EVALB_PARAMS_PATH
)
from app.tasks.task import Task
import hydra
import logging
from math import exp
import os
import re
import subprocess
import time

class EvaluateTask(Task):

    def __init__(self, device, data, sampler):
        """
        :type device: torch.device
        :type data: str
        :type sampler: app.samplers.sampler.Sampler
        """
        super().__init__()
        self._logger = logging.getLogger('evaluate')
        self._device = device
        self._data = data
        self._sampler = sampler
        self._tool_path = hydra.utils.to_absolute_path(EVALB_TOOL_PATH)
        self._params_path = hydra.utils.to_absolute_path(EVALB_PARAMS_PATH)

    def run(self):
        time_start = time.time()
        self._logger.info('Starting evaluation')
        self._logger.info(f'Saving output to {os.getcwd()}')
        self._logger.info(f'Using device: {self._device}')
        self._logger.info(f'Data: {self._data}')
        self._logger.info(f'Sampler: {self._sampler}')
        self._evaluate()
        time_stop = time.time()
        self._logger.info('Finished evaluation')
        self._logger.info(f'Time taken: {time_stop - time_start:0.2f} s')

    def _evaluate(self):
        gold_trees, gold_tokens, gold_tags, predicted_trees, gold_log_probs, action_log_probs, token_log_probs = self._sampler.evaluate()
        gold_path = self._save_trees_to_brackets('gold', 'trees.gld', gold_trees, gold_tokens, gold_tags)
        predicted_path = self._save_trees_to_brackets('predicted', 'trees.tst', predicted_trees, gold_tokens, gold_tags)
        self._log_stats('Gold trees', self._get_stats(gold_log_probs, gold_trees))
        self._log_stats('Predicted trees', self._get_stats(action_log_probs, gold_trees))
        self._log_stats('Tokens', self._get_stats(token_log_probs, gold_tokens))
        self._bracket_scores(gold_path, predicted_path)

    def _get_path(self, filename):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, filename)
        return path

    def _save_trees_to_brackets(self, name, filename, trees, trees_tokens, trees_tags):
        path = self._get_path(filename)
        brackets = self._trees2brackets(trees, trees_tokens, trees_tags)
        content = '\n'.join(brackets)
        with open(path, 'w') as file:
            file.write(content)
        self._logger.info(f'Saved {name} trees at {path}')
        return path

    def _trees2brackets(self, trees, trees_tokens, trees_tags):
        brackets = []
        for tree, tokens, tags in zip(trees, trees_tokens, trees_tags):
            tree_brackets = []
            token_index = 0
            for action in tree:
                type = action.type()
                if type == ACTION_NON_TERMINAL_TYPE:
                    tree_brackets.append(f' ({action.argument}')
                elif type == ACTION_REDUCE_TYPE:
                    tree_brackets.append(')')
                elif type == ACTION_SHIFT_TYPE:
                    tree_brackets.append(f' ({tags[token_index]} {tokens[token_index]})')
                    token_index += 1
                elif type == ACTION_GENERATE_TYPE:
                    tree_brackets.append(f' ({tags[token_index]} {action.argument})')
                else:
                    raise Exception(f'Unknown action: {type}')
            tree_brackets_string = ''.join(tree_brackets).strip()
            brackets.append(tree_brackets_string)
        return brackets

    def _get_stats(self, log_probs, items):
        if log_probs is None:
            return None
        count = sum(map(len, items))
        log_likelihood = sum(log_probs)
        likelihood = exp(log_likelihood)
        perplexity = exp(- log_likelihood / count)
        return log_likelihood, likelihood, perplexity

    def _log_stats(self, name, stats):
        if stats is not None:
            log_likelihood, likelihood, perplexity = stats
            self._logger.info(f'{name} log likelihood = {log_likelihood:0.8f}')
            self._logger.info(f'{name} likelihood     = {likelihood:0.8f}')
            self._logger.info(f'{name} perplexity     = {perplexity:0.8f}')

    def _bracket_scores(self, gold_path, predicted_path):
        process = subprocess.run([self._tool_path, '-p', self._params_path, gold_path, predicted_path], capture_output=True)
        output = str(process.stdout, 'utf-8')
        lines = output.split('\n')
        self._logger.info(f'Recall    = {self._get_recall(lines)}')
        self._logger.info(f'Precision = {self._get_precision(lines)}')
        self._logger.info(f'F1        = {self._get_f1(lines)}')
        filename = 'evalb.txt'
        path = self._get_path(filename)
        with open(filename, 'w') as file:
            file.write(output)
        self._logger.info(f'Saved evalb output to {path}')

    def _get_recall(self, lines):
        return self._get_bracket_score('Bracketing Recall', lines)

    def _get_precision(self, lines):
        return self._get_bracket_score('Bracketing Precision', lines)

    def _get_f1(self, lines):
        return self._get_bracket_score('Bracketing FMeasure', lines)

    def _get_bracket_score(self, name, lines):
        expression = re.compile(f'{name}\\s*=\\s*(\\d+(\\.\\d*))')
        for line in lines:
            match = expression.match(line)
            if match is not None:
                return match.group(1)
        raise Exception(f'Could not read "{name}" from evalb output.')
