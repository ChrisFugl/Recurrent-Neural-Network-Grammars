from app.tasks.evaluate.discriminative import DiscriminativeEvaluator
from app.tasks.evaluate.generative import GenerativeEvaluator
from app.scores import scores_from_samples
from app.tasks.task import Task
from app.data.converters.utils import action_string_dis2gen
import logging
from math import exp
import numpy as np
import os
import pandas as pd
import time
import torch

EVALB_FILENAME = 'evalb.txt'
LENGTH_FILENAME = 'scores_by_length.csv'

class EvaluateTask(Task):

    def __init__(self, device, model, generative, action_converter, token_converter, tag_converter, non_terminal_converter, samples, max_batch_size):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type generative: bool
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type non_terminal_converter: app.data.converters.non_terminal.NonTerminalConverter
        :type samples: object
        :type max_batch_size: int
        """
        super().__init__()
        self.logger = logging.getLogger('evaluate')
        self.device = device
        self.generative = generative
        self.samples = samples
        self.action_converter = action_converter
        self.model = model
        self.max_batch_size = max_batch_size
        if generative:
            self.evaluator = GenerativeEvaluator(device, model, action_converter, token_converter, tag_converter, non_terminal_converter, max_batch_size)
        else:
            self.evaluator = DiscriminativeEvaluator(model, action_converter, token_converter, tag_converter, non_terminal_converter)

    @torch.no_grad()
    def run(self):
        time_start = time.time()
        self.logger.info('Starting evaluation')
        self.logger.info(f'Saving output to {os.getcwd()}')
        self.logger.info(f'Using device: {self.device}')
        if self.max_batch_size is not None:
            self.logger.info(f'Max. batch size: {self.max_batch_size}')
        self.logger.info(f'Model:\n{self.model}')
        self.evaluate()
        time_stop = time.time()
        self.logger.info('Finished evaluation')
        self.logger.info(f'Time taken: {time_stop - time_start:0.2f} s')

    def evaluate(self):
        tokens, tags, gold_actions, gold_log_likelihoods, predicted_actions, evaluations = self.load_samples()
        self.evaluate_by_sentence_length(tokens, tags, gold_actions, predicted_actions)
        predicted_log_likelihoods = self.evaluator.get_predicted_log_likelihoods(evaluations)
        scores, evalb_output, gold_path, predicted_path = scores_from_samples(tokens, tags, gold_actions, predicted_actions)
        self.logger.info(f'Saved gold trees at {gold_path}')
        self.logger.info(f'Saved predicted trees at {predicted_path}')
        self.log_prob_stats('Gold trees', gold_actions, gold_log_likelihoods)
        self.log_prob_stats('Predicted trees', predicted_actions, predicted_log_likelihoods)
        extra_stats = self.evaluator.get_extra_evaluation_stats(evaluations)
        for name, value in extra_stats:
            self.logger.info(f'{name}: {value}')
        f1, precision, recall = scores
        self.logger.info(f'Recall    = {recall}')
        self.logger.info(f'Precision = {precision}')
        self.logger.info(f'F1        = {f1}')
        path = self.get_path(EVALB_FILENAME)
        with open(path, 'w') as file:
            file.write(evalb_output)
        self.logger.info(f'Saved evalb output to {path}')

    def evaluate_by_sentence_length(self, tokens, tags, gold_actions, predicted_actions):
        lengths = list(set(map(len, tokens)))
        lengths.sort()
        n_features = 4
        data = np.ndarray((len(lengths), n_features))
        for i, length in enumerate(lengths):
            group = self.group_by_length(tokens, tags, gold_actions, predicted_actions, length)
            tokens_group, tags_group, gold_actions_group, predicted_actions_group = group
            scores, _, _, _ = scores_from_samples(tokens_group, tags_group, gold_actions_group, predicted_actions_group)
            data[i] = (length, *scores)
        columns = ['length', 'f1', 'precision', 'recall']
        dtype = {'length': int, 'f1': float, 'precision': float, 'recall': float}
        dataframe = pd.DataFrame(data, columns=columns)
        dataframe = dataframe.astype(dtype)
        path = self.get_path(LENGTH_FILENAME)
        dataframe.to_csv(path, index=False)
        self.logger.info(f'Saved evaluation scores by sentence length at {path}')

    def group_by_length(self, tokens, tags, gold_actions, predicted_actions, length):
        tokens_group = []
        tags_group = []
        gold_actions_group = []
        predicted_actions_group = []
        for i in range(len(tokens)):
            if len(tokens[i]) == length:
                tokens_group.append(tokens[i])
                tags_group.append(tags[i])
                gold_actions_group.append(gold_actions[i])
                predicted_actions_group.append(predicted_actions[i])
        return tokens_group, tags_group, gold_actions_group, predicted_actions_group

    def get_path(self, filename):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, filename)
        return path

    def load_samples(self):
        tokens = []
        tags = []
        gold_actions = []
        gold_log_likelihoods = []
        predicted_actions = []
        evaluations = []
        for tree in self.samples['trees']:
            gold = tree['gold']
            prediction, evaluation = self.evaluator.evaluate_predictions(
                gold['tokens'],
                gold['unknownified_tokens'],
                gold['tags'],
                tree['predictions']
            )
            tokens.append(gold['tokens'])
            tags.append(gold['tags'])
            gold_actions.append(self.string2action(gold['tokens'], gold['actions']))
            gold_log_likelihoods.append(gold['log_likelihood'])
            predicted_actions.append(prediction)
            evaluations.append(evaluation)
        return tokens, tags, gold_actions, gold_log_likelihoods, predicted_actions, evaluations

    def log_prob_stats(self, name, actions, log_likelihoods):
        """
        :type name: str
        :type actions: list of list of app.data.actions.action.Action
        :type log_likelihoods: list of float
        """
        likelihoods = []
        n_actions = 0
        for i, log_likelihood in enumerate(log_likelihoods):
            try:
                n_actions += len(actions[i])
                likelihoods.append(exp(log_likelihood))
            except Exception:
                self.logger.warning(f'Failed to compute likelihood of "{name}" tree at index {i}')
        log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        likelihood = sum(likelihoods) / len(likelihoods)
        perplexity = exp(- sum(log_likelihoods) / n_actions)
        self.logger.info(f'{name} mean log likelihood = {log_likelihood:0.8f}')
        self.logger.info(f'{name} mean likelihood     = {likelihood:0.8f}')
        self.logger.info(f'{name} perplexity     = {perplexity:0.8f}')

    def string2action(self, tokens, actions):
        if self.generative:
            return action_string_dis2gen(self.action_converter, tokens, actions)
        else:
            return list(map(self.action_converter.string2action, actions))
