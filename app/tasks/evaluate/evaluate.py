from app.tasks.evaluate.discriminative import DiscriminativeEvaluator
from app.tasks.evaluate.generative import GenerativeEvaluator
from app.scores import scores_from_samples
from app.tasks.task import Task
from app.data.converters.utils import action_string_dis2gen
import logging
from math import exp
import os
import time
import torch

EVALB_FILENAME = 'evalb.txt'

class EvaluateTask(Task):

    def __init__(self, device, model, generative, action_converter, token_converter, tag_converter, non_terminal_converter, samples):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type generative: bool
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type non_terminal_converter: app.data.converters.non_terminal.NonTerminalConverter
        :type samples: object
        """
        super().__init__()
        self.logger = logging.getLogger('evaluate')
        self.device = device
        self.generative = generative
        self.samples = samples
        self.action_converter = action_converter
        self.model = model
        if generative:
            self.evaluator = GenerativeEvaluator(device, model, action_converter, token_converter, tag_converter, non_terminal_converter)
        else:
            self.evaluator = DiscriminativeEvaluator(model, action_converter, token_converter, tag_converter, non_terminal_converter)

    @torch.no_grad()
    def run(self):
        time_start = time.time()
        self.logger.info('Starting evaluation')
        self.logger.info(f'Saving output to {os.getcwd()}')
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Model:\n{self.model}')
        self.evaluate()
        time_stop = time.time()
        self.logger.info('Finished evaluation')
        self.logger.info(f'Time taken: {time_stop - time_start:0.2f} s')

    def evaluate(self):
        tokens, tags, gold_actions, gold_log_likelihoods, predicted_actions, evaluations = self.load_samples()
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
        perplexities = []
        for i, log_likelihood in enumerate(log_likelihoods):
            try:
                perplexity = exp(- log_likelihood / len(actions[i]))
                likelihoods.append(exp(log_likelihood))
                perplexities.append(perplexity)
            except Exception:
                self.logger.warning(f'Failed to compute likelihood/perplexity of "{name}" tree at index {i}')
        log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        likelihood = sum(likelihoods) / len(likelihoods)
        perplexity = sum(perplexities) / len(perplexities)
        self.logger.info(f'{name} mean log likelihood = {log_likelihood:0.8f}')
        self.logger.info(f'{name} mean likelihood     = {likelihood:0.8f}')
        self.logger.info(f'{name} mean perplexity     = {perplexity:0.8f}')

    def string2action(self, tokens, actions):
        if self.generative:
            return action_string_dis2gen(self.action_converter, tokens, actions)
        else:
            return list(map(self.action_converter.string2action, actions))
