from app.data.batch import Batch
from app.data.batch_utils import sequences2tensor, sequences2lengths
from app.data.converters.utils import action_string_dis2gen
from app.tasks.evaluate.evaluator import Evaluator
from math import exp, log
from operator import itemgetter
import torch

class GenerativeEvaluator(Evaluator):
    """
    Importance sampling.
    """

    def __init__(self, device, model, action_converter, token_converter, tag_converter, non_terminal_converter):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type non_terminal_converter: app.data.converters.non_terminal.NonTerminalConverter
        """
        super().__init__(model, action_converter, token_converter, tag_converter, non_terminal_converter)
        self.device = device

    def evaluate_predictions(self, tokens, unknownified_tokens, tags, predictions):
        """
        :type tokens: list of str
        :type unknownified_tokens: list of str
        :type tags: list of str
        :type predictions: list of object
        :rtype: list of app.data.actions.action.Action, (float, float, int)
        """
        self.model.eval()
        actions = self.predictions2actions(unknownified_tokens, predictions)
        batch = self.create_batch(tokens, unknownified_tokens, tags, actions)
        generative_log_likelihoods = self.evaluate_batch(batch)
        discriminative_log_likelihoods = self.predictions2log_likelihoods(predictions)
        best_prediction, best_log_likelihood = self.get_best_prediction(unknownified_tokens, predictions, generative_log_likelihoods)
        weights = [exp(g_log_lik - d_log_lik) for g_log_lik, d_log_lik in zip(generative_log_likelihoods, discriminative_log_likelihoods)]
        tokens_likelihood = sum(weights) / len(weights)
        tokens_length = len(tokens)
        return best_prediction, (best_log_likelihood, tokens_likelihood, tokens_length)

    def get_predicted_log_likelihoods(self, evaluations):
        """
        :type evaluations: list of (float, float, int)
        """
        return list(map(itemgetter(0), evaluations))

    def get_extra_evaluation_stats(self, evaluations):
        """
        :rtype: list of (str, str)
        """
        likelihoods = []
        perplexities = []
        for _, tokens_likelihood, tokens_length in evaluations:
            tokens_log_likelihood = log(tokens_likelihood)
            perplexity = exp(- tokens_log_likelihood / tokens_length)
            likelihoods.append(tokens_likelihood)
            perplexities.append(perplexity)
        likelihood = sum(likelihoods) / len(likelihoods)
        perplexity = sum(perplexities) / len(perplexities)
        return [
            ('Sentence mean likelihood', f'{likelihood:0.8f}'),
            ('Sentence mean perplexity', f'{perplexity:0.8f}'),
        ]

    def predictions2actions(self, tokens, predictions):
        """
        :type tokens: list of str
        :type predictions: list of object
        :rtype: list of list of app.data.actions.action.Action
        """
        batch_actions = []
        for prediction in predictions:
            action_strings = prediction['actions']
            actions = action_string_dis2gen(self.action_converter, tokens, action_strings)
            batch_actions.append(actions)
        return batch_actions

    def create_batch(self, tokens, unknownified_tokens, tags, actions):
        """
        :type tokens: list of str
        :type unknownified_tokens: list of str
        :type tags: list of tags
        :type actions: list of list of app.data.actions.action.Action
        """
        batch_size = len(actions)
        actions_tensor = sequences2tensor(self.device, self.action_converter.action2integer, actions)
        actions_lengths = sequences2lengths(self.device, actions)
        tokens_tensor = self.tokens2tensor(batch_size, tokens)
        tokens_lengths = torch.tensor([len(tokens)] * batch_size, device=self.device, dtype=torch.long)
        unknownified_tokens_tensor = self.tokens2tensor(batch_size, unknownified_tokens)
        singletons_tensor = self.get_singletons_tensor(batch_size, tokens)
        tags_tensor = self.tags2tensor(batch_size, tags)
        return Batch(
            actions_tensor, actions_lengths, actions,
            tokens_tensor, tokens_lengths, [tokens] * batch_size,
            unknownified_tokens_tensor, [unknownified_tokens] * batch_size,
            singletons_tensor,
            tags_tensor, [tags] * batch_size,
        )

    def get_singletons_tensor(self, batch_size, tokens):
        singletons = list(map(self.token_converter.is_singleton, tokens))
        singletons = [[singleton] * batch_size for singleton in singletons]
        tensor = torch.tensor(singletons, device=self.device, dtype=torch.bool)
        return tensor

    def tokens2tensor(self, batch_size, tokens):
        integers = list(map(self.token_converter.token2integer, tokens))
        integers = [[token] * batch_size for token in integers]
        tensor = torch.tensor(integers, device=self.device, dtype=torch.long)
        return tensor

    def tags2tensor(self, batch_size, tags):
        integers = list(map(self.tag_converter.tag2integer, tags))
        integers = [[tag] * batch_size for tag in integers]
        tensor = torch.tensor(integers, device=self.device, dtype=torch.long)
        return tensor

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of float
        """
        batch_log_probs, _ = self.model.batch_log_likelihood(batch)
        indices = batch.actions.tensor.unsqueeze(dim=2)
        selected_log_probs = torch.gather(batch_log_probs, 2, indices).view(batch.max_actions_length, batch.size)
        selected_log_probs = [selected_log_probs[:length, i] for i, length in enumerate(batch.actions.lengths)]
        log_likelihoods = [self.log_likelihood(log_probs) for log_probs in selected_log_probs]
        return log_likelihoods

    def log_likelihood(self, log_probs):
        return log_probs.sum().cpu().item()

    def predictions2log_likelihoods(self, predictions):
        """
        :type predictions: list of object
        :rtype: list of float
        """
        return list(map(itemgetter('log_likelihood'), predictions))

    def get_best_prediction(self, tokens, predictions, log_likelihoods):
        best_prediction = None
        best_log_likelihood = None
        for prediction, log_likelihood in zip(predictions, log_likelihoods):
            if best_log_likelihood is None or best_log_likelihood < log_likelihood:
                best_prediction = prediction['actions']
                best_log_likelihood = log_likelihood
        best_prediction = action_string_dis2gen(self.action_converter, tokens, best_prediction)
        return best_prediction, best_log_likelihood
