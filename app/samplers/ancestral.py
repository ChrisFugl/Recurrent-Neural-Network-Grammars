from app.data.preprocessing.unknowns import fine_grained_unknownifier
from app.samplers.sampler import Sampler
import torch
from torch.distributions import Categorical

class AncestralSampler(Sampler):

    def __init__(self, device, model, iterator, action_converter, token_converter, posterior_scaling, samples):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type model: torch.model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type posterior_scaling: float
        :type samples: int
        """
        super().__init__()
        self._device = device
        self._model = model
        self._iterator = iterator
        self._action_converter = action_converter
        self._token_converter = token_converter
        self._posterior_scaling = posterior_scaling
        self._samples = samples

    def evaluate(self):
        """
        :returns: trees and log probability of each tree
        :rtype: list of list of app.data.actions.action.Action, list of float
        """
        gold_trees = []
        predicted_trees = []
        predicted_log_probs = []
        for batch in self._iterator:
            for batch_index in range(batch.size):
                element = batch.get(batch_index)
                gold_tree, predicted_tree, predicted_log_prob = self._evaluate_tree(element)
                gold_trees.append(gold_tree)
                predicted_trees.append(predicted_tree)
                predicted_log_probs.append(predicted_log_prob)
        return gold_trees, predicted_trees, predicted_log_probs

    def sample(self, tokens):
        """
        :type tokens: list of str
        :rtype: list of app.data.actions.action.Action
        """
        tokens_tensor = self._tokens2tensor(tokens)
        return self._sample_from_tokens_tensor(tokens_tensor)

    def _evaluate_tree(self, tree):
        """
        :type tree: app.data.batch.BatchElement
        """
        self._model.eval()
        tokens_tensor = tree.tokens.tensor[:tree.tokens.length, :]
        gold_tree = tree.actions.actions
        best_predicted_tree = None
        best_predicted_tree_log_prob = None
        for _ in range(self._samples):
            predicted_tree = self._sample_from_tokens_tensor(tokens_tensor)
            predicted_tree_tensor = self._actions2tensor(predicted_tree)
            predicted_tree_log_probs = self._model.tree_log_probs(tokens_tensor, predicted_tree_tensor, predicted_tree)
            predicted_tree_log_prob = predicted_tree_log_probs.sum()
            if best_predicted_tree_log_prob is None or best_predicted_tree_log_prob < predicted_tree_log_prob:
                best_predicted_tree = predicted_tree
                best_predicted_tree_log_prob = predicted_tree_log_prob
        return gold_tree, best_predicted_tree, best_predicted_tree_log_prob.cpu().item()

    def _sample_from_tokens_tensor(self, tokens):
        tokens_length = len(tokens)
        actions = []
        state = self._model.initial_state(tokens)
        while not self._is_finished_sampling(actions, tokens_length):
            log_probs, index2action_index = self._model.next_action_log_probs(state, posterior_scaling=self._posterior_scaling)
            distribution = Categorical(logits=log_probs)
            sample = index2action_index[distribution.sample()]
            action = self._action_converter.integer2action(self._device, sample)
            actions.append(action)
            state = self._model.next_state(state, action)
        return actions

    def _actions2tensor(self, actions):
        action_integers = [self._action_converter.action2integer(action) for action in actions]
        actions_tensor = torch.tensor(action_integers, device=self._device, dtype=torch.long)
        return actions_tensor.reshape((len(actions), 1))

    def _tokens2tensor(self, tokens):
        known_tokens = self._token_converter.tokens()
        unknownified_tokens = [fine_grained_unknownifier(known_tokens, (None, token)) for token in tokens]
        token_integers = [self._token_converter.token2integer(token) for token in unknownified_tokens]
        tokens_tensor = torch.tensor(token_integers, device=self._device, dtype=torch.long)
        return tokens_tensor.reshape((len(tokens), 1))

    def __str__(self):
        return f'Ancestral(posterior_scaling={self._posterior_scaling})'
