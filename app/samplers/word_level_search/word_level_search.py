from app.constants import ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE
from app.data.actions.reduce import ReduceAction
from app.samplers.sample import Sample
from app.samplers.sampler import Sampler
from app.samplers.word_level_search.hypothesis import Hypothesis
from heapq import nlargest
from math import exp
import torch

class WordLevelSearchSampler(Sampler):

    def __init__(self, device, model, iterator, action_converter, beam_size, samples, fast_track, log=True):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type beam_size: int
        :type samples: int
        :type fast_track: int
        :type log: bool
        """
        super().__init__(log=log)
        self._device = device
        self._model = model
        self._iterator = iterator
        self._action_converter = action_converter
        self._beam_size = beam_size
        self._samples = samples
        self._fast_track = fast_track

    def get_iterator(self):
        return self._iterator, self._iterator.size()

    def evaluate_element(self, batch, batch_index):
        """
        :type batch: app.data.batch.Batch
        :type batch_index: int
        :rtype: app.samplers.sample.Sample
        """
        self._model.eval()
        element = batch.get(batch_index)
        tokens = element.tokens.tokens
        tokens_tensor = element.tokens.tensor[:element.tokens.length, :]
        tags_tensor = element.tags.tensor[:element.tags.length, :]
        samples = self._search(tokens, tokens_tensor, tags_tensor)
        best_tree = None
        best_log_prob = None
        sample_probs = []
        for tree, log_prob in samples:
            sample_probs.append(exp(log_prob))
            if best_log_prob is None or best_log_prob < log_prob:
                best_tree = tree
                best_log_prob = log_prob
        predicted_tree_tensor = self.actions2tensor(self._action_converter, best_tree)
        predicted_tree_log_probs = self._model.tree_log_probs(tokens_tensor, tags_tensor, predicted_tree_tensor, best_tree)
        predicted_probs = [prob.cpu().item() for prob in predicted_tree_log_probs.sum(dim=1).exp()]
        gold_tree = element.actions.actions
        gold_tree_tensor = element.actions.tensor[:element.actions.length, :]
        gold_log_probs = self._model.tree_log_probs(tokens_tensor, tags_tensor, gold_tree_tensor, gold_tree)
        gold_probs = [prob.cpu().item() for prob in gold_log_probs.sum(dim=1).exp()]
        gold_log_prob = gold_log_probs.sum()
        tokens_prob = sum(sample_probs)
        return Sample(
            gold_tree, element.tokens.tokens, element.tags.tags, gold_log_prob, gold_probs,
            best_tree, best_log_prob, predicted_probs,
            tokens_prob
        )

    def _search(self, tokens, tokens_tensor, tags_tensor):
        """
        :type tokens: list of str
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :rtype: list of (list of app.data.actions.action.Action, float)
        """
        sentence_length = len(tokens)
        initial_state = self._model.initial_state(tokens_tensor, tags_tensor)
        initial_hypothesis = Hypothesis(self._device, initial_state, None, 0)
        hypothesises = [initial_hypothesis]
        for token_index in range(sentence_length + 1):
            has_generated_all_tokens = token_index == sentence_length
            has_tokens_left = not has_generated_all_tokens
            token = None if has_generated_all_tokens else tokens[token_index]
            bucket = hypothesises
            next_bucket = []
            fast_tracks = []
            while len(fast_tracks) + len(next_bucket) < self._samples:
                bucket = self._get_successors(token, bucket, has_tokens_left)
                if has_tokens_left:
                    fast_tracked, fast_tracked_indices = self._get_fast_tracked(bucket)
                    fast_tracks.extend(fast_tracked)
                    bucket = [hypothesis for i, hypothesis in enumerate(bucket) if not i in fast_tracked_indices]
                bucket, _ = self._top_hypothesises(bucket, self._beam_size)
                if has_tokens_left:
                    lexical, lexical_indices = self._get_lexical_hypothesises(bucket)
                    next_bucket.extend(lexical)
                    bucket = [hypothesis for i, hypothesis in enumerate(bucket) if not i in lexical_indices]
                else:
                    next_bucket = bucket
            hypothesises, _ = self._top_hypothesises(next_bucket, self._samples - len(fast_tracks))
            hypothesises.extend(fast_tracks)
        samples = [self._hypothesis2actions(tokens_tensor, tokens, tags_tensor, hypothesis) for hypothesis in hypothesises]
        return samples

    def _get_successors(self, token, hypothesises, include_nt):
        """
        :type token: str
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :type include_nt: bool
        :rtype: list of app.samplers.word_level_search.Hypothesis
        """
        successors = []
        for hypothesis in hypothesises:
            hypothesis_successors = hypothesis.successors(self._model, self._action_converter, token, include_nt)
            successors.extend(hypothesis_successors)
        return successors

    def _get_fast_tracked(self, hypothesises):
        """
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :rtype: list of app.samplers.word_level_search.Hypothesis, list of int
        """
        lexical, lexical_indices = self._get_lexical_hypothesises(hypothesises)
        fast_tracked, fast_tracked_indices = self._top_hypothesises(lexical, self._fast_track)
        indices = [lexical_indices[index] for index in fast_tracked_indices]
        return fast_tracked, indices

    def _get_lexical_hypothesises(self, hypothesises):
        """
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :rtype: list of app.samplers.word_level_search.Hypothesis, list of int
        """
        lexical = []
        indices = []
        for index, hypothesis in enumerate(hypothesises):
            if self._is_lexical_hypothesis(hypothesis):
                lexical.append(hypothesis)
                indices.append(index)
        return lexical, indices

    def _top_hypothesises(self, hypothesises, size):
        """
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :type size: int
        :rtype: list of app.samplers.word_level_search.Hypothesis, list of int
        """
        top_zipped = nlargest(size, enumerate(hypothesises), key=lambda pair: pair[1].log_prob)
        if len(top_zipped) == 0:
            return [], []
        indices, top = zip(*top_zipped)
        return list(top), list(indices)

    def _is_lexical_hypothesis(self, hypothesis):
        """
        :type hypothesises: app.samplers.word_level_search.Hypothesis
        :rtype: bool
        """
        return hypothesis.action.type() == ACTION_GENERATE_TYPE

    def _hypothesis2actions(self, tokens_tensor, tokens, tags_tensor, hypothesis):
        """
        A hypothesis may contain an incomplete tree,
        since it may have open non-terminals. Append
        reduce actions to solve this.

        :type tokens_tensor: torch.Tensor
        :type tokens: list of str
        :type hypothesis: app.samplers.word_level_search.Hypothesis
        :rtype: list of app.data.actions.action.Action, float
        """
        actions = hypothesis.actions()
        non_terminal_count = len(list(filter(lambda action: action.type() == ACTION_NON_TERMINAL_TYPE, actions)))
        reduce_count = len(list(filter(lambda action: action.type() == ACTION_REDUCE_TYPE, actions)))
        open_non_terminals_count = non_terminal_count - reduce_count
        while 0 < open_non_terminals_count:
            actions.append(ReduceAction())
            open_non_terminals_count -= 1
        action_indices = [self._action_converter.action2integer(action) for action in actions]
        actions_tensor = torch.tensor(action_indices, device=self._device, dtype=torch.long).view(len(actions), 1)
        log_prob = self._model.tree_log_probs(tokens_tensor, tags_tensor, actions_tensor, actions).sum()
        return actions, log_prob

    def __str__(self):
        return f'WordLevelSearch(beam_size={self._beam_size}, samples={self._samples}, fast_track={self._fast_track})'
