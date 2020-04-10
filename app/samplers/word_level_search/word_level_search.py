from app.constants import ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE
from app.data.actions.reduce import ReduceAction
from app.data.batch import Batch
from app.data.batch_utils import sequences2lengths, sequences2tensor
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
        super().__init__(device, action_converter, log=log)
        self.model = model
        self.iterator = iterator
        self.beam_size = beam_size
        self.samples = samples
        self.fast_track = fast_track

    def get_iterator(self):
        return self.iterator, self.iterator.size()

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :type batch_index: int
        :rtype: list of app.samplers.sample.Sample
        """
        self.model.eval()
        samples = []
        for i in range(batch.size):
            sample = self.evaluate_element(batch, i)
            samples.append(sample)
        return samples

    def evaluate_element(self, batch, batch_index):
        """
        :type batch: app.data.batch.Batch
        :type batch_index: int
        :rtype: app.samplers.sample.Sample
        """
        element = batch.get(batch_index)
        tokens = element.tokens.tokens
        tags = element.tags.tags
        tokens_tensor = element.tokens.tensor[:element.tokens.length, :]
        tags_tensor = element.tags.tensor[:element.tags.length, :]
        samples = self.search(tokens, tokens_tensor, tags, tags_tensor, element.tokens.length.reshape(1))
        best_actions = None
        best_log_prob = None
        sample_probs = []
        for actions, log_prob in samples:
            sample_probs.append(exp(log_prob))
            if best_log_prob is None or best_log_prob < log_prob:
                best_actions = actions
                best_log_prob = log_prob
        p_tensor = sequences2tensor(self.device, self.action_converter.action2integer, [best_actions])
        p_lengths = sequences2lengths(self.device, [best_actions])
        p_batch = Batch(
            p_tensor, p_lengths, [best_actions],
            element.tokens.tensor, element.tokens.length.unsqueeze(dim=0), [tokens],
            element.tags.tensor, element.tags.length.unsqueeze(dim=0), [tags],
        )
        p_log_probs = self.model.batch_log_likelihood(p_batch)
        p_log_prob, p_probs = self.batch_stats(p_log_probs, p_batch.actions.lengths)
        g_actions = element.actions.actions
        g_batch = Batch(
            element.actions.tensor, element.actions.length.unsqueeze(dim=0), [g_actions],
            element.tokens.tensor, element.tokens.length.unsqueeze(dim=0), [tokens],
            element.tags.tensor, element.tags.length.unsqueeze(dim=0), [tags],
        )
        g_log_probs = self.model.batch_log_likelihood(g_batch)
        g_log_prob, g_probs = self.batch_stats(g_log_probs, g_batch.actions.lengths)
        tokens_prob = sum(sample_probs)
        return Sample(g_actions, tokens, tags, g_log_prob, g_probs, best_actions, p_log_prob, p_probs, tokens_prob)

    def search(self, tokens, tokens_tensor, tags, tags_tensor, tokens_length):
        """
        :type tokens: list of str
        :type tokens_tensor: torch.Tensor
        :type tags: list of str
        :type tags_tensor: torch.Tensor
        :type tokens_length: torch.Tensor
        :rtype: list of (list of app.data.actions.action.Action, float)
        """
        sentence_length = len(tokens)
        initial_state = self.model.initial_state(tokens_tensor, tags_tensor, tokens_length)
        initial_hypothesis = Hypothesis(self.device, initial_state, None, 0)
        hypothesises = [initial_hypothesis]
        for token_index in range(sentence_length + 1):
            has_generated_all_tokens = token_index == sentence_length
            has_tokens_left = not has_generated_all_tokens
            token = None if has_generated_all_tokens else tokens[token_index]
            bucket = hypothesises
            next_bucket = []
            fast_tracks = []
            while len(fast_tracks) + len(next_bucket) < self.samples:
                bucket = self.get_successors(token, bucket, has_tokens_left)
                if has_tokens_left:
                    fast_tracked, fast_tracked_indices = self.get_fast_tracked(bucket)
                    fast_tracks.extend(fast_tracked)
                    bucket = [hypothesis for i, hypothesis in enumerate(bucket) if not i in fast_tracked_indices]
                bucket, _ = self.top_hypothesises(bucket, self.beam_size)
                if has_tokens_left:
                    lexical, lexical_indices = self.get_lexical_hypothesises(bucket)
                    next_bucket.extend(lexical)
                    bucket = [hypothesis for i, hypothesis in enumerate(bucket) if not i in lexical_indices]
                else:
                    next_bucket = bucket
            hypothesises, _ = self.top_hypothesises(next_bucket, self.samples - len(fast_tracks))
            hypothesises.extend(fast_tracks)
        samples = [self.hypothesis2actions(tokens_tensor, tokens, tags_tensor, tags, hypothesis) for hypothesis in hypothesises]
        return samples

    def get_successors(self, token, hypothesises, include_nt):
        """
        :type token: str
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :type include_nt: bool
        :rtype: list of app.samplers.word_level_search.Hypothesis
        """
        successors = []
        for hypothesis in hypothesises:
            hypothesis_successors = hypothesis.successors(self.model, self.action_converter, token, include_nt)
            successors.extend(hypothesis_successors)
        return successors

    def get_fast_tracked(self, hypothesises):
        """
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :rtype: list of app.samplers.word_level_search.Hypothesis, list of int
        """
        lexical, lexical_indices = self.get_lexical_hypothesises(hypothesises)
        fast_tracked, fast_tracked_indices = self.top_hypothesises(lexical, self.fast_track)
        indices = [lexical_indices[index] for index in fast_tracked_indices]
        return fast_tracked, indices

    def get_lexical_hypothesises(self, hypothesises):
        """
        :type hypothesises: list of app.samplers.word_level_search.Hypothesis
        :rtype: list of app.samplers.word_level_search.Hypothesis, list of int
        """
        lexical = []
        indices = []
        for index, hypothesis in enumerate(hypothesises):
            if self.is_lexical_hypothesis(hypothesis):
                lexical.append(hypothesis)
                indices.append(index)
        return lexical, indices

    def top_hypothesises(self, hypothesises, size):
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

    def is_lexical_hypothesis(self, hypothesis):
        """
        :type hypothesises: app.samplers.word_level_search.Hypothesis
        :rtype: bool
        """
        return hypothesis.action.type() == ACTION_GENERATE_TYPE

    def hypothesis2actions(self, tokens_tensor, tokens, tags_tensor, tags, hypothesis):
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
        action_indices = [self.action_converter.action2integer(action) for action in actions]
        actions_tensor = torch.tensor(action_indices, device=self.device, dtype=torch.long).view(len(actions), 1)
        actions_length = sequences2lengths(self.device, [actions])
        tokens_length = sequences2lengths(self.device, [tokens])
        batch = Batch(
            actions_tensor, actions_length, [actions],
            tokens_tensor, tokens_length, [tokens],
            tags_tensor, tokens_length, tags,
        )
        log_prob = self.model.batch_log_likelihood(batch).sum()
        return actions, log_prob

    def __str__(self):
        return f'WordLevelSearch(beam_size={self.beam_size}, samples={self.samples}, fast_track={self.fast_track})'
