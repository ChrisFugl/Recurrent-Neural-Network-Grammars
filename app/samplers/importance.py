from app.constants import ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.data.actions.generate import GenerateAction
from app.data.actions.non_terminal import NonTerminalAction
from app.data.actions.reduce import ReduceAction
from app.data.batch import Batch
from app.data.batch_utils import sequences2tensor
from app.samplers.sample import Sample
from app.samplers.sampler import Sampler
from math import exp
from torch.distributions import Categorical

class ImportanceSampler(Sampler):

    def __init__(self,
        device, posterior_scaling, samples,
        model_dis, iterator_dis, action_converter_dis,
        model_gen, iterator_gen, action_converter_gen,
        log=True
    ):
        """
        :type device: torch.device
        :type posterior_scaling: float
        :type samples: int
        :type model_dis: torch.model
        :type model_gen: torch.model
        :type iterator_dis: app.data.iterators.iterator.Iterator
        :type iterator_gen: app.data.iterators.iterator.Iterator
        :type action_converter_dis: app.data.converters.action.ActionConverter
        :type action_converter_gen: app.data.converters.action.ActionConverter
        :type log: bool
        """
        super().__init__(device, action_converter_dis, log=log)
        self.posterior_scaling = posterior_scaling
        self.samples = samples
        self.model_dis = model_dis
        self.model_gen = model_gen
        self.iterator_dis = iterator_dis
        self.iterator_gen = iterator_gen
        self.action_converter_dis = action_converter_dis
        self.action_converter_gen = action_converter_gen

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch, app.data.batch.Batch
        :rtype: app.samplers.sample.Sample
        """
        self.model_dis.eval()
        self.model_gen.eval()
        batch_dis, batch_gen = batch
        weights = [[] for _ in range(batch_dis.size)]
        best_actions = [None] * batch_dis.size
        best_probs = [None] * batch_dis.size
        best_log_prob = [None] * batch_dis.size
        for _ in range(self.samples):
            pred_batch_dis = self.sample(batch_dis)
            pred_actions_gen = [self.dis2gen(pred_batch_dis.actions.actions[i], batch_gen.tokens.tokens[i]) for i in range(batch_dis.size)]
            pred_tensor_gen = sequences2tensor(self.device, self.action_converter_gen.action2integer, pred_actions_gen)
            pred_batch_gen = Batch(
                pred_tensor_gen, pred_batch_dis.actions.lengths, pred_actions_gen,
                batch_gen.tokens.tensor, batch_gen.tokens.lengths, batch_gen.tokens.tokens,
                batch_gen.tags.tensor, batch_gen.tags.lengths, batch_gen.tags.tags,
            )
            pred_log_probs_dis = self.model_dis.batch_log_likelihood(pred_batch_dis)
            pred_log_prob_dis, pred_probs_dis = self.batch_stats(pred_log_probs_dis, pred_batch_dis.actions.lengths)
            pred_log_probs_gen = self.model_gen.batch_log_likelihood(pred_batch_gen)
            pred_log_prob_gen, pred_probs_gen = self.batch_stats(pred_log_probs_gen, pred_batch_gen.actions.lengths)
            for i in range(batch_gen.size):
                weight = exp(pred_log_prob_gen[i] - pred_log_prob_dis[i])
                weights[i].append(weight)
                if best_log_prob[i] is None or best_log_prob[i] < pred_log_prob_gen[i]:
                    best_actions[i] = pred_actions_gen[i]
                    best_probs[i] = pred_probs_gen[i]
                    best_log_prob[i] = pred_log_prob_gen[i]
        gold_log_probs = self.model_gen.batch_log_likelihood(batch_gen)
        gold_log_prob, gold_probs = self.batch_stats(gold_log_probs, batch_gen.actions.lengths)
        samples = []
        for i in range(batch_gen.size):
            g_actions = batch_gen.actions.actions[i]
            g_tokens = batch_gen.tokens.tokens[i]
            g_tags = batch_gen.tags.tags[i]
            g_log_prob = gold_log_prob[i]
            g_probs = gold_probs[i]
            p_actions = best_actions[i]
            p_log_prob = best_log_prob[i]
            p_probs = best_probs[i]
            tokens_prob = sum(weights[i]) / len(weights[i])
            sample = Sample(g_actions, g_tokens, g_tags, g_log_prob, g_probs, p_actions, p_log_prob, p_probs, tokens_prob)
            samples.append(sample)
        return samples

    def get_iterator(self):
        count = self.iterator_dis.size()
        iterator = zip(self.iterator_dis, self.iterator_gen)
        return iterator, count

    def get_initial_state(self, batch):
        return self.model_dis.initial_state(batch.tokens.tensor, batch.tags.tensor, batch.tokens.lengths)

    def get_next_log_probs(self, state):
        return self.model_dis.next_action_log_probs(state, posterior_scaling=self.posterior_scaling)

    def get_next_state(self, state, actions):
        return self.model_dis.next_state(state, actions)

    def sample_actions(self, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: torch.Tensor
        """
        distribution = Categorical(logits=log_probs)
        return distribution.sample()

    def dis2gen(self, actions_dis, tokens):
        token_index = 0
        actions_gen = []
        for action_dis in actions_dis:
            type = action_dis.type()
            if type == ACTION_REDUCE_TYPE:
                action_gen = ReduceAction()
            elif type == ACTION_NON_TERMINAL_TYPE:
                argument = action_dis.argument
                action_gen = NonTerminalAction(argument)
            else:
                argument = tokens[token_index]
                action_gen = GenerateAction(argument)
                token_index += 1
            actions_gen.append(action_gen)
        return actions_gen

    def __str__(self):
        return f'Importance(posterior_scaling={self.posterior_scaling}, samples={self.samples})'
