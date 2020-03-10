from app.constants import ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.data.actions.generate import GenerateAction
from app.data.actions.non_terminal import NonTerminalAction
from app.data.actions.reduce import ReduceAction
from app.samplers.sample import Sample
from app.samplers.sampler import Sampler
from math import log
from torch.distributions import Categorical

class ImportanceSampler(Sampler):

    def __init__(self,
        device, posterior_scaling, samples,
        model_dis, iterator_dis, action_converter_dis, token_converter_dis,
        model_gen, iterator_gen, action_converter_gen, token_converter_gen
    ):
        """
        :type device: torch.device
        :type posterior_scaling: float
        :type samples: int
        :type action_converter: app.data.converters.action.ActionConverter
        :type model_dis: torch.model
        :type model_gen: torch.model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        """
        super().__init__()
        self._device = device
        self._posterior_scaling = posterior_scaling
        self._samples = samples
        self._model_dis = model_dis
        self._model_gen = model_gen
        self._iterator_dis = iterator_dis
        self._iterator_gen = iterator_gen
        self._action_converter_dis = action_converter_dis
        self._action_converter_gen = action_converter_gen
        self._token_converter_dis = token_converter_dis
        self._token_converter_gen = token_converter_gen

    def evaluate_element(self, batch, batch_index):
        """
        :type batch: app.data.batch.Batch, app.data.batch.Batch
        :type batch_index: int
        :rtype: app.samplers.sample.Sample
        """
        self._model_dis.eval()
        self._model_gen.eval()
        batch_dis, batch_gen = batch
        element_dis = batch_dis.get(batch_index)
        element_gen = batch_gen.get(batch_index)
        tokens_tensor_dis = element_dis.tokens.tensor[:element_dis.tokens.length, :]
        tokens_tensor_gen = element_gen.tokens.tensor[:element_gen.tokens.length, :]
        weights = []
        best_tree = None
        best_log_prob = None
        for _ in range(self._samples):
            tree_dis = self._sample_from_tokens_tensor(tokens_tensor_dis)
            tree_dis_tensor = self._actions2tensor(self._action_converter_dis, tree_dis)
            tree_gen = self._discriminative2generative(tree_dis, element_gen.tokens.tokens)
            tree_gen_tensor = self._actions2tensor(self._action_converter_gen, tree_gen)
            log_prob_dis = self._get_tree_log_prob(self._model_dis, tokens_tensor_dis, tree_dis_tensor, tree_dis)
            log_prob_gen = self._get_tree_log_prob(self._model_gen, tokens_tensor_gen, tree_gen_tensor, tree_gen)
            weight = (log_prob_gen - log_prob_dis).exp().cpu().item()
            weights.append(weight)
            if best_log_prob is None or best_log_prob < log_prob_gen:
                best_tree = tree_gen
                best_log_prob = log_prob_gen
        gold_tree = element_gen.actions.actions
        gold_tree_tensor = element_gen.actions.tensor[:element_gen.actions.length, :]
        gold_log_probs = self._model_gen.tree_log_probs(tokens_tensor_gen, gold_tree_tensor, gold_tree)
        gold_log_prob = gold_log_probs.sum()
        best_log_prob = best_log_prob.cpu().item()
        tokens_log_prob = log(sum(weights) / len(weights))
        return Sample(gold_tree, element_gen.tokens.tokens, element_gen.tags, gold_log_prob, best_tree, best_log_prob, tokens_log_prob)

    def get_batch_size(self, batch):
        """
        :rtype: int
        """
        batch_dis, _ = batch
        return batch_dis.size

    def get_iterator(self):
        count = self._iterator_dis.size()
        iterator = zip(self._iterator_dis, self._iterator_gen)
        return iterator, count

    def sample(self, tokens):
        """
        :type tokens: list of str
        :rtype: list of app.data.actions.action.Action
        """
        # TODO: consider multiple samples
        self._model_dis.eval()
        self._model_gen.eval()
        tokens_tensor = self._tokens2tensor(self._token_converter_gen, tokens)
        return self._sample_from_tokens_tensor(tokens_tensor)

    def _sample_from_tokens_tensor(self, tokens):
        self._model_dis.eval()
        tokens_length = len(tokens)
        actions = []
        state = self._model_dis.initial_state(tokens)
        while not self._is_finished_sampling(actions, tokens_length):
            log_probs, index2action_index = self._model_dis.next_action_log_probs(state, posterior_scaling=self._posterior_scaling)
            distribution = Categorical(logits=log_probs)
            sample = index2action_index[distribution.sample()]
            action = self._action_converter_dis.integer2action(self._device, sample)
            actions.append(action)
            state = self._model_dis.next_state(state, action)
        return actions

    def _discriminative2generative(self, actions_dis, tokens):
        non_terminal_offset = self._action_converter_gen.get_non_terminal_offset()
        terminal_offset = self._action_converter_gen.get_terminal_offset()
        token_index = 0
        actions_gen = []
        for action_dis in actions_dis:
            type = action_dis.type()
            if type == ACTION_REDUCE_TYPE:
                action_gen = ReduceAction(self._device)
            elif type == ACTION_NON_TERMINAL_TYPE:
                argument = action_dis.argument
                argument_index = self._action_converter_gen.string2integer(f'NT({argument})') - non_terminal_offset
                action_gen = NonTerminalAction(self._device, argument, argument_index)
            else:
                argument = tokens[token_index]
                argument_index = self._action_converter_gen.string2integer(f'GEN({argument})') - terminal_offset
                action_gen = GenerateAction(self._device, argument, argument_index)
                token_index += 1
            actions_gen.append(action_gen)
        return actions_gen

    def _get_tree_log_prob(self, model, tokens_tensor, tree_tensor, tree):
        log_probs = model.tree_log_probs(tokens_tensor, tree_tensor, tree)
        log_prob = log_probs.sum()
        return log_prob

    def __str__(self):
        return f'Importance(posterior_scaling={self._posterior_scaling}, samples={self._samples})'