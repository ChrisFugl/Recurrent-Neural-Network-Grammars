from app.data.actions.non_terminal import NonTerminalAction
from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_NON_TERMINAL_INDEX, ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX,
    ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE,
    START_ACTION_INDEX, START_TOKEN_INDEX
)
from app.models.model import Model
from app.models.rnng.action_args import ActionOutputs, ActionLogProbs
from app.models.rnng.state import RNNGState
from joblib import Parallel, delayed
import torch
from torch import nn

# base actions: (REDUCE, GENERATE, NON_TERMINAL) or (REDUCE, SHIFT, NON_TERMINAL)
ACTIONS_COUNT = 3

class RNNG(Model):

    def __init__(
        self, device, generative,
        action_embedding, token_embedding,
        non_terminal_embedding, non_terminal_compose_embedding,
        action_history, token_buffer, stack,
        representation, representation_size,
        composer,
        token_distribution,
        non_terminal_count,
        action_set,
        threads,
        action_converter,
        token_converter,
    ):
        """
        :type device: torch.device
        :type generative: bool
        :type action_embedding: torch.Embedding
        :type token_embedding: torch.Embedding
        :type non_terminal_embedding: torch.Embedding
        :type non_terminal_compose_embedding: torch.Embedding
        :type action_history: app.models.rnng.stack.Stack
        :type token_buffer: app.models.rnng.stack.Stack
        :type stack: app.models.rnng.stack.Stack
        :type representation: app.representations.representation.Representation
        :type representation_size: int
        :type composer: app.composers.composer.Composer
        :type token_distribution: app.distributions.distribution.Distribution
        :type non_terminal_count: int
        :type action_set: app.data.action_set.ActionSet
        :type threads: int
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        """
        super().__init__()
        self._device = device
        self._generative = generative
        self._threads = threads
        self._action_set = action_set
        self._action_embedding = action_embedding
        self._non_terminal_embedding = non_terminal_embedding
        self._non_terminal_compose_embedding = non_terminal_compose_embedding
        self._token_embedding = token_embedding
        self._action_history = action_history
        self._token_buffer = token_buffer
        self._stack = stack
        self._representation = representation
        self._representation2logits = nn.Linear(in_features=representation_size, out_features=ACTIONS_COUNT, bias=True)
        self._composer = composer
        self._logits2log_prob = nn.LogSoftmax(dim=2)
        self._representation2non_terminal_logits = nn.Linear(in_features=representation_size, out_features=non_terminal_count, bias=True)
        if self._generative:
            self._token_distribution = token_distribution
        self._non_terminal_count = non_terminal_count
        self._action_converter = action_converter
        self._token_converter = token_converter

        self._type2action = {
            ACTION_REDUCE_TYPE: self._reduce,
            ACTION_NON_TERMINAL_TYPE: self._non_terminal,
            ACTION_SHIFT_TYPE: self._shift,
            ACTION_GENERATE_TYPE: self._generate,
        }

        # TODO: initialize

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        # stack operations
        jobs_args = []
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            tokens_tensor = element.tokens.tensor[:element.tokens.length, :]
            actions_tensor = element.actions.tensor[:element.actions.length, :]
            actions = element.actions.actions
            actions_max_length = element.actions.max_length
            job_args = (tokens_tensor, actions_tensor, actions, actions_max_length)
            jobs_args.append(job_args)
        if 1 < self._threads:
            get_log_probs = delayed(self.tree_log_probs)
            log_probs_list = Parallel(n_jobs=self._threads, backend='threading')(get_log_probs(*job_args) for job_args in jobs_args)
        else:
            log_probs_list = [self.tree_log_probs(*args) for args in jobs_args]
        log_probs = torch.stack(log_probs_list, dim=1)
        return log_probs

    def tree_log_probs(self, tokens_tensor, actions_tensor, actions, actions_max_length=None):
        """
        Compute log probs of each action in a tree.

        :type tokens_tensor: torch.Tensor
        :type actions_tensor: torch.Tensor
        :type actions: list of app.data.actions.action.Action
        :type actions_max_length: int
        :rtype: torch.Tensor
        """
        # initialize structures
        start_action_tensor = self._index2tensor(START_ACTION_INDEX)
        start_action_embedding = self._action_embedding(start_action_tensor)
        start_token_tensor = self._index2tensor(START_TOKEN_INDEX)
        start_token_embedding = self._token_embedding(start_token_tensor)
        stack_top = self._stack.push(start_action_embedding)
        action_top = self._action_history.push(start_action_embedding)
        token_top = self._token_buffer.push(start_token_embedding, data=start_token_tensor)

        actions_length = len(actions)
        tokens_length = len(tokens_tensor)
        if actions_max_length is None:
            actions_max_length = actions_length

        # discriminative model processes tokens in reverse order
        if not self._generative:
            token_embeddings = self._token_embedding(tokens_tensor)
            for token_index in range(tokens_length - 1, -1, -1):
                token = token_embeddings[token_index:token_index+1, :]
                token_tensor = tokens_tensor[token_index:token_index+1, :]
                token_top = self._token_buffer.push(token, data=token_tensor, top=token_top)

        token_counter = 0
        open_non_terminals_count = 0
        log_probs = torch.zeros((actions_max_length,), dtype=torch.float, device=self._device, requires_grad=False)
        for action_index in range(actions_length):
            action = actions[action_index]
            last_action = stack_top.data
            stack_embedding = self._stack.contents(stack_top)
            action_history_embedding = self._action_history.contents(action_top)
            token_buffer_embedding = self._token_buffer.contents(token_top)
            representation = self._representation(action_history_embedding, stack_embedding, token_buffer_embedding)
            valid_actions, action2index = self._action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            base_logits = self._representation2logits(representation)
            valid_base_logits = base_logits[:, :, valid_actions]
            valid_base_log_probs = self._logits2log_prob(valid_base_logits)
            # get log probability of action
            action_args_log_probs = ActionLogProbs(representation, valid_base_log_probs, action2index)
            action_args_outputs = ActionOutputs(stack_top, token_top, open_non_terminals_count, token_counter)
            action_outputs = self._type2action[action.type()](action_args_log_probs, action_args_outputs, action)
            action_log_prob, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
            log_probs[action_index] = action_log_prob
            action_tensor = actions_tensor[action_index:action_index+1, :]
            action_embedding = self._action_embedding(action_tensor)
            action_top = self._action_history.push(action_embedding, data=action, top=action_top)
        return log_probs

    def initial_state(self, tokens):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :returns: initial state
        :rtype: app.models.rnng.state.RNNGState
        """
        # initialize structures
        start_action_tensor = self._index2tensor(START_ACTION_INDEX)
        start_action_embedding = self._action_embedding(start_action_tensor)
        start_token_tensor = self._index2tensor(START_TOKEN_INDEX)
        start_token_embedding = self._token_embedding(start_token_tensor)
        stack_top = self._stack.push(start_action_embedding)
        action_top = self._action_history.push(start_action_embedding)
        token_top = self._token_buffer.push(start_token_embedding, data=start_token_tensor)

        # discriminative model processes tokens in reverse order
        tokens_length = tokens.size(0)
        if not self._generative:
            token_embeddings = self._token_embedding(tokens)
            for token_index in range(tokens_length - 1, -1, -1):
                token = token_embeddings[token_index:token_index+1, :]
                token_tensor = tokens[token_index:token_index+1, :]
                token_top = self._token_buffer.push(token, data=token_tensor, top=token_top)

        token_counter = 0
        open_non_terminals_count = 0
        return RNNGState(stack_top, action_top, token_top, tokens, tokens_length, open_non_terminals_count, token_counter)

    def next_state(self, state, action):
        """
        Advance state of the model to the next state.

        :param state: model specific previous state
        :type state: app.models.rnng.state.RNNGState
        :type action: app.data.actions.action.Action
        :rtype: app.models.rnng.state.RNNGState
        """
        tokens_length = state.tokens_length
        token_counter = state.token_counter
        last_action = state.stack_top.data
        open_non_terminals_count = state.open_non_terminals_count
        valid_actions, action2index = self._action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
        assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
        action_args_outputs = ActionOutputs(state.stack_top, state.token_top, open_non_terminals_count, token_counter)
        action_outputs = self._type2action[action.type()](None, action_args_outputs, action)
        _, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
        action_index = self._action_converter.action2integer(action)
        action_tensor = self._index2tensor(action_index)
        action_embedding = self._action_embedding(action_tensor)
        action_top = self._action_history.push(action_embedding, data=action, top=state.action_top)
        return state.next(stack_top, action_top, token_top, open_non_terminals_count, token_counter)

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.rnng.state.RNNGState
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor, list of int
        """
        tokens_length = state.tokens_length
        token_counter = state.token_counter
        last_action = state.stack_top.data
        open_non_terminals_count = state.open_non_terminals_count
        stack_embedding = self._stack.contents(state.stack_top)
        action_history_embedding = self._action_history.contents(state.action_top)
        token_buffer_embedding = self._token_buffer.contents(state.token_top)
        representation = self._representation(action_history_embedding, stack_embedding, token_buffer_embedding)
        valid_base_actions, action2index = self._action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
        index2action_index = []
        singleton_offset = self._action_converter.get_singleton_offset()
        index2action_index.extend([singleton_offset + index for index in valid_base_actions])
        # base log probabilities
        base_logits = self._representation2logits(representation)
        valid_base_logits = base_logits[:, :, valid_base_actions]
        valid_base_log_probs = self._logits2log_prob(posterior_scaling * valid_base_logits).view((-1,))
        # log_probs should not contain class actions: generate, non-terminal
        log_probs_list = []
        if ACTION_REDUCE_INDEX in valid_base_actions:
            log_probs_list.append(valid_base_log_probs[action2index[ACTION_REDUCE_INDEX]].view(1))
        if not self._generative and ACTION_SHIFT_INDEX in valid_base_actions:
            log_probs_list.append(valid_base_log_probs[action2index[ACTION_SHIFT_INDEX]].view(1))
        # token log probabilities for generative model
        if self._generative and ACTION_GENERATE_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_GENERATE_INDEX)
            base_generate_log_prob = valid_base_log_probs[action2index[ACTION_GENERATE_INDEX]]
            if token is None and include_gen:
                conditional_token_log_probs, token_index2action_index = self._token_distribution.log_probs(
                    representation,
                    posterior_scaling=posterior_scaling
                )
                token_log_probs = base_generate_log_prob + conditional_token_log_probs
                log_probs_list.append(token_log_probs)
                index2action_index.extend(token_index2action_index)
            if token is not None:
                token_distribution_log_prob = self._token_distribution.log_prob(representation, token, posterior_scaling=posterior_scaling).view(1)
                token_log_prob = base_generate_log_prob + token_distribution_log_prob
                log_probs_list.append(token_log_prob)
                token_action_string = f'GEN({token})'
                token_action_index = self._action_converter.string2integer(token_action_string)
                index2action_index.append(token_action_index)
        # non-terminal log probabilities
        if ACTION_NON_TERMINAL_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_NON_TERMINAL_INDEX)
            if include_nt:
                non_terminal_logits = self._representation2non_terminal_logits(representation)
                base_non_terminal_log_prob = valid_base_log_probs[action2index[ACTION_NON_TERMINAL_INDEX]]
                conditional_non_terminal_log_probs = self._logits2log_prob(posterior_scaling * non_terminal_logits).view((-1,))
                non_terminal_log_probs = base_non_terminal_log_prob + conditional_non_terminal_log_probs
                log_probs_list.append(non_terminal_log_probs)
                non_terminal_offset = self._action_converter.get_non_terminal_offset()
                non_terminal2action_index = [non_terminal_offset + index for index in range(self._non_terminal_count)]
                index2action_index.extend(non_terminal2action_index)
        log_probs = torch.cat(log_probs_list, dim=0)
        return log_probs, index2action_index

    def save(self, path):
        """
        Save model parameters.

        :type path: str
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        """
        Load model parameters from file.

        :type path: str
        """
        state_dict = torch.load(path, map_location=self._device)
        self.load_state_dict(state_dict)

    def __str__(self):
        return (
            'RNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + ('' if not self._generative else f'  token_distribution={self._token_distribution}\n')
            + ')'
        )

    def _reduce(self, log_probs, outputs, action):
        stack_top = outputs.stack_top
        children = []
        while True:
            action, state = stack_top.data, stack_top.output
            if action.type() == ACTION_NON_TERMINAL_TYPE and action.open:
                break
            stack_top = self._stack.pop(stack_top)
            children.append(state)
        children.reverse()
        children_tensor = torch.cat(children, dim=0)
        compose_action = NonTerminalAction(self._device, action.argument, action.argument_index, open=False)
        stack_top = self._stack.pop(stack_top)
        non_terminal_embedding, _ = self._get_non_terminal_embedding(self._non_terminal_compose_embedding, compose_action)
        composed = self._composer(non_terminal_embedding, children_tensor)
        stack_top = self._stack.push(composed, data=compose_action, top=stack_top)
        action_log_prob = self._get_base_log_prop(log_probs, ACTION_REDUCE_INDEX)
        open_non_terminals_count = outputs.open_non_terminals_count - 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def _non_terminal(self, log_probs, outputs, action):
        non_terminal_embedding, argument_index = self._get_non_terminal_embedding(self._non_terminal_embedding, action)
        stack_top = self._stack.push(non_terminal_embedding, data=action, top=outputs.stack_top)
        if log_probs is None:
            action_log_prob = None
        else:
            non_terminal_log_prob = self._get_base_log_prop(log_probs, ACTION_NON_TERMINAL_INDEX)
            non_terminal_logits = self._representation2non_terminal_logits(log_probs.representation)
            conditional_non_terminal_log_probs = self._logits2log_prob(non_terminal_logits)
            conditional_non_terminal_log_prob = conditional_non_terminal_log_probs[:, :, argument_index]
            action_log_prob = non_terminal_log_prob + conditional_non_terminal_log_prob
        open_non_terminals_count = outputs.open_non_terminals_count + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def _shift(self, log_probs, outputs, action):
        token_tensor = outputs.token_top.data
        token_embedding = self._token_embedding(token_tensor)
        token_top = self._token_buffer.pop(outputs.token_top)
        stack_top = self._stack.push(token_embedding, data=action, top=outputs.stack_top)
        action_log_prob = self._get_base_log_prop(log_probs, ACTION_SHIFT_INDEX)
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, token_top=token_top, token_counter=token_counter)

    def _generate(self, log_probs, outputs, action):
        token_index = self._token_converter.token2integer(action.argument)
        token_tensor = self._index2tensor(token_index)
        token_embedding = self._token_embedding(token_tensor)
        stack_top = self._stack.push(token_embedding, data=action, top=outputs.stack_top)
        token_top = self._token_buffer.push(token_embedding, data=token_tensor, top=outputs.token_top)
        if log_probs is None:
            action_log_prob = None
        else:
            generate_log_prob = self._get_base_log_prop(log_probs, ACTION_GENERATE_INDEX)
            token_log_prob = self._token_distribution.log_prob(log_probs.representation, action.argument)
            action_log_prob = generate_log_prob + token_log_prob
        token_counter = outputs.token_counter + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, token_top=token_top, token_counter=token_counter)

    def _index2tensor(self, index):
        return torch.tensor([[index]], device=self._device, dtype=torch.long)

    def _get_non_terminal_embedding(self, embeddings, action):
        argument_index = action.argument_index_as_tensor()
        non_terminal_embedding = embeddings(argument_index).unsqueeze(dim=0).unsqueeze(dim=0)
        return non_terminal_embedding, argument_index

    def _get_base_log_prop(self, log_probs, action_index):
        if log_probs is None:
            return None
        else:
            return log_probs.log_prob_base[:, :, log_probs.action2index[action_index]]
