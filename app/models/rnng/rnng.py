from app.constants import (
    ACTION_GENERATE_INDEX, ACTION_NON_TERMINAL_INDEX, ACTION_REDUCE_INDEX, ACTION_SHIFT_INDEX,
    START_ACTION_INDEX, START_TOKEN_INDEX
)
from app.models.model import Model
from app.models.rnng.actions import call_action, ActionArgs, ActionOutputs, ActionFunctions, ActionEmbeddings, ActionLogProbs
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
        reverse_tokens,
        action_converter,
    ):
        """
        :type device: torch.device
        :type generative: bool
        :type action_embedding: torch.Embedding
        :type token_embedding: torch.Embedding
        :type non_terminal_embedding: torch.Embedding
        :type non_terminal_compose_embedding: torch.Embedding
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type representation: app.representations.representation.Representation
        :type representation_size: int
        :type composer: app.composers.composer.Composer
        :type token_distribution: app.distributions.distribution.Distribution
        :type non_terminal_count: int
        :type action_set: app.data.action_set.ActionSet
        :type threads: int
        :type reverse_tokens: bool
        :type action_converter: app.data.converters.action.ActionConverter
        """
        super().__init__()
        self._device = device
        self._generative = generative
        self._threads = threads
        self._reverse_tokens = reverse_tokens
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
        self._action_embeddings = ActionEmbeddings(self._non_terminal_embedding, self._non_terminal_compose_embedding)
        self._action_functions = ActionFunctions(self._representation2non_terminal_logits, self._logits2log_prob, composer, token_distribution)
        self._action_converter = action_converter

        # TODO: initialize

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        # initialize action history and token buffer with start action and start terminal, respectively
        start_action = torch.tensor([START_ACTION_INDEX] * batch.size, device=self._device, dtype=torch.long).view(1, batch.size)
        start_action_embedding = self._action_embedding(start_action)
        start_token = torch.tensor([START_TOKEN_INDEX] * batch.size, device=self._device, dtype=torch.long).view(1, batch.size)
        start_token_embedding = self._token_embedding(start_token)
        actions_embedding = self._action_embedding(batch.actions.tensor)
        actions_embedding = torch.cat((start_action_embedding, actions_embedding), dim=0)
        tokens_embedding = self._token_embedding(batch.tokens.tensor)
        tokens_embedding = torch.cat((start_token_embedding, tokens_embedding), dim=0)
        if self._reverse_tokens:
            tokens_embedding = tokens_embedding.flip([0])

        # add to action history and token buffer
        action_history = self._action_history.new()
        token_buffer = self._token_buffer.new()
        action_history.add(actions_embedding)
        token_buffer.add(tokens_embedding)

        # stack operations
        jobs_args = []
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            job_args = (
                action_history, token_buffer, actions_embedding, tokens_embedding, element.index,
                element.actions.max_length, element.actions.length, element.actions.actions,
                element.tokens.max_length, element.tokens.length
            )
            jobs_args.append(job_args)
        if 1 < self._threads:
            get_log_probs = delayed(self._log_probs)
            log_probs_list = Parallel(n_jobs=self._threads, backend='threading')(get_log_probs(*job_args) for job_args in jobs_args)
        else:
            log_probs_list = list(map(lambda job_args: self._log_probs(*job_args), jobs_args))
        log_probs = torch.stack(log_probs_list, dim=1)

        return log_probs

    def tree_log_probs(self, tokens_tensor, actions_tensor, actions):
        """
        Compute log probs of each action in a tree.

        :type tokens_tensor: torch.Tensor
        :type actions_tensor: torch.Tensor
        :type actions: list of app.data.actions.action.Action
        :rtype: torch.Tensor
        """
        # initialize action history and token buffer with start action and start terminal, respectively
        start_action = torch.tensor([START_ACTION_INDEX], device=self._device, dtype=torch.long).view(1, 1)
        start_action_embedding = self._action_embedding(start_action)
        start_token = torch.tensor([START_TOKEN_INDEX], device=self._device, dtype=torch.long).view(1, 1)
        start_token_embedding = self._token_embedding(start_token)
        actions_embedding = self._action_embedding(actions_tensor)
        actions_embedding = torch.cat((start_action_embedding, actions_embedding), dim=0)
        tokens_embedding = self._token_embedding(tokens_tensor)
        tokens_embedding = torch.cat((start_token_embedding, tokens_embedding), dim=0)
        if self._reverse_tokens:
            tokens_embedding = tokens_embedding.flip([0])

        # add to action history and token buffer
        action_history = self._action_history.new()
        token_buffer = self._token_buffer.new()
        action_history.add(actions_embedding)
        token_buffer.add(tokens_embedding)

        batch_index = 0
        actions_max_length = len(actions)
        actions_length = len(actions)
        tokens_max_length = len(tokens_tensor)
        tokens_length = len(tokens_tensor)
        return self._log_probs(
            action_history, token_buffer, actions_embedding, tokens_embedding,
            batch_index, actions_max_length, actions_length, actions, tokens_max_length, tokens_length
        )

    def initial_state(self, tokens):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :returns: initial state
        :rtype: app.models.rnng.state.RNNGState
        """
        # initialize embeddings
        start_action = torch.tensor([[START_ACTION_INDEX]], device=self._device, dtype=torch.long)
        start_action_embedding = self._action_embedding(start_action)
        start_token = torch.tensor([[START_TOKEN_INDEX]], device=self._device, dtype=torch.long)
        start_token_embedding = self._token_embedding(start_token)
        tokens_embedding = self._token_embedding(tokens)
        tokens_embedding = torch.cat((start_token_embedding, tokens_embedding), dim=0)
        if self._reverse_tokens:
            tokens_embedding = tokens_embedding.flip([0])

        # initialize structures
        stack = self._stack.new()
        stack.push(start_action_embedding)
        action_history = self._action_history.new()
        action_history.add(start_action_embedding)
        token_buffer = self._token_buffer.new()
        token_buffer.add(tokens_embedding)

        tokens_length = tokens.size(0)
        token_index = 1 if not self._reverse_tokens else tokens_length
        token_counter = 0
        open_non_terminals_count = 0
        action_index = 0
        return RNNGState(
            stack, action_history, token_buffer, tokens_embedding, tokens_length,
            token_index, token_counter, open_non_terminals_count, action_index
        )

    def next_state(self, state, action):
        """
        Advance state of the model to the next state.

        :param state: model specific previous state
        :type state: app.models.rnng.state.RNNGState
        :type action: app.data.actions.action.Action
        :rtype: app.models.rnng.state.RNNGState
        """
        valid_actions, action2index = self._action_set.valid_actions(state.tokens_length, state.token_counter, state.stack, state.open_non_terminals_count)
        assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
        action_tensor = action.index_as_tensor().view(1, 1)
        action_embedding = self._action_embedding(action_tensor)
        state.action_history.add(action_embedding)
        action_outputs = ActionOutputs(state.open_non_terminals_count, state.token_index, state.token_counter)
        action_args = ActionArgs(
            self._action_embeddings, self._action_functions, state.stack, None, action_outputs,
            state.tokens_embedding, 0, state.tokens_length, action
        )
        _, open_non_terminals_count, token_index, token_counter = call_action(action.type(), action_args)
        return state.next(token_index, token_counter, open_non_terminals_count)

    def next_action_log_probs(self, state, posterior_scaling=1.0):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.rnng.state.RNNGState
        :rtype: torch.Tensor, list of int
        """
        representation = self._representation(state.action_history, state.stack, state.token_buffer, state.action_index, state.token_index, 0)
        valid_base_actions, action2index = self._action_set.valid_actions(state.tokens_length, state.token_counter, state.stack, state.open_non_terminals_count)
        index2action_index = []
        singleton_offset = self._action_converter.get_singleton_offset()
        index2action_index.extend([singleton_offset + index for index in valid_base_actions])
        # base log probabilities
        logits_base = self._representation2logits(representation)
        logits_base_valid = logits_base[:, :, valid_base_actions]
        log_prob_base_valid = self._logits2log_prob(posterior_scaling * logits_base_valid).view((-1,))
        # log_probs should not contain class actions: generate, non-terminal
        log_probs = None
        if ACTION_REDUCE_INDEX in valid_base_actions:
            log_probs = self._create_or_concat(log_probs, log_prob_base_valid[action2index[ACTION_REDUCE_INDEX]].view(1))
        if ACTION_SHIFT_INDEX in valid_base_actions:
            log_probs = self._create_or_concat(log_probs, log_prob_base_valid[action2index[ACTION_SHIFT_INDEX]].view(1))
        # token log probabilities for generative model
        if self._generative and ACTION_GENERATE_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_GENERATE_INDEX)
            token_log_probs, token_index2action_index = self._token_distribution.log_probs(representation, posterior_scaling=posterior_scaling)
            log_probs = self._create_or_concat(log_probs, token_log_probs)
            index2action_index.extend(token_index2action_index)
        # non-terminal log probabilities
        if ACTION_NON_TERMINAL_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_NON_TERMINAL_INDEX)
            non_terminal_logits = self._representation2non_terminal_logits(representation)
            base_non_terminal_log_prob = log_prob_base_valid[action2index[ACTION_NON_TERMINAL_INDEX]]
            conditional_non_terminal_log_probs = self._logits2log_prob(posterior_scaling * non_terminal_logits).view((-1,))
            non_terminal_log_probs = base_non_terminal_log_prob + conditional_non_terminal_log_probs
            log_probs = self._create_or_concat(log_probs, non_terminal_log_probs)
            non_terminal_offset = self._action_converter.get_non_terminal_offset()
            non_terminal2action_index = [non_terminal_offset + index for index in range(self._non_terminal_count)]
            index2action_index.extend(non_terminal2action_index)
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

    def _log_probs(self,
        action_history, token_buffer, actions_embedding, tokens_embedding,
        batch_index, actions_max_length, actions_length, actions, tokens_max_length, tokens_length
    ):
        """
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type actions_embedding: torch.Tensor
        :type tokens_embedding: torch.Tensor
        :type batch_index: int
        :type actions_max_length: int
        :type actions_length: int
        :type actions: list of app.data.actions.action.Action
        :type tokens_max_length: int
        :type tokens_length: int
        """
        stack = self._stack.new()
        # first action is always the start-action, use this to initialize stack
        start_action_embedding = actions_embedding[0:1, batch_index:batch_index+1, :]
        stack.push(start_action_embedding, None)

        token_index = 1 if not self._reverse_tokens else tokens_max_length
        token_counter = 0
        open_non_terminals_count = 0
        log_probs = torch.zeros((actions_max_length,), dtype=torch.float, device=self._device, requires_grad=False)
        for action_index in range(actions_length):
            action = actions[action_index]
            representation = self._representation(action_history, stack, token_buffer, action_index, token_index, batch_index)
            valid_actions, action2index = self._action_set.valid_actions(tokens_length, token_counter, stack, open_non_terminals_count)
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            logits_base = self._representation2logits(representation)
            logits_base_valid = logits_base[:, :, valid_actions]
            log_prob_base_valid = self._logits2log_prob(logits_base_valid)

            # get log probability of action
            action_log_probs = ActionLogProbs(representation, log_prob_base_valid, action2index)
            action_outputs = ActionOutputs(open_non_terminals_count, token_index, token_counter)
            action_args = ActionArgs(
                self._action_embeddings, self._action_functions, stack, action_log_probs, action_outputs,
                tokens_embedding, batch_index, tokens_length, action
            )
            action_log_prob, open_non_terminals_count, token_index, token_counter = call_action(action.type(), action_args)
            log_probs[action_index] = action_log_prob

        return log_probs

    def _create_or_concat(self, tensor1, tensor2, dim=0):
        if tensor1 is None:
            return tensor2
        else:
            return torch.cat((tensor1, tensor2), dim=dim)

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
