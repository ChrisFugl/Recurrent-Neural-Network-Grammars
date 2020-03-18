from app.constants import ACTION_EMBEDDING_OFFSET, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.models.model import Model
from app.models.parallel_rnng.preprocess_batch import preprocess_batch
import torch
from torch import nn

# TODO: remove time
import time

class ParallelRNNG(Model):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        """
        super().__init__()
        self._device = device
        self._action_converter, self._token_converter, self._tag_converter = converters
        action_size, token_size, rnn_input_size, rnn_size = sizes
        self._action_count = self._action_converter.count() - ACTION_EMBEDDING_OFFSET
        self._rnn_input_size = rnn_input_size

        self._action_embedding, self._token_embedding, self._nt_embedding, self._nt_compose_embedding = embeddings
        self._action_history, self._token_buffer, self._stack = structures
        self._representation = representation
        self._representation2logits = nn.Linear(in_features=rnn_size, out_features=self._action_count, bias=True)
        self._composer = composer
        self._logits2log_prob = nn.LogSoftmax(dim=2)

        start_action_embedding = torch.FloatTensor(1, 1, action_size).uniform_(-1, 1)
        self._start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        start_stack_embedding = torch.FloatTensor(1, 1, rnn_input_size).uniform_(-1, 1)
        self._start_stack_embedding = nn.Parameter(start_stack_embedding, requires_grad=True)

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        # Groundtruth actions should correspond to the set of available actions that
        # the model can predict. Padding should therefore be excluded from the set of
        # groundtruth actions and be replaced by some dummy action. This replacement
        # is feasible since the loss masks out padded actions.
        groundtruth_actions = torch.clamp(batch.actions.tensor, min=ACTION_EMBEDDING_OFFSET) - ACTION_EMBEDDING_OFFSET
        history, stack, token_buffer = self._initialize_structures(batch.tokens.tensor, batch.tags.tensor)
        # time_pre_start = time.time()
        preprocessed_batch = preprocess_batch(self._device, batch)
        # time_pre_stop = time.time()
        # print(f'preprocess batch time = {time_pre_stop-time_pre_start:0.4f}')
        output_log_probs = torch.zeros((batch.max_actions_length, batch.size), device=self._device, dtype=torch.float)
        for action_index in range(batch.max_actions_length):
            # time_start = time.time()
            representation = self._get_representation(history, stack, token_buffer)
            # TODO: should valid actions be considered?
            output_log_probs[action_index] = self._get_log_probs(batch.size, groundtruth_actions[action_index], representation)
            stack, token_buffer = self._do_actions(batch, preprocessed_batch, stack, token_buffer, action_index)
            action_embedding = self._action_embedding(batch.actions.tensor)
            history = self._action_history.push(history, action_embedding)
            # time_stop = time.time()
            # print(f'action time = {time_stop-time_start:0.4f}')
        return output_log_probs

    def tree_log_probs(self, tokens_tensor, tags_tensor, actions_tensor, actions, actions_max_length=None):
        """
        Compute log probs of each action in a tree.

        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type actions_tensor: torch.Tensor
        :type actions: list of app.data.actions.action.Action
        :type actions_max_length: int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('not implemented yet')

    def initial_state(self, tokens, tags):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :returns: initial state
        :rtype: app.models.parallel_rnng.state.State
        """
        raise NotImplementedError('not implemented yet')

    def next_state(self, state, action):
        """
        Advance state of the model to the next state.

        :param state: model specific previous state
        :type state: app.models.parallel_rnng.state.State
        :type action: app.data.actions.action.Action
        :rtype: app.models.parallel_rnng.state.State
        """
        raise NotImplementedError('not implemented yet')

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.parallel_rnng.state.State
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor, list of int
        """
        raise NotImplementedError('not implemented yet')

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

    def _initialize_structures(self, tokens, tags):
        batch_size = tokens.size(1)
        push_all_op = self._push_op(batch_size)
        history = self._initialize_action_history(batch_size)
        stack = self._initialize_stack(batch_size, push_all_op)
        token_buffer = self._initialize_token_buffer(tokens, tags)
        return history, stack, token_buffer

    def _initialize_action_history(self, batch_size):
        start_action_embedding = self._batch_one_element_tensor(self._start_action_embedding, batch_size)
        history = self._action_history.initialize(batch_size)
        history = self._action_history.push(history, start_action_embedding)
        return history

    def _initialize_stack(self, batch_size, push_all_op):
        start_stack_embedding = self._batch_one_element_tensor(self._start_stack_embedding, batch_size)
        stack = self._stack.initialize(batch_size)
        stack = self._stack.hold_or_push(stack, start_stack_embedding, push_all_op)
        return stack

    def _initialize_token_buffer(self, tokens_tensor, tags_tensor):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        raise NotImplementedError('must be implemented by subclass')

    def _get_word_embedding(self, preprocessed, token_action_indices):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type token_action_indices: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def _update_token_buffer(self, batch_size, token_action_indices, token_buffer, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type token_buffer: app.models.parallel_rnng.stack_lstm.Stack
        :type word_embeddings: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        raise NotImplementedError('must be implemented by subclass')

    def _batch_one_element_tensor(self, tensor, batch_size):
        hidden_size = tensor.size(2)
        return tensor.repeat(1, batch_size, 1).view(1, batch_size, hidden_size)

    def _hold_op(self, batch_size):
        return torch.zeros((batch_size,), device=self._device, dtype=torch.long)

    def _pop_op(self, batch_size):
        tensor = torch.empty((batch_size,), device=self._device, dtype=torch.long)
        tensor.fill_(-1)
        return tensor

    def _push_op(self, batch_size):
        return torch.ones((batch_size,), device=self._device, dtype=torch.long)

    def _get_representation(self, history, stack, token_buffer):
        """
        :type history: app.models.parallel_rnng.history_lstm.History
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :type token_buffer: app.models.parallel_rnng.stack_lstm.Stack
        """
        history_embedding, history_lengths = self._action_history.contents(history)
        stack_embedding, stack_lengths = self._stack.contents(stack)
        token_buffer_embedding, token_buffer_lengths = self._token_buffer.contents(token_buffer)
        return self._representation(
            history_embedding, history_lengths,
            stack_embedding, stack_lengths,
            token_buffer_embedding, token_buffer_lengths
        )

    def _get_log_probs(self, batch_size, actions, representation):
        """
        :type batch_size: int
        :type actions: torch.Tensor
        :type representation: torch.Tensor
        :rtype: torch.Tensor
        """
        logits = self._representation2logits(representation)
        log_probs = self._logits2log_prob(logits).squeeze()
        indicies = actions.view(batch_size, 1)
        action_log_probs = torch.gather(log_probs, 1, indicies).squeeze()
        return action_log_probs

    def _do_actions(self, batch, preprocessed_batch, stack, token_buffer, action_index):
        """
        :type batch: app.data.batch.Batch
        :type preprocessed_batch: list of app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :type token_buffer: app.models.parallel_rnng.stack_lstm.Stack
        :type action_index: int
        """
        preprocessed = preprocessed_batch[action_index]
        token_action_indices = self._get_indices_for_action(preprocessed.actions_indices, self._is_token_action)
        reduce_action_indices = self._get_indices_for_action(preprocessed.actions_indices, self._is_reduce_action)
        non_terminal_action_indices = self._get_indices_for_action(preprocessed.actions_indices, self._is_non_terminal_action)
        non_pad_indices = token_action_indices + reduce_action_indices + non_terminal_action_indices
        stack_input = torch.zeros((1, batch.size, self._rnn_input_size), device=self._device, dtype=torch.float)
        # NT
        if len(non_terminal_action_indices) != 0:
            nt_indices = preprocessed.non_terminal_index[non_terminal_action_indices]
            nt_embeddings = self._nt_embedding(nt_indices)
            stack_input[:, non_terminal_action_indices] = nt_embeddings
        # SHIFT or GEN
        if len(token_action_indices) != 0:
            word_embeddings = self._get_word_embedding(preprocessed, token_action_indices)
            stack_input[:, token_action_indices] = word_embeddings
            token_buffer = self._update_token_buffer(batch.size, token_action_indices, token_buffer, word_embeddings)
        # REDUCE
        if len(reduce_action_indices) != 0:
            children = []
            non_zero_indices = preprocessed.number_of_children != 0
            max_reduce_children = torch.max(preprocessed.number_of_children)
            for child_index in range(max_reduce_children):
                reduce_children_op = self._hold_op(batch.size)
                reduce_children_op[child_index < preprocessed.number_of_children] = -1
                stack, output = self._stack.hold_or_pop(stack, reduce_children_op)
                children.append(output[non_zero_indices])
            children.reverse()
            children = torch.stack(children, dim=0)
            compose_nt_index = preprocessed.compose_non_terminal_index[non_zero_indices]
            reduce_nt_embeddings = self._nt_compose_embedding(compose_nt_index).unsqueeze(dim=0)
            composed = self._composer(reduce_nt_embeddings, children, preprocessed.number_of_children[non_zero_indices])
            stack_input[:, non_zero_indices] = composed[0]
        stack_op = self._hold_op(batch.size)
        stack_op[non_pad_indices] = 1
        stack = self._stack.hold_or_push(stack, stack_input, stack_op)
        return stack, token_buffer

    def _is_token_action(self, action):
        type = action.type()
        return type == ACTION_SHIFT_TYPE or type == ACTION_GENERATE_TYPE

    def _is_reduce_action(self, action):
        return action.type() == ACTION_REDUCE_TYPE

    def _is_non_terminal_action(self, action):
        return action.type() == ACTION_NON_TERMINAL_TYPE

    def _get_indices_for_action(self, actions_indices, condition):
        indices = [index for index, action in actions_indices if condition(action)]
        indices.sort()
        return indices
