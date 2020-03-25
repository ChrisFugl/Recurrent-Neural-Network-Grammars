from app.constants import ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.models.model import Model
from app.models.parallel_rnng.preprocess_batch import preprocess_batch
import torch
from torch import nn

class ParallelRNNG(Model):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.buffer_lstm.BufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        """
        super().__init__()
        self.device = device
        self.action_converter, self.token_converter, self.tag_converter = converters
        action_size, token_size, rnn_input_size, rnn_size = sizes
        self.action_count = self.action_converter.count()
        self.rnn_input_size = rnn_input_size

        self.action_embedding, self.token_embedding, self.nt_embedding, self.nt_compose_embedding = embeddings
        self.action_history, self.token_buffer, self.stack = structures
        self.representation = representation
        self.representation2logits = nn.Linear(in_features=rnn_size, out_features=self.action_count, bias=True)
        self.composer = composer
        self.logits2log_prob = nn.LogSoftmax(dim=2)

        start_action_embedding = torch.FloatTensor(1, action_size).uniform_(-1, 1)
        self.start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        start_stack_embedding = torch.FloatTensor(1, rnn_input_size).uniform_(-1, 1)
        self.start_stack_embedding = nn.Parameter(start_stack_embedding, requires_grad=True)

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        preprocessed_batches, stack_size = preprocess_batch(self.device, batch)
        self.initialize_structures(
            batch.actions.tensor,
            batch.tokens.tensor,
            batch.tags.tensor,
            # plus one to account for start embeddings
            batch.actions.lengths + 1,
            batch.tokens.lengths + 1,
            stack_size + 1,
        )
        output_shape = (batch.max_actions_length, batch.size, self.action_count)
        output_log_probs = torch.zeros(output_shape, device=self.device, dtype=torch.float)
        for action_index in range(batch.max_actions_length):
            preprocessed_batch = preprocessed_batches[action_index]
            representation = self.get_representation()
            output_log_probs[action_index] = self.get_log_probs(representation)
            self.do_actions(batch, preprocessed_batch)
            action_tensor = batch.actions.tensor[action_index].unsqueeze(dim=0)
            action_embedding = self.action_embedding(action_tensor)
            self.action_history.push(action_embedding)
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
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def initialize_structures(self, actions, tokens, tags, action_lengths, token_lengths, stack_size):
        batch_size = tokens.size(1)
        self.initialize_action_history(batch_size, actions, action_lengths)
        self.initialize_stack(stack_size, batch_size)
        self.initialize_token_buffer(tokens, tags, token_lengths)

    def initialize_action_history(self, batch_size, actions, action_lengths):
        start_action_embedding = self.batch_one_element_tensor(self.start_action_embedding, batch_size).unsqueeze(dim=0)
        self.action_history.initialize(action_lengths)
        self.action_history.push(start_action_embedding)

    def initialize_stack(self, stack_size, batch_size):
        start_stack_embedding = self.batch_one_element_tensor(self.start_stack_embedding, batch_size)
        self.stack.initialize(stack_size, batch_size)
        push_all_op = self.push_op(batch_size)
        self.stack.hold_or_push(start_stack_embedding, push_all_op)

    def initialize_token_buffer(self, tokens_tensor, tags_tensor):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_word_embedding(self, preprocessed, token_action_indices):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        :type token_action_indices: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def update_token_buffer(self, batch_size, token_action_indices, word_embeddings):
        """
        :type batch_size: int
        :type token_action_indices: torch.Tensor
        :type word_embeddings: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def batch_one_element_tensor(self, tensor, batch_size):
        return tensor.repeat(batch_size, 1)

    def hold_op(self, batch_size):
        return torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    def pop_op(self, batch_size):
        tensor = torch.empty((batch_size,), device=self.device, dtype=torch.long)
        tensor.fill_(-1)
        return tensor

    def push_op(self, batch_size):
        return torch.ones((batch_size,), device=self.device, dtype=torch.long)

    def get_representation(self):
        history_embedding, history_lengths = self.action_history.contents()
        stack_embedding, stack_lengths = self.stack.contents()
        token_buffer_embedding, token_buffer_lengths = self.token_buffer.contents()
        return self.representation(
            history_embedding, history_lengths,
            stack_embedding, stack_lengths,
            token_buffer_embedding, token_buffer_lengths
        )

    def get_log_probs(self, representation):
        """
        :type representation: torch.Tensor
        :type forbidden_actions: torch.Tensor
        :rtype: torch.Tensor
        """
        logits = self.representation2logits(representation)
        log_probs = self.logits2log_prob(logits)
        return log_probs

    def do_actions(self, batch, preprocessed):
        """
        :type batch: app.data.batch.Batch
        :type preprocessed_batch: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        """
        token_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_token_action)
        reduce_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_reduce_action)
        non_terminal_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_non_terminal_action)
        non_pad_indices = token_action_indices + reduce_action_indices + non_terminal_action_indices
        stack_input = torch.zeros((batch.size, self.rnn_input_size), device=self.device, dtype=torch.float)
        # NT
        if len(non_terminal_action_indices) != 0:
            nt_indices = preprocessed.non_terminal_index[non_terminal_action_indices]
            nt_embeddings = self.nt_embedding(nt_indices)
            stack_input[non_terminal_action_indices] = nt_embeddings
        # SHIFT or GEN
        if len(token_action_indices) != 0:
            word_embeddings = self.get_word_embedding(preprocessed, token_action_indices)
            stack_input[token_action_indices] = word_embeddings[token_action_indices]
            self.update_token_buffer(batch.size, token_action_indices, word_embeddings)
        # REDUCE
        if len(reduce_action_indices) != 0:
            children = []
            max_reduce_children = torch.max(preprocessed.number_of_children)
            for child_index in range(max_reduce_children):
                reduce_children_op = self.hold_op(batch.size)
                reduce_children_op[child_index < preprocessed.number_of_children] = -1
                output = self.stack.hold_or_pop(reduce_children_op)
                children.append(output[reduce_action_indices])
            children.reverse()
            children = torch.stack(children, dim=0)
            # pop non-terminal from stack
            non_terminal_pop_op = self.hold_op(batch.size)
            non_terminal_pop_op[reduce_action_indices] = -1
            self.stack.hold_or_pop(non_terminal_pop_op)
            # compose and push composed constituent to stack
            compose_nt_index = preprocessed.compose_non_terminal_index[reduce_action_indices]
            reduce_nt_embeddings = self.nt_compose_embedding(compose_nt_index).unsqueeze(dim=0)
            composed = self.composer(reduce_nt_embeddings, children, preprocessed.number_of_children[reduce_action_indices])
            stack_input[reduce_action_indices] = composed[0]
        stack_op = self.hold_op(batch.size)
        stack_op[non_pad_indices] = 1
        self.stack.hold_or_push(stack_input, stack_op)

    def is_token_action(self, action):
        type = action.type()
        return type == ACTION_SHIFT_TYPE or type == ACTION_GENERATE_TYPE

    def is_reduce_action(self, action):
        return action.type() == ACTION_REDUCE_TYPE

    def is_non_terminal_action(self, action):
        return action.type() == ACTION_NON_TERMINAL_TYPE

    def get_indices_for_action(self, actions_indices, condition):
        indices = [index for index, action in actions_indices if condition(action)]
        return indices
