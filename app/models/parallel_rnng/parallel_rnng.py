from app.constants import ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.data.actions.non_terminal import NonTerminalAction
from app.models.abstract_rnng import AbstractRNNG
from app.models.parallel_rnng.preprocess_batch import preprocess_batch
from app.models.parallel_rnng.state import State
from app.utils import padded_reverse
import torch
from torch import nn

INVALID_ACTION_FILL = -9999999999.0

class ParallelRNNG(AbstractRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, action_set, generative):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.buffer_lstm.BufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type action_set: app.data.action_sets.action_set.ActionSet
        :type generative: bool
        """
        action_converter = converters[0]
        self.action_count = action_converter.count()
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, action_set, generative, self.action_count)
        _, _, rnn_input_size, _ = sizes
        self.rnn_input_size = rnn_input_size
        self.reduce_index = self.action_converter.string2integer('REDUCE')
        if not self.generative:
            self.shift_index = self.action_converter.string2integer('SHIFT')
        else:
            self.shift_index = None
        nt_start = self.action_converter.get_non_terminal_offset()
        nt_count = self.action_converter.count_non_terminals()
        self.nt_indices = list(range(nt_start, nt_start + nt_count))
        gen_start = self.action_converter.get_terminal_offset()
        gen_count = self.action_converter.count_terminals()
        self.gen_indices = list(range(gen_start, gen_start + gen_count))

        self.valid_args = (self.action_set, self.generative, self.action_count, self.reduce_index, self.shift_index, gen_start, gen_count, nt_start, nt_count)

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        preprocessed_batches, stack_size = preprocess_batch(
            self.device,
            self.valid_args,
            self.non_terminal_converter,
            self.token_converter,
            batch.size,
            batch.max_actions_length,
            batch.actions.actions,
            batch.tokens.lengths,
            batch.tokens.tensor,
            batch.tags.tensor,
        )
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
        output_log_probs = torch.zeros(output_shape, device=self.device, dtype=torch.float, requires_grad=False)
        for sequence_index in range(batch.max_actions_length):
            preprocessed_batch = preprocessed_batches[sequence_index]
            representation = self.get_representation()
            output_log_probs[sequence_index] = self.get_log_probs(representation, preprocessed_batch.invalid_mask)
            self.do_actions(batch.size, preprocessed_batch)
            action_tensor = batch.actions.tensor[sequence_index].unsqueeze(dim=0)
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
        batch_size = 1
        if actions_max_length is None:
            actions_max_length = len(actions)
        tokens_length = torch.tensor([tokens_tensor.size(0)], device=self.device, dtype=torch.long)
        actions_length = torch.tensor([len(actions)], device=self.device, dtype=torch.long)
        preprocessed, stack_size = preprocess_batch(
            self.device,
            self.valid_args,
            self.non_terminal_converter,
            self.token_converter,
            batch_size,
            actions_max_length,
            [actions],
            tokens_length,
            tokens_tensor,
            tags_tensor
        )
        self.initialize_structures(
            actions_tensor,
            tokens_tensor,
            tags_tensor,
            # plus one to account for start embeddings
            actions_length + 1,
            tokens_length + 1,
            stack_size + 1,
        )
        output_shape = (actions_max_length, self.action_count)
        output_log_probs = torch.zeros(output_shape, device=self.device, dtype=torch.float)
        for sequence_index in range(actions_max_length):
            preprocessed_batch = preprocessed[sequence_index]
            representation = self.get_representation()
            action = actions[sequence_index]
            action_index = self.action_converter.action2integer(action)
            log_probs = self.get_log_probs(representation, preprocessed_batch.invalid_mask)
            action_log_prob = log_probs[:, :, action_index]
            output_log_probs[sequence_index, action_index] = action_log_prob
            self.do_actions(batch_size, preprocessed_batch)
            action_tensor = actions_tensor[sequence_index].unsqueeze(dim=0)
            action_embedding = self.action_embedding(action_tensor)
            self.action_history.push(action_embedding)
        return output_log_probs

    def initial_state(self, tokens, tags):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :returns: initial state
        :rtype: app.models.parallel_rnng.state.State
        """
        batch_size = tokens.size(1)
        push_op = self.push_op(batch_size)
        start_action_embedding = self.start_action_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        history_state = self.action_history.inference_initialize(batch_size)
        history_state = self.action_history.inference_push(history_state, start_action_embedding)
        start_actions = [None] * batch_size
        start_stack_embedding = self.start_stack_embedding.view(1, -1).expand(batch_size, -1)
        stack_state = self.stack.inference_initialize(batch_size)
        stack_state = self.stack.inference_hold_or_push(stack_state, start_actions, start_stack_embedding, push_op)
        buffer_state = self.inference_initialize_token_buffer(tokens, tags)
        tokens_length = tokens.size(0)
        token_counter = 0
        last_action = None
        open_nt_count = 0
        state = State(history_state, stack_state, buffer_state, tokens_length, token_counter, last_action, open_nt_count)
        return state

    def next_state(self, state, action):
        """
        Advance state of the model to the next state.
        Assumes a batch size of 1.

        :type state: app.models.parallel_rnng.state.State
        :type action: app.data.actions.action.Action
        :rtype: app.models.parallel_rnng.state.State
        """
        batch_size = 1
        history_state = state.history
        stack_state = state.stack
        buffer_state = state.buffer
        push_op = self.push_op(batch_size)
        token_counter = state.token_counter
        open_nt_count = state.open_nt_count
        if action.type() == ACTION_NON_TERMINAL_TYPE:
            nt_index = self.non_terminal_converter.non_terminal2integer(action.argument)
            nt_tensor = torch.tensor([nt_index], device=self.device, dtype=torch.long)
            nt_embedding = self.nt_embedding(nt_tensor)
            stack_state = self.stack.inference_hold_or_push(stack_state, [action], nt_embedding, push_op)
            open_nt_count = open_nt_count + 1
        elif action.type() == ACTION_SHIFT_TYPE or action.type() == ACTION_GENERATE_TYPE:
            buffer_state = self.inference_update_token_buffer(buffer_state)
            word_embeddings = self.token_buffer.inference_top_embeddings(buffer_state)
            stack_state = self.stack.inference_hold_or_push(stack_state, [action], word_embeddings, push_op)
            token_counter = token_counter + 1
        else: # reduce
            pop_op = self.pop_op(batch_size)
            children = []
            while True:
                child_action = stack_state.actions[0]
                if child_action.type() == ACTION_NON_TERMINAL_TYPE and child_action.open:
                    break
                stack_state, child = self.stack.inference_hold_or_pop(stack_state, pop_op)
                children.append(child)
            children_tensor = torch.stack(children, dim=0)
            compose_action = NonTerminalAction(child_action.argument, open=False)
            stack_state, _ = self.stack.inference_hold_or_pop(stack_state, pop_op)
            nt_index = self.non_terminal_converter.non_terminal2integer(compose_action.argument)
            nt_tensor = torch.tensor([nt_index], device=self.device, dtype=torch.long)
            nt_embedding = self.nt_compose_embedding(nt_tensor).view(1, 1, -1)
            children_lengths = torch.tensor([len(children)], device=self.device, dtype=torch.long)
            composed = self.composer(nt_embedding, children_tensor, children_lengths).view(1, -1)
            stack_state = self.stack.inference_hold_or_push(stack_state, [compose_action], composed, push_op)
            open_nt_count = open_nt_count - 1
        # add to history
        action_index = self.action_converter.action2integer(action)
        action_tensor = torch.tensor([[action_index]], device=self.device, dtype=torch.long)
        action_embedding = self.action_embedding(action_tensor)
        history_state = self.action_history.inference_push(history_state, action_embedding)
        next_state = State(history_state, stack_state, buffer_state, state.tokens_length, token_counter, action, open_nt_count)
        return next_state

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.parallel_rnng.state.State
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor, list of int
        """
        history_embedding, history_lengths = self.action_history.inference_contents(state.history)
        stack_embedding, stack_lengths = self.stack.inference_contents(state.stack)
        token_buffer_embedding, token_buffer_lengths = self.token_buffer.inference_contents(state.buffer)
        representation = self.representation(
            history_embedding, history_lengths,
            stack_embedding, stack_lengths,
            token_buffer_embedding, token_buffer_lengths
        )
        valid_actions = self.action_set.valid_actions(state.tokens_length, state.token_counter, state.last_action, state.open_nt_count)
        valid_indices = self.get_valid_indices(valid_actions, token=token, include_gen=include_gen, include_nt=include_nt)
        logits = self.representation2logits(representation)
        valid_logits = logits[:, :, valid_indices]
        log_probs = self.logits2log_prob(posterior_scaling * valid_logits).view(-1)
        return log_probs, valid_indices

    def initialize_structures(self, actions, tokens, tags, action_lengths, token_lengths, stack_size):
        batch_size = tokens.size(1)
        self.initialize_action_history(batch_size, actions, action_lengths)
        self.initialize_stack(stack_size, batch_size)
        self.initialize_token_buffer(tokens, tags, token_lengths)

    def initialize_action_history(self, batch_size, actions, action_lengths):
        start_action_embedding = self.start_action_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        self.action_history.initialize(action_lengths)
        self.action_history.push(start_action_embedding)

    def initialize_stack(self, stack_size, batch_size):
        start_stack_embedding = self.start_stack_embedding.view(1, -1).expand(batch_size, -1)
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

    def inference_initialize_token_buffer(self, tokens_tensor, tags_tensor):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_word_embedding(self, preprocessed):
        """
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
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

    def inference_update_token_buffer(self, state):
        """
        :type state: app.models.parallel_rnng.buffer_lstm.BufferState
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        raise NotImplementedError('must be implemented by subclass')

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

    def get_log_probs(self, representation, invalid_mask):
        """
        :type representation: torch.Tensor
        :type invalid_mask: torch.Tensor
        :rtype: torch.Tensor
        """
        logits = self.representation2logits(representation)
        valid_logits = logits.masked_fill(invalid_mask, INVALID_ACTION_FILL)
        log_probs = self.logits2log_prob(valid_logits)
        return log_probs

    def do_actions(self, batch_size, preprocessed):
        """
        :type batch_size: int
        :type preprocessed: app.models.parallel_rnng.preprocessed_batch.Preprocessed
        """
        token_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_token_action)
        reduce_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_reduce_action)
        non_terminal_action_indices = self.get_indices_for_action(preprocessed.actions_indices, self.is_non_terminal_action)
        non_pad_indices = token_action_indices + reduce_action_indices + non_terminal_action_indices
        stack_input = torch.zeros((batch_size, self.rnn_input_size), device=self.device, dtype=torch.float)
        # NT
        if len(non_terminal_action_indices) != 0:
            nt_indices = preprocessed.non_terminal_index[non_terminal_action_indices]
            nt_embeddings = self.nt_embedding(nt_indices)
            stack_input[non_terminal_action_indices] = nt_embeddings
        # SHIFT or GEN
        if len(token_action_indices) != 0:
            word_embeddings = self.get_word_embedding(preprocessed)
            stack_input[token_action_indices] = word_embeddings[token_action_indices]
            self.update_token_buffer(batch_size, token_action_indices, word_embeddings)
        # REDUCE
        if len(reduce_action_indices) != 0:
            children = []
            max_reduce_children = torch.max(preprocessed.number_of_children)
            for child_index in range(max_reduce_children):
                batch_pop = child_index < preprocessed.number_of_children
                reduce_children_op = self.hold_op(batch_size)
                reduce_children_op[batch_pop] = -1
                output = self.stack.hold_or_pop(reduce_children_op)
                children.append(output[reduce_action_indices])
            children_tensor = torch.stack(children, dim=0)
            # pop non-terminal from stack
            non_terminal_pop_op = self.hold_op(batch_size)
            non_terminal_pop_op[reduce_action_indices] = -1
            self.stack.hold_or_pop(non_terminal_pop_op)
            # compose and push composed constituent to stack
            compose_nt_index = preprocessed.compose_non_terminal_index[reduce_action_indices]
            compose_nt_embeddings = self.nt_compose_embedding(compose_nt_index).unsqueeze(dim=0)
            compose_lengths = preprocessed.number_of_children[reduce_action_indices]
            composed = self.composer(compose_nt_embeddings, children_tensor, compose_lengths)
            stack_input[reduce_action_indices] = composed[0]
        stack_op = self.hold_op(batch_size)
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

    def get_valid_indices(self, valid_actions, include_nt=True, include_gen=True, token=None):
        valid_indices = []
        if ACTION_REDUCE_TYPE in valid_actions:
            valid_indices.append(self.reduce_index)
        if not self.generative and ACTION_SHIFT_TYPE in valid_actions:
            valid_indices.append(self.shift_index)
        if self.generative and ACTION_GENERATE_TYPE in valid_actions:
            gen_indices = []
            if token is None and include_gen:
                gen_indices = self.gen_indices
            elif token is not None:
                token_action_index = self.action_converter.token2integer(token)
                gen_indices = [token_action_index]
            valid_indices.extend(gen_indices)
        if include_nt and ACTION_NON_TERMINAL_TYPE in valid_actions:
            valid_indices.extend(self.nt_indices)
        return valid_indices
