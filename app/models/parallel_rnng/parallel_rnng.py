from app.constants import PAD_INDEX
from app.models.abstract_rnng import AbstractRNNG
from app.models.parallel_rnng.state import StateFactory
import torch

INVALID_ACTION_FILL = - 10e10

class ParallelRNNG(AbstractRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, sample_stack_size, action_set, generative):
        """
        :type device: torch.device
        :type embeddings: app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding, app.embeddings.embedding.Embedding
        :type structures: app.models.parallel_rnng.history_lstm.HistoryLSTM, app.models.parallel_rnng.buffer_lstm.BufferLSTM, app.models.parallel_rnng.stack_lstm.StackLSTM
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type sample_stack_size: int
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
            self.gen_indices = None
        else:
            self.shift_index = None
            gen_start = self.action_converter.get_terminal_offset()
            gen_count = self.action_converter.count_terminals()
            self.gen_indices = list(range(gen_start, gen_start + gen_count))
        nt_start = self.action_converter.get_non_terminal_offset()
        nt_count = self.action_converter.count_non_terminals()
        self.nt_indices = list(range(nt_start, nt_start + nt_count))
        self.state_factory = StateFactory(
            self.device, self.action_converter, self.non_terminal_converter, self.token_converter,
            self.action_set, self.generative, self.action_count,
            self.reduce_index, self.shift_index, self.gen_indices, self.nt_indices
        )
        self.sample_stack_size = sample_stack_size

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor, dict
        """
        self.reset(batch.size)
        states, stack_size = self.preprocess_batch(batch)
        self.initialize_structures(
            batch.tokens.tokens,
            batch.tokens.tensor,
            batch.unknownified_tokens.tensor,
            batch.singletons,
            batch.tags.tensor,
            # plus one to account for start embeddings
            batch.tokens.lengths + 1,
            stack_size + 1,
            batch.actions.tensor
        )
        output_shape = (batch.max_actions_length, batch.size, self.action_count)
        output_log_probs = torch.zeros(output_shape, device=self.device, dtype=torch.float)
        history_attention_weights = []
        buffer_attention_weights = []
        stack_attention_weights = []
        weighted_attention_weights = []
        for sequence_index in range(batch.max_actions_length):
            state = states[sequence_index]
            representation, representation_info = self.get_representation()
            if 'history' in representation_info:
                history_attention_weights.append(representation_info['history'].detach().cpu())
            if 'buffer' in representation_info:
                buffer_attention_weights.append(representation_info['buffer'].detach().cpu())
            if 'stack' in representation_info:
                stack_attention_weights.append(representation_info['stack'].detach().cpu())
            if 'weighted' in representation_info:
                weighted_attention_weights.append(representation_info['weighted'].detach().cpu())
            output_log_probs[sequence_index] = self.get_log_probs(representation)
            self.do_actions(batch.size, state)
            if self.uses_history:
                action_op = self.hold_op(batch.size)
                action_op[state.non_pad_actions] = 1
                self.action_history.hold_or_push(action_op)
        info = {}
        if len(history_attention_weights) != 0:
            info['history'] = history_attention_weights
        if len(buffer_attention_weights) != 0:
            info['buffer'] = buffer_attention_weights
        if len(stack_attention_weights) != 0:
            info['stack'] = stack_attention_weights
        if len(weighted_attention_weights) != 0:
            info['weighted'] = weighted_attention_weights
        return output_log_probs, info

    def initial_state(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, lengths):
        """
        Get initial state of model in a parse.

        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: app.models.parallel_rnng.state.State
        """
        batch_size = lengths.size(0)
        self.reset(batch_size)
        # plus one to account for start embeddings
        self.initialize_structures(
            tokens, tokens_tensor, unknownified_tokens_tensor,
            singletons_tensor, tags_tensor, lengths + 1, self.sample_stack_size
        )
        state = self.state_factory.initialize(
            batch_size, tokens, tokens_tensor, unknownified_tokens_tensor,
            singletons_tensor, tags_tensor, lengths
        )
        return state

    def next_state(self, state, actions):
        """
        Advance state of the model to the next state.

        :type state: app.models.parallel_rnng.state.State
        :type actions: list of app.data.actions.action.Action
        :rtype: app.models.parallel_rnng.state.State
        """
        batch_size = len(actions)
        next_state = self.state_factory.next(state, actions)
        self.do_actions(batch_size, next_state)
        if self.uses_history:
            action_indices = []
            for action in actions:
                if action is None:
                    action_indices.append(PAD_INDEX)
                else:
                    action_indices.append(self.action_converter.action2integer(action))
            action_tensor = torch.tensor(action_indices, device=self.device, dtype=torch.long).view(1, batch_size)
            action_embedding = self.action_embedding(action_tensor)
            action_op = self.hold_op(batch_size)
            action_op[next_state.non_pad_actions] = 1
            self.action_history.hold_or_push(action_op, action_embedding)
        return next_state

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type state: app.models.parallel_rnng.state.State
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor
        """
        batch_size = len(state.token_counter)
        representation, _ = self.get_representation()
        log_probs = self.get_log_probs(representation, invalid_mask=state.invalid_mask, posterior_scaling=posterior_scaling)
        return log_probs.view(batch_size, self.action_count)

    def valid_actions(self, state):
        """
        :type state: app.models.parallel_rnng.state.State
        :rtype: list of list of int
        """
        iterator = zip(state.tokens_lengths, state.token_counter, state.last_action, state.open_nt_count)
        batch_valid_actions = []
        for tokens_length, token_counter, last_action, open_nt_count in iterator:
            valid_actions = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_nt_count)
            batch_valid_actions.append(valid_actions)
        return batch_valid_actions

    def preprocess_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of app.models.parallel_rnng.state.State, list of torch.Tensor, int
        """
        states = []
        state = self.state_factory.initialize(
            batch.size,
            batch.tokens.tokens,
            batch.tokens.tensor,
            batch.unknownified_tokens.tensor,
            batch.singletons,
            batch.tags.tensor,
            batch.tokens.lengths,
            make_invalid_mask=False
        )
        for action_index in range(batch.max_actions_length):
            next_actions = []
            for actions in batch.actions.actions:
                if action_index < len(actions):
                    next_actions.append(actions[action_index])
                else:
                    next_actions.append(None)
            state = self.state_factory.next(state, next_actions, make_invalid_mask=False)
            states.append(state)
        return states, state.max_stack_size

    def initialize_structures(self,
        tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor,
        tags_tensor, token_lengths, stack_size, actions=None
    ):
        batch_size = tokens_tensor.size(1)
        if self.uses_history:
            self.initialize_action_history(batch_size, actions)
        if self.uses_stack:
            self.initialize_stack(stack_size, batch_size)
        if self.uses_buffer:
            self.initialize_token_buffer(tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, token_lengths)

    def initialize_action_history(self, batch_size, actions):
        start_action_embedding = self.start_action_embedding.view(1, 1, -1).expand(1, batch_size, -1)
        action_op = self.push_op(batch_size)
        if actions is None:
            self.action_history.initialize(batch_size)
            self.action_history.hold_or_push(action_op, start_action_embedding)
        else:
            action_embeddings = self.action_embedding(actions)
            init_actions = torch.cat((start_action_embedding, action_embeddings), dim=0)
            self.action_history.initialize(batch_size, init_actions)

    def initialize_stack(self, stack_size, batch_size):
        start_stack_embedding = self.start_stack_embedding.view(1, -1).expand(batch_size, -1)
        self.stack.initialize(stack_size, batch_size)
        push_all_op = self.push_op(batch_size)
        self.stack.hold_or_push(start_stack_embedding, push_all_op)

    def initialize_token_buffer(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, token_lengths):
        """
        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type token_lengths: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_word_embedding(self, state):
        """
        :type state: app.models.parallel_rnng.state.State
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

    def hold_op(self, batch_size):
        return torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    def pop_op(self, batch_size):
        tensor = torch.empty((batch_size,), device=self.device, dtype=torch.long)
        tensor.fill_(-1)
        return tensor

    def push_op(self, batch_size):
        return torch.ones((batch_size,), device=self.device, dtype=torch.long)

    def get_representation(self):
        if self.representation.top_only():
            history_embedding, stack_embedding, buffer_embedding = None, None, None
            if self.uses_history:
                history_embedding = self.action_history.top()
            if self.uses_stack:
                stack_embedding = self.stack.top().unsqueeze(dim=0)
            if self.uses_buffer:
                buffer_embedding = self.token_buffer.top()
            return self.representation(history_embedding, None, stack_embedding, None, buffer_embedding, None)
        else:
            history_embedding, stack_embedding, buffer_embedding = None, None, None
            history_lengths, stack_lengths, buffer_lengths = None, None, None
            if self.uses_history:
                history_embedding, history_lengths = self.action_history.contents()
            if self.uses_stack:
                stack_embedding, stack_lengths = self.stack.contents()
            if self.uses_buffer:
                buffer_embedding, buffer_lengths = self.token_buffer.contents()
            return self.representation(history_embedding, history_lengths, stack_embedding, stack_lengths, buffer_embedding, buffer_lengths)

    def get_log_probs(self, representation, invalid_mask=None, posterior_scaling=1.0):
        """
        :type representation: torch.Tensor
        :type invalid_mask: torch.Tensor
        :rtype: torch.Tensor
        """
        logits = self.representation2logits(representation)
        logits = posterior_scaling * logits
        if invalid_mask is not None:
            logits = logits.masked_fill(invalid_mask, INVALID_ACTION_FILL)
        log_probs = self.logits2log_prob(logits)
        return log_probs

    def do_actions(self, batch_size, state):
        """
        :type batch_size: int
        :type state: app.models.parallel_rnng.state.State
        """
        token_action_indices = state.shift_actions + state.gen_actions
        reduce_action_indices = state.reduce_actions
        non_terminal_action_indices = state.nt_actions
        if self.uses_stack:
            non_pad_indices = state.non_pad_actions
            stack_input = torch.zeros((batch_size, self.rnn_input_size), device=self.device, dtype=torch.float)
        # NT
        if self.uses_stack and len(non_terminal_action_indices) != 0:
            nt_indices = state.nt_index[non_terminal_action_indices]
            nt_embeddings = self.nt_embedding(nt_indices)
            stack_input[non_terminal_action_indices] = nt_embeddings
        # SHIFT or GEN
        if (self.uses_stack or self.uses_buffer) and len(token_action_indices) != 0:
            word_embeddings = self.get_word_embedding(state)
            if self.uses_stack:
                stack_input[token_action_indices] = word_embeddings[token_action_indices]
            if self.uses_buffer:
                self.update_token_buffer(batch_size, token_action_indices, word_embeddings)
        # REDUCE
        if self.uses_stack and len(reduce_action_indices) != 0:
            children = []
            max_reduce_children = torch.max(state.number_of_children)
            for child_index in range(max_reduce_children):
                batch_pop = child_index < state.number_of_children
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
            compose_nt_index = state.compose_nt_index[reduce_action_indices]
            compose_nt_embeddings = self.nt_compose_embedding(compose_nt_index).unsqueeze(dim=0)
            compose_lengths = state.number_of_children[reduce_action_indices]
            composed = self.composer(compose_nt_embeddings, children_tensor, compose_lengths)
            stack_input[reduce_action_indices] = composed[0]
        if self.uses_stack:
            stack_op = self.hold_op(batch_size)
            stack_op[non_pad_indices] = 1
            self.stack.hold_or_push(stack_input, stack_op)

    def reset(self, batch_size):
        self.action_embedding.reset()
        self.nt_embedding.reset()
        self.nt_compose_embedding.reset()
        self.token_embedding.reset()
        if not self.generative:
            self.pos_embedding.reset()
            if self.pretrained is not None:
                self.pretrained.reset()
        if self.uses_buffer:
            self.token_buffer.reset(batch_size)
        if self.uses_history:
            self.action_history.reset(batch_size)
        if self.uses_stack:
            self.stack.reset(batch_size)
        self.composer.reset(batch_size)
        self.representation.reset(batch_size)
