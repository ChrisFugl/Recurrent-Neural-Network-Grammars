from app.data.actions.non_terminal import NonTerminalAction
from app.constants import ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE
from app.models.abstract_rnng import AbstractRNNG
from app.models.rnng.action_args import ActionOutputs, ActionLogProbs
from app.models.rnng.state import RNNGState
from joblib import Parallel, delayed
import torch

INVALID_ACTION_LOG_PROB = - 10e10

class RNNG(AbstractRNNG):

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads, action_set, generative):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.stack.Stack, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter, app.data.converters.non_terminal.NonTerminalConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type threads: int
        :type action_set: app.data.action_sets.action_set.ActionSet
        :type generative: bool
        """
        action_converter = converters[0]
        self.action_count = action_converter.count()
        base_action_count = self.action_count
        if generative:
            base_action_count = self.action_count - action_converter.count_terminals() + 1
        super().__init__(device, embeddings, structures, converters, representation, composer, sizes, action_set, generative, base_action_count)
        self.threads = threads
        self.nt_count = self.action_converter.count_non_terminals()
        self.nt_offset = self.action_converter.get_non_terminal_offset()
        self.nt_action_indices = list(range(self.nt_offset, self.nt_offset + self.nt_count))
        self.gen_count = self.action_converter.count_terminals()
        self.gen_offset = self.action_converter.get_terminal_offset()
        self.gen_action_indices = list(range(self.gen_offset, self.gen_offset + self.gen_count))

        # start at 1 since index 0 is reserved for padding
        self.reduce_index = 1
        self.index2action = {self.reduce_index: 1}
        if self.generative:
            self.gen_index = 2
            self.index2action[self.gen_index] =2
        else:
            self.shift_index = 2
            self.index2action[self.shift_index] = 2
        self.nt_start = 3
        self.nt_indices = list(range(self.nt_start, self.nt_start + self.nt_count))
        for nt_index in self.nt_indices:
            self.index2action[nt_index] = nt_index
        self.type2action = {
            ACTION_REDUCE_TYPE: self.reduce,
            ACTION_NON_TERMINAL_TYPE: self.non_terminal,
            ACTION_SHIFT_TYPE: self.shift,
            ACTION_GENERATE_TYPE: self.generate,
        }

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        jobs_args = []
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            tokens_tensor = element.tokens.tensor[:element.tokens.length]
            tags_tensor = element.tags.tensor[:element.tags.length]
            actions_tensor = element.actions.tensor[:element.actions.length]
            actions = element.actions.actions
            actions_max_length = element.actions.max_length
            job_args = (tokens_tensor, tags_tensor, actions_tensor, actions, actions_max_length)
            jobs_args.append(job_args)
        if 1 < self.threads:
            get_log_probs = delayed(self.tree_log_probs)
            log_probs_list = Parallel(n_jobs=self.threads, backend='threading')(get_log_probs(*job_args) for job_args in jobs_args)
        else:
            log_probs_list = [self.tree_log_probs(*args) for args in jobs_args]
        log_probs = torch.stack(log_probs_list, dim=1)
        return log_probs

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
        actions_length = len(actions)
        tokens_length = len(tokens_tensor)
        if actions_max_length is None:
            actions_max_length = actions_length
        token_counter = 0
        open_non_terminals_count = 0
        action_top, stack_top, token_top = self.initialize_structures(tokens_tensor, tags_tensor, tokens_length)
        output_log_probs = torch.zeros((actions_max_length, self.action_count), dtype=torch.float, device=self.device)
        for sequence_index in range(actions_length):
            action = actions[sequence_index]
            representation = self.get_representation(action_top, stack_top, token_top)
            logits = self.representation2logits(representation)
            log_probs = self.logits2log_prob(logits)
            action_args_log_probs = ActionLogProbs(representation, log_probs, self.index2action)
            action_args_outputs = ActionOutputs(stack_top, token_top, open_non_terminals_count, token_counter)
            action_fn = self.type2action[action.type()]
            action_outputs = action_fn(action_args_log_probs, action_args_outputs, action)
            action_log_prob, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
            action_index = self.action_converter.action2integer(action)
            output_log_probs[sequence_index, action_index] = action_log_prob
            action_tensor = actions_tensor[sequence_index].unsqueeze(dim=0)
            action_embedding = self.action_embedding(action_tensor)
            action_top = self.action_history.push(action_embedding, data=action, top=action_top)
        return output_log_probs

    def initial_state(self, tokens, tags, lengths):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :type lengths: torch.Tensor
        :returns: initial state
        :rtype: list of app.models.rnng.state.RNNGState
        """
        states = []
        for i, length in enumerate(lengths):
            token_counter = 0
            open_non_terminals_count = 0
            tokens_tensor = tokens[:length, i].unsqueeze(dim=1)
            tags_tensor = tags[:length, i].unsqueeze(dim=1)
            action_top, stack_top, token_top = self.initialize_structures(tokens_tensor, tags_tensor, length)
            state = RNNGState(stack_top, action_top, token_top, tokens_tensor, length, open_non_terminals_count, token_counter)
            states.append(state)
        return states

    def next_state(self, states, actions):
        """
        Advance state of the model to the next state.

        :type states: list of app.models.rnng.state.RNNGState
        :type actions: list of app.data.actions.action.Action
        :rtype: list of app.models.rnng.state.RNNGState
        """
        next_states = []
        for i, (state, action) in enumerate(zip(states, actions)):
            if action is None:
                next_state = state
            else:
                tokens_length = state.tokens_length
                token_counter = state.token_counter
                last_action = state.stack_top.data
                open_non_terminals_count = state.open_non_terminals_count
                valid_actions = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
                assert action.type() in valid_actions, f'{action} is not a valid action. (valid_actions = {valid_actions})'
                action_args_outputs = ActionOutputs(state.stack_top, state.token_top, open_non_terminals_count, token_counter)
                action_outputs = self.type2action[action.type()](None, action_args_outputs, action)
                _, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
                action_index = self.action_converter.action2integer(action)
                action_tensor = torch.tensor([[action_index]], device=self.device, dtype=torch.long)
                action_embedding = self.action_embedding(action_tensor)
                action_top = self.action_history.push(action_embedding, data=action, top=state.action_top)
                next_state = state.next(stack_top, action_top, token_top, open_non_terminals_count, token_counter)
            next_states.append(next_state)
        return next_states

    def next_action_log_probs(self, states, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :type states: list of app.models.rnng.state.RNNGState
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor
        """
        batch_size = len(states)
        batch_log_probs = torch.empty((batch_size, self.action_count), device=self.device, dtype=torch.float)
        batch_log_probs.fill_(INVALID_ACTION_LOG_PROB)
        batch_valid_actions = self.valid_actions(states)
        for i, (state, valid_actions) in enumerate(zip(states, batch_valid_actions)):
            representation = self.get_representation(state.action_top, state.stack_top, state.token_top)
            valid_indices, action2index = self.get_valid_indices(valid_actions)
            # base log probabilities
            logits = self.representation2logits(representation)
            valid_logits = logits[:, :, valid_indices]
            valid_log_probs = self.logits2log_prob(posterior_scaling * valid_logits).view((-1,))
            if ACTION_REDUCE_TYPE in valid_actions:
                batch_log_probs[i, self.reduce_index] = valid_log_probs[action2index[self.reduce_index]].view(1)
            if not self.generative and ACTION_SHIFT_TYPE in valid_actions:
                batch_log_probs[i, self.shift_index] = valid_log_probs[action2index[self.shift_index]].view(1)
            # token log probabilities for generative model
            if self.generative and ACTION_GENERATE_TYPE in valid_actions:
                gen_log_prob = valid_log_probs[action2index[self.gen_index]]
                if token is None and include_gen:
                    tokens_log_probs = self.token_distribution.log_probs(representation, posterior_scaling=posterior_scaling)
                    token_log_probs = gen_log_prob + tokens_log_probs
                    batch_log_probs[i, self.gen_action_indices] = token_log_probs
                if token is not None:
                    conditional_token_log_probs = self.token_distribution.log_prob(representation, token, posterior_scaling=posterior_scaling)
                    token_log_prob = gen_log_prob + conditional_token_log_probs.view(1)
                    token_action_index = self.action_converter.token2integer(token)
                    batch_log_probs[i, token_action_index] = token_log_prob
            # non-terminal log probabilities
            if include_nt and ACTION_NON_TERMINAL_TYPE in valid_actions:
                nt_valid_indices = [action2index[nt_index] for nt_index in self.nt_indices]
                batch_log_probs[i, self.nt_action_indices] = valid_log_probs[nt_valid_indices].view(-1)
        return batch_log_probs

    def valid_actions(self, states):
        """
        :type states: list of app.models.rnng.state.RNNGState
        :rtype: list of list of int
        """
        batch_valid_actions = []
        for state in states:
            tokens_length = state.tokens_length
            token_counter = state.token_counter
            last_action = state.stack_top.data
            open_non_terminals_count = state.open_non_terminals_count
            valid_actions = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
            batch_valid_actions.append(valid_actions)
        return batch_valid_actions

    def reduce(self, log_probs, outputs, action):
        stack_top = outputs.stack_top
        children = []
        while True:
            popped_action, state = stack_top.data, stack_top.output
            if popped_action.type() == ACTION_NON_TERMINAL_TYPE and popped_action.open:
                break
            stack_top = self.stack.pop(stack_top)
            children.append(state)
        children_tensor = torch.cat(children, dim=0)
        compose_action = NonTerminalAction(popped_action.argument, open=False)
        stack_top = self.stack.pop(stack_top)
        nt_embedding = self.get_nt_embedding(self.nt_compose_embedding, compose_action)
        children_lengths = torch.tensor([len(children)], device=self.device, dtype=torch.long)
        composed = self.composer(nt_embedding, children_tensor, children_lengths)
        stack_top = self.stack.push(composed, data=compose_action, top=stack_top)
        action_log_prob = self.get_base_log_prop(log_probs, self.reduce_index)
        open_non_terminals_count = outputs.open_non_terminals_count - 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def non_terminal(self, log_probs, outputs, action):
        nt_embedding = self.get_nt_embedding(self.nt_embedding, action)
        stack_top = self.stack.push(nt_embedding, data=action, top=outputs.stack_top)
        if log_probs is None:
            action_log_prob = None
        else:
            nt_action_index = self.action_converter.action2integer(action) - self.nt_offset
            action_log_prob = self.get_base_log_prop(log_probs, self.nt_start + nt_action_index)
        open_non_terminals_count = outputs.open_non_terminals_count + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def shift(self, log_probs, outputs, action):
        raise NotImplementedError('must be implemented by subclass')

    def generate(self, log_probs, outputs, action):
        raise NotImplementedError('must be implemented by subclass')

    def initialize_structures(self, tokens, tags, length):
        action_top = self.action_history.push(self.start_action_embedding.view(1, 1, -1))
        stack_top = self.stack.push(self.start_stack_embedding.view(1, 1, -1))
        token_top = self.initialize_token_buffer(tokens, tags, length)
        return action_top, stack_top, token_top

    def initialize_token_buffer(self, tokens_tensor, tags_tensor, length):
        """
        :type tokens_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type length: int
        :rtype: app.models.rnng.stack.StackNode
        """
        raise NotImplementedError('must be implemented by subclass')

    def get_nt_embedding(self, embeddings, action):
        nt_index = self.non_terminal_converter.non_terminal2integer(action.argument)
        nt_tensor = torch.tensor([nt_index], device=self.device, dtype=torch.long)
        nt_embedding = embeddings(nt_tensor).unsqueeze(dim=0)
        return nt_embedding

    def get_base_log_prop(self, log_probs, action_index):
        if log_probs is None:
            return None
        else:
            return log_probs.log_prob_base[:, :, log_probs.action2index[action_index]]

    def get_representation(self, action_top, stack_top, token_top):
        """
        :type action_top: app.models.rnng.stack.StackNode
        :type stack_top: app.models.rnng.stack.StackNode
        :type token_top: app.models.rnng.stack.StackNode
        """
        action_history_embedding = self.action_history.contents(action_top)
        stack_embedding = self.stack.contents(stack_top)
        token_buffer_embedding = self.token_buffer.contents(token_top)
        return self.representation(
            action_history_embedding, action_top.length_as_tensor(self.device),
            stack_embedding, stack_top.length_as_tensor(self.device),
            token_buffer_embedding, token_top.length_as_tensor(self.device),
        )

    def get_valid_indices(self, valid_actions):
        valid_indices = []
        action2index = {}
        counter = 0
        if ACTION_REDUCE_TYPE in valid_actions:
            valid_indices.append(self.reduce_index)
            action2index[self.reduce_index] = counter
            counter += 1
        if ACTION_SHIFT_TYPE in valid_actions:
            valid_indices.append(self.shift_index)
            action2index[self.shift_index] = counter
            counter += 1
        if ACTION_GENERATE_TYPE in valid_actions:
            valid_indices.append(self.gen_index)
            action2index[self.gen_index] = counter
            counter += 1
        if ACTION_NON_TERMINAL_TYPE in valid_actions:
            valid_indices.extend(self.nt_indices)
            for nt_index in self.nt_indices:
                action2index[nt_index] = counter
                counter += 1
        return valid_indices, action2index
