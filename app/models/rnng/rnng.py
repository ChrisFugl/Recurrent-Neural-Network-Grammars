from app.data.actions.non_terminal import NonTerminalAction
from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_NON_TERMINAL_INDEX, ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX,
    ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE,
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

    def __init__(self, device, embeddings, structures, converters, representation, composer, sizes, threads):
        """
        :type device: torch.device
        :type embeddings: torch.Embedding, torch.Embedding, torch.Embedding, torch.Embedding
        :type structures: app.models.rnng.stack.Stack, app.models.rnng.stack.Stack, app.models.rnng.stack.Stack
        :type converters: app.data.converters.action.ActionConverter, app.data.converters.token.TokenConverter, app.data.converters.tag.TagConverter
        :type representation: app.representations.representation.Representation
        :type composer: app.composers.composer.Composer
        :type sizes: int, int, int, int
        :type threads: int
        """
        super().__init__()
        action_size, token_size, rnn_input_size, rnn_size = sizes
        self.device = device
        self.threads = threads
        self.action_converter, self.token_converter, self.tag_converter = converters
        self.non_terminal_count = self.action_converter.count_non_terminals()
        self.action_count = self.action_converter.count()

        self.action_embedding, self.token_embedding, self.nt_embedding, self.nt_compose_embedding = embeddings
        self.action_history, self.token_buffer, self.stack = structures
        self.representation = representation
        self.representation2logits = nn.Linear(in_features=rnn_size, out_features=ACTIONS_COUNT, bias=True)
        self.composer = composer
        self.logits2log_prob = nn.LogSoftmax(dim=2)
        self.representation2nt_logits = nn.Linear(in_features=rnn_size, out_features=self.non_terminal_count, bias=True)

        start_action_embedding = torch.FloatTensor(1, 1, action_size).uniform_(-1, 1)
        self.start_action_embedding = nn.Parameter(start_action_embedding, requires_grad=True)
        start_token_embedding = torch.FloatTensor(1, 1, token_size).uniform_(-1, 1)
        self.start_token_embedding = nn.Parameter(start_token_embedding, requires_grad=True)
        start_stack_embedding = torch.FloatTensor(1, 1, rnn_input_size).uniform_(-1, 1)
        self.start_stack_embedding = nn.Parameter(start_stack_embedding, requires_grad=True)

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
        # stack operations
        jobs_args = []
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            tokens_tensor = element.tokens.tensor[:element.tokens.length, :]
            tags_tensor = element.tags.tensor[:element.tags.length, :]
            actions_tensor = element.actions.tensor[:element.actions.length, :]
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
        log_probs = torch.zeros((actions_max_length, self.action_count), dtype=torch.float, device=self.device, requires_grad=False)
        for sequence_index in range(actions_length):
            action = actions[sequence_index]
            last_action = stack_top.data
            representation = self.get_representation(action_top, stack_top, token_top)
            valid_actions, action2index = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            base_logits = self.representation2logits(representation)
            valid_base_logits = base_logits[:, :, valid_actions]
            valid_base_log_probs = self.logits2log_prob(valid_base_logits)
            # get log probability of action
            action_args_log_probs = ActionLogProbs(representation, valid_base_log_probs, action2index)
            action_args_outputs = ActionOutputs(stack_top, token_top, open_non_terminals_count, token_counter)
            action_outputs = self.type2action[action.type()](action_args_log_probs, action_args_outputs, action)
            action_log_prob, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
            action_index = self.action_converter.action2integer(action)
            log_probs[sequence_index, action_index] = action_log_prob
            action_tensor = actions_tensor[sequence_index:sequence_index+1, :]
            action_embedding = self.action_embedding(action_tensor)
            action_top = self.action_history.push(action_embedding, data=action, top=action_top)
        return log_probs

    def initial_state(self, tokens, tags):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :returns: initial state
        :rtype: app.models.rnng.state.RNNGState
        """
        tokens_length = tokens.size(0)
        token_counter = 0
        open_non_terminals_count = 0
        action_top, stack_top, token_top = self.initialize_structures(tokens, tags, tokens_length)
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
        valid_actions, action2index = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
        assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
        action_args_outputs = ActionOutputs(state.stack_top, state.token_top, open_non_terminals_count, token_counter)
        action_outputs = self.type2action[action.type()](None, action_args_outputs, action)
        _, stack_top, token_top, open_non_terminals_count, token_counter = action_outputs
        action_index = self.action_converter.action2integer(action)
        action_tensor = self.index2tensor(action_index)
        action_embedding = self.action_embedding(action_tensor)
        action_top = self.action_history.push(action_embedding, data=action, top=state.action_top)
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
        representation = self.get_representation(state.action_top, state.stack_top, state.token_top)
        valid_base_actions, action2index = self.action_set.valid_actions(tokens_length, token_counter, last_action, open_non_terminals_count)
        index2action_index = []
        singleton_offset = self.action_converter.get_singleton_offset()
        index2action_index.extend([singleton_offset + index for index in valid_base_actions])
        # base log probabilities
        base_logits = self.representation2logits(representation)
        valid_base_logits = base_logits[:, :, valid_base_actions]
        valid_base_log_probs = self.logits2log_prob(posterior_scaling * valid_base_logits).view((-1,))
        # log_probs should not contain class actions: generate, non-terminal
        log_probs_list = []
        if ACTION_REDUCE_INDEX in valid_base_actions:
            log_probs_list.append(valid_base_log_probs[action2index[ACTION_REDUCE_INDEX]].view(1))
        if not self.generative and ACTION_SHIFT_INDEX in valid_base_actions:
            log_probs_list.append(valid_base_log_probs[action2index[ACTION_SHIFT_INDEX]].view(1))
        # token log probabilities for generative model
        if self.generative and ACTION_GENERATE_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_GENERATE_INDEX)
            base_generate_log_prob = valid_base_log_probs[action2index[ACTION_GENERATE_INDEX]]
            if token is None and include_gen:
                conditional_token_log_probs, token_index2action_index = self.token_distribution.log_probs(
                    representation,
                    posterior_scaling=posterior_scaling
                )
                token_log_probs = base_generate_log_prob + conditional_token_log_probs
                log_probs_list.append(token_log_probs)
                index2action_index.extend(token_index2action_index)
            if token is not None:
                token_distribution_log_prob = self.token_distribution.log_prob(representation, token, posterior_scaling=posterior_scaling).view(1)
                token_log_prob = base_generate_log_prob + token_distribution_log_prob
                log_probs_list.append(token_log_prob)
                token_action_index = self.action_converter.token2integer(token)
                index2action_index.append(token_action_index)
        # non-terminal log probabilities
        if ACTION_NON_TERMINAL_INDEX in valid_base_actions:
            index2action_index.remove(singleton_offset + ACTION_NON_TERMINAL_INDEX)
            if include_nt:
                non_terminal_logits = self.representation2nt_logits(representation)
                base_non_terminal_log_prob = valid_base_log_probs[action2index[ACTION_NON_TERMINAL_INDEX]]
                conditional_non_terminal_log_probs = self.logits2log_prob(posterior_scaling * non_terminal_logits).view((-1,))
                non_terminal_log_probs = base_non_terminal_log_prob + conditional_non_terminal_log_probs
                log_probs_list.append(non_terminal_log_probs)
                non_terminal_offset = self.action_converter.get_non_terminal_offset()
                non_terminal2action_index = [non_terminal_offset + index for index in range(self.non_terminal_count)]
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
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def reduce(self, log_probs, outputs, action):
        stack_top = outputs.stack_top
        children = []
        while True:
            action, state = stack_top.data, stack_top.output
            if action.type() == ACTION_NON_TERMINAL_TYPE and action.open:
                break
            stack_top = self.stack.pop(stack_top)
            children.append(state)
        children.reverse()
        children_tensor = torch.cat(children, dim=0)
        compose_action = NonTerminalAction(self.device, action.argument, action.argument_index, open=False)
        stack_top = self.stack.pop(stack_top)
        nt_embeding, _ = self.get_nt_embedding(self.nt_compose_embedding, compose_action)
        children_lengths = torch.tensor([len(children)], device=self.device, dtype=torch.long)
        composed = self.composer(nt_embeding, children_tensor, children_lengths)
        stack_top = self.stack.push(composed, data=compose_action, top=stack_top)
        action_log_prob = self.get_base_log_prop(log_probs, ACTION_REDUCE_INDEX)
        open_non_terminals_count = outputs.open_non_terminals_count - 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def non_terminal(self, log_probs, outputs, action):
        nt_embeding, argument_index = self.get_nt_embedding(self.nt_embedding, action)
        stack_top = self.stack.push(nt_embeding, data=action, top=outputs.stack_top)
        if log_probs is None:
            action_log_prob = None
        else:
            non_terminal_log_prob = self.get_base_log_prop(log_probs, ACTION_NON_TERMINAL_INDEX)
            non_terminal_logits = self.representation2nt_logits(log_probs.representation)
            conditional_non_terminal_log_probs = self.logits2log_prob(non_terminal_logits)
            conditional_non_terminal_log_prob = conditional_non_terminal_log_probs[:, :, argument_index]
            action_log_prob = non_terminal_log_prob + conditional_non_terminal_log_prob
        open_non_terminals_count = outputs.open_non_terminals_count + 1
        return outputs.update(action_log_prob=action_log_prob, stack_top=stack_top, open_non_terminals_count=open_non_terminals_count)

    def shift(self, log_probs, outputs, action):
        raise NotImplementedError('must be implemented by subclass')

    def generate(self, log_probs, outputs, action):
        raise NotImplementedError('must be implemented by subclass')

    def initialize_structures(self, tokens, tags, length):
        action_top = self.action_history.push(self.start_action_embedding)
        stack_top = self.stack.push(self.start_stack_embedding)
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

    def index2tensor(self, index):
        return torch.tensor([[index]], device=self.device, dtype=torch.long)

    def get_nt_embedding(self, embeddings, action):
        argument_index = action.argument_index_as_tensor()
        nt_embeding = embeddings(argument_index).unsqueeze(dim=0).unsqueeze(dim=0)
        return nt_embeding, argument_index

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
