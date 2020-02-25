from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX, ACTION_NON_TERMINAL_INDEX,
    ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE
)
from app.models.model import Model
from joblib import Parallel, delayed
import threading
import torch
# import torch.multiprocessing as mp
from torch import nn

# base actions: (REDUCE, GENERATE, NON_TERMINAL) or (REDUCE, SHIFT, NON_TERMINAL)
ACTIONS_COUNT = 3

class RNNG(Model):

    def __init__(
        self, device,
        action_embedding, token_embedding,
        non_terminal_embedding, non_terminal_compose_embedding,
        action_history, token_buffer, stack,
        representation, representation_size,
        composer,
        token_distribution,
        non_terminal_count,
        action_set,
        threads
    ):
        """
        :type device: torch.device
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
        """
        super().__init__()
        self._device = device
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
        self._token_distribution = token_distribution
        self._logits2log_prob = nn.LogSoftmax(dim=2)
        self._representation2non_terminal_logits = nn.Linear(in_features=representation_size, out_features=non_terminal_count, bias=True)

    def forward(self, tokens, actions):
        """
        :type tokens: torch.Tensor, list of int, list of list of str
        :type actions: torch.Tensor, list of int, list of list of str
        :rtype: torch.Tensor
        """
        return self.log_likelihood(tokens, actions)

    def log_likelihood(self, batch_tokens, batch_actions):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type tokens: torch.Tensor, list of int, list of list of str
        :type actions: torch.Tensor, list of int, list of list of app.actions.action.Action
        :rtype: torch.Tensor
        """
        action_history = self._action_history.new()
        token_buffer = self._token_buffer.new()

        tokens_tensor, tokens_lengths, _ = batch_tokens
        actions_tensor, actions_lengths, actions = batch_actions
        actions_embedding = self._action_embedding(actions_tensor)
        tokens_embedding = self._token_embedding(tokens_tensor)

        # add to action history and token buffer
        actions_after_first_embedding = actions_embedding[1:]
        action_history.add(actions_after_first_embedding)
        token_buffer.add(tokens_embedding)

        # stack operations
        max_sequence_length, batch_size = actions_tensor.shape
        jobs_args = []
        for batch_index in range(batch_size):
            job_args = (
                action_history, token_buffer,
                max_sequence_length, batch_index,
                actions_embedding, tokens_embedding,
                actions[batch_index], actions_lengths[batch_index],
                tokens_lengths[batch_index]
            )
            jobs_args.append(job_args)
        if 1 < self._threads:
            get_log_probs = delayed(self._actions_log_probs)
            log_probs_list = Parallel(n_jobs=self._threads, backend='threading')(get_log_probs(*job_args) for job_args in jobs_args)
        else:
            log_probs_list = list(map(lambda job_args: self._actions_log_probs(*job_args), jobs_args))
        log_probs = torch.stack(log_probs_list, dim=1)

        return log_probs

    def next_state(self, previous_state):
        """
        Compute next state and action probs given the previous state.

        :returns: next state, next_actions
        """
        # TODO
        raise NotImplementedError('not implemented yet')

    def parse(self, tokens):
        """
        Generate a parse of a sentence.

        :type tokens: torch.Tensor
        """
        # TODO
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
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def _actions_log_probs(self,
        action_history, token_buffer,
        max_sequence_length, batch_index,
        actions_embedding, tokens_embedding,
        actions, sequence_length, sequence_tokens_count
    ):
        stack = self._stack.new()
        token_index = 0
        token_counter = 0
        # first action is always NT(S), use this to initialize stack
        action_first = actions[0]
        self._push_to_stack(stack, actions_embedding, 0, batch_index, action_first)
        open_non_terminals_count = 1
        actions_log_probs = torch.zeros((max_sequence_length - 1,), dtype=torch.float, device=self._device, requires_grad=False)
        for action_counter in range(1, sequence_length):
            action_index = action_counter - 1
            action = actions[action_counter]
            representation = self._representation(action_history, stack, token_buffer, action_index, token_index, batch_index)
            valid_actions, action2index = self._action_set.valid_actions(token_buffer, token_counter, stack, open_non_terminals_count)
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            logits_base = self._representation2logits(representation)
            logits_base_valid = logits_base[:, :, valid_actions]
            log_prob_base_valid = self._logits2log_prob(logits_base_valid)
            # get log probability of action
            action_type = action.type()
            if action_type == ACTION_REDUCE_TYPE:
                reduce_log_prob = log_prob_base_valid[:, :, action2index[ACTION_REDUCE_INDEX]]
                action_log_prob = self._reduce(reduce_log_prob, stack)
                open_non_terminals_count -= 1
            elif action_type == ACTION_SHIFT_TYPE:
                shift_log_prob = log_prob_base_valid[:, :, action2index[ACTION_SHIFT_INDEX]]
                action_log_prob = self._shift(shift_log_prob, stack, token_buffer, token_index, batch_index, action)
                token_index = min(token_index + 1, sequence_tokens_count - 1)
                token_counter += 1
            elif action_type == ACTION_GENERATE_TYPE:
                generate_log_prob = log_prob_base_valid[:, :, action2index[ACTION_GENERATE_INDEX]]
                push_args = tokens_embedding, token_index, batch_index, action
                action_log_prob = self._generate(generate_log_prob, stack, representation, *push_args)
                token_index = min(token_index + 1, sequence_tokens_count - 1)
                token_counter += 1
            elif action_type == ACTION_NON_TERMINAL_TYPE:
                non_terminal_log_prob = log_prob_base_valid[:, :, action2index[ACTION_NON_TERMINAL_INDEX]]
                action_log_prob = self._non_terminal(non_terminal_log_prob, stack, representation, action)
                open_non_terminals_count += 1
            else:
                raise Exception(f'Unknown action: {action.type}')
            actions_log_probs[action_index] = action_log_prob
        return actions_log_probs

    def _push_to_stack(self, stack, embeddings, item_index, batch_index, action):
        action_embedding = embeddings[item_index, batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
        return stack.push(action_embedding, action)

    def _reduce(self, base_reduce_log_prob, stack):
        popped_items = []
        action = None
        while action is None or action.type() != ACTION_NON_TERMINAL_TYPE:
            state, action = stack.pop()
            popped_items.append(state)
        popped_tensor = torch.cat(popped_items, dim=0)
        action.close()
        non_terminal_embedding, _ = self._get_non_terminal_embedding(self._non_terminal_compose_embedding, action)
        composed = self._composer(non_terminal_embedding, popped_tensor)
        stack.push(composed, action)
        return base_reduce_log_prob

    def _shift(self, base_shift_log_prob, stack, token_buffer, token_index, batch_index, action):
        embedding = token_buffer.get(token_index, batch_index)
        stack.push(embedding, action)
        return base_shift_log_prob

    def _generate(self, base_generate_log_prob, stack, representation, tokens_embedding, token_index, batch_index, action):
        self._push_to_stack(stack, tokens_embedding, token_index, batch_index, action)
        token_log_prob = self._token_distribution.log_prob(representation, action.argument)
        return base_generate_log_prob + token_log_prob

    def _non_terminal(self, base_non_terminal_log_prob, stack, representation, action):
        non_terminal_embedding, argument_index = self._get_non_terminal_embedding(self._non_terminal_embedding, action)
        stack.push(non_terminal_embedding, action)
        non_terminal_logits = self._representation2non_terminal_logits(representation)
        non_terminal_log_probs = self._logits2log_prob(non_terminal_logits)
        conditional_non_terminal_log_prob = non_terminal_log_probs[:, :, argument_index]
        action_log_prob = base_non_terminal_log_prob + conditional_non_terminal_log_prob
        return action_log_prob

    def _get_non_terminal_embedding(self, embeddings, action):
        argument_index = action.argument_index_as_tensor()
        non_terminal_embedding = embeddings(argument_index).unsqueeze(dim=0).unsqueeze(dim=0)
        return non_terminal_embedding, argument_index
