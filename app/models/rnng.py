from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX, ACTION_NON_TERMINAL_INDEX,
    ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE
)
from app.models.model import Model
import torch
from torch import nn

# base actions: (REDUCE, GENERATE, NON_TERMINAL) or (REDUCE, SHIFT, NON_TERMINAL)
ACTIONS_COUNT = 3

class RNNG(Model):

    # TODO: composition function

    def __init__(
        self, device,
        action_embedding, token_embedding,
        non_terminal_embedding, non_terminal_compose_embedding,
        action_history, token_buffer, stack,
        representation, representation_size,
        composer,
        non_terminal_count,
        action_set
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
        :type non_terminal_count: int
        :type action_set: app.data.action_set.ActionSet
        """
        super().__init__()
        self._device = device
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
        self._reset()

        tokens_tensor, tokens_lengths, _ = batch_tokens
        actions_tensor, actions_lengths, actions = batch_actions
        actions_embedding = self._action_embedding(actions_tensor)
        tokens_embedding = self._token_embedding(tokens_tensor)

        # add to action history and token buffer
        actions_after_first_embedding = actions_embedding[1:]
        self._action_history.add(actions_after_first_embedding)
        self._token_buffer.add(tokens_embedding)

        # stack operations
        max_sequence_length, batch_size = actions_tensor.shape
        action_log_probs_shape = (max_sequence_length - 1, batch_size)
        batch_action_log_probs = torch.empty(action_log_probs_shape, dtype=torch.float, device=self._device, requires_grad=False)
        for batch_index in range(batch_size):
            sequence_tokens_count = tokens_lengths[batch_index]
            batch_action_log_probs[:, batch_index] = self._actions_log_probs(
            max_sequence_length, batch_index,
            actions_embedding, tokens_embedding,
            actions[batch_index], actions_lengths,
            sequence_tokens_count
        )
        return batch_action_log_probs

    def next_state(self, previous_state):
        """
        Compute next state and action probs given the previous state.

        :returns: next state, next_actions
        """
        # TODO
        raise NotImplementedError('method must be implemented by a subclass')

    def parse(self, tokens):
        """
        Generate a parse of a sentence.

        :type tokens: torch.Tensor
        """
        # TODO
        raise NotImplementedError('method must be implemented by a subclass')

    def _actions_log_probs(self, max_sequence_length, batch_index, actions_embedding, tokens_embedding, actions, actions_lengths, sequence_tokens_count):
        self._stack.reset()
        sequence_length = actions_lengths[batch_index]
        token_index = 0
        token_counter = 0
        # first action is always NT(S), use this to initialize stack
        action_first = actions[0]
        self._push_to_stack(actions_embedding, 0, batch_index, action_first)
        open_non_terminals_count = 1
        actions_log_probs = torch.zeros((max_sequence_length - 1,), dtype=torch.float, device=self._device, requires_grad=False)
        for action_counter in range(1, sequence_length):
            action_index = action_counter - 1
            action = actions[action_counter]
            representation = self._representation(
                self._action_history, self._stack, self._token_buffer,
                action_index, token_index, batch_index
            )
            valid_actions, action2index = self._action_set.valid_actions(
                self._token_buffer, token_counter,
                self._stack, open_non_terminals_count
            )
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            logits_base = self._representation2logits(representation)
            logits_base_valid = logits_base[:, :, valid_actions]
            log_prob_base_valid = self._logits2log_prob(logits_base_valid)
            # get log probability of action
            action_type = action.type()
            if action_type == ACTION_REDUCE_TYPE:
                reduce_log_prob = log_prob_base_valid[:, :, action2index[ACTION_REDUCE_INDEX]]
                action_log_prob = self._reduce(reduce_log_prob)
                open_non_terminals_count -= 1
            elif action_type == ACTION_SHIFT_TYPE:
                shift_log_prob = log_prob_base_valid[:, :, action2index[ACTION_SHIFT_INDEX]]
                action_log_prob = self._shift(shift_log_prob, token_index, batch_index, action)
                token_index = min(token_index + 1, sequence_tokens_count - 1)
                token_counter += 1
            elif action_type == ACTION_GENERATE_TYPE:
                generate_log_prob = log_prob_base_valid[:, :, action2index[ACTION_GENERATE_INDEX]]
                push_args = tokens_embedding, token_index, batch_index, action
                action_log_prob = self._generate(generate_log_prob, representation, *push_args)
                token_index = min(token_index + 1, sequence_tokens_count - 1)
                token_counter += 1
            elif action_type == ACTION_NON_TERMINAL_TYPE:
                non_terminal_log_prob = log_prob_base_valid[:, :, action2index[ACTION_NON_TERMINAL_INDEX]]
                action_log_prob = self._non_terminal(non_terminal_log_prob, representation, action)
                open_non_terminals_count += 1
            else:
                raise Exception(f'Unknown action: {action.type}')
            actions_log_probs[action_index] = action_log_prob
        return actions_log_probs

    def _reset(self):
        self._action_history.reset()
        self._token_buffer.reset()
        self._stack.reset()

    def _push_to_stack(self, embeddings, item_index, batch_index, action):
        action_embedding = embeddings[item_index, batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
        return self._stack.push(action_embedding, action)

    def _reduce(self, base_reduce_log_prob):
        popped_items = []
        action = None
        while action is None or action.type() != ACTION_NON_TERMINAL_TYPE:
            state, action = self._stack.pop()
            popped_items.append(state)
        popped_tensor = torch.cat(popped_items, dim=0)
        action.close()
        non_terminal_embedding, _ = self._get_non_terminal_embedding(action)
        composed = self._composer(non_terminal_embedding, popped_tensor)
        self._stack.push(composed, action)
        return base_reduce_log_prob

    def _shift(self, base_shift_log_prob, token_index, batch_index, action):
        embedding = self._token_buffer.get(token_index, batch_index)
        self._stack.push(embedding, action)
        return base_shift_log_prob

    def _generate(self, base_generate_log_prob, representation, tokens_embedding, token_index, batch_index, action):
        self._push_to_stack(tokens_embedding, token_index, batch_index, action)
        # TODO: compute clustered conditional log probability
        return base_generate_log_prob

    def _non_terminal(self, base_non_terminal_log_prob, representation, action):
        non_terminal_embedding, argument_index = self._get_non_terminal_embedding(action)
        self._stack.push(non_terminal_embedding, action)
        non_terminal_logits = self._representation2non_terminal_logits(representation)
        non_terminal_log_probs = self._logits2log_prob(non_terminal_logits)
        conditional_non_terminal_log_prob = non_terminal_log_probs[:, :, argument_index]
        action_log_prob = base_non_terminal_log_prob + conditional_non_terminal_log_prob
        return action_log_prob

    def _get_non_terminal_embedding(self, action):
        argument_index = action.argument_index_as_tensor()
        non_terminal_embedding = self._non_terminal_embedding(argument_index).unsqueeze(dim=0).unsqueeze(dim=0)
        return non_terminal_embedding, argument_index
