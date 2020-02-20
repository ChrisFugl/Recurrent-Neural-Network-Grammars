from app.data.actions.generative import Generative
from app.models.model import Model
import torch
from torch import nn

# base actions: REDUCE, GENERATE, NON_TERMINAL
ACTIONS_COUNT = 3

class RNNG(Model):

    # TODO: how to handle generative vs. discriminative
    # TODO: composition function

    def __init__(self, device, action_embedding, token_embedding, action_history, token_buffer, stack, representation, representation_size):
        """
        :type device: torch.device
        :type action_embedding: torch.Embedding
        :type token_embedding: torch.Embedding
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type representation: app.representations.representation.Representation
        :type representation_size: int
        """
        super().__init__()
        self._device = device
        self._action_embedding = action_embedding
        self._token_embedding = token_embedding
        self._action_history = action_history
        self._token_buffer = token_buffer
        self._stack = stack
        self._representation = representation
        self._representation2logits = nn.Linear(in_features=representation_size, out_features=ACTIONS_COUNT, bias=True)
        self._logits2probits = nn.Softmax(dim=2)

        # TODO: determine action set based on wheter task is discriminative or generative
        self._action_set = Generative()

        # TODO: remove this
        self._tmp_action = nn.Linear(in_features=representation_size, out_features=1)
        self._tmp_sigmoid = nn.Sigmoid()

    def forward(self, tokens, actions):
        """
        :type tokens: torch.Tensor, list of int, list of list of str
        :type actions: torch.Tensor, list of int, list of list of str
        :rtype: torch.Tensor
        """
        return self.likelihood(tokens, actions)

    def likelihood(self, tokens, actions):
        """
        Compute likelihood of each sentence/tree in a batch.

        :type tokens: torch.Tensor, list of int, list of list of str
        :type actions: torch.Tensor, list of int, list of list of str
        :rtype: torch.Tensor
        """
        self._reset()

        tokens_tensor, tokens_lengths, tokens_strings = tokens
        actions_tensor, actions_lengths, actions_strings = actions
        actions_embedding = self._action_embedding(actions_tensor)
        tokens_embedding = self._token_embedding(tokens_tensor)

        # add to action history and token buffer
        actions_after_first_embedding = actions_embedding[1:]
        self._action_history.add(actions_after_first_embedding)
        self._token_buffer.add(tokens_embedding)

        # stack operations
        sequence_length, batch_size = actions_tensor.shape
        batch_action_probabilities = torch.empty(
            (sequence_length - 1, batch_size),
            dtype=torch.float,
            device=self._device,
            requires_grad=False
        )
        for batch_index in range(batch_size):
            self._stack.reset()
            # first action is always NT(S), use this to initialize stack
            self._push_to_stack(actions_embedding, 0, batch_index)
            sequence_actions_strings = actions_strings[batch_index]
            sequence_tokens_length = len(tokens_strings[batch_index])
            sequence_length = actions_lengths[batch_index]
            token_index = 0
            for action_counter in range(1, sequence_length):
                action_index = action_counter - 1
                action_string = sequence_actions_strings[action_counter]
                action = self._action_set.string2action(action_string)
                representation = self._representation(
                    self._action_history,
                    self._stack,
                    self._token_buffer,
                    action_index,
                    token_index,
                    batch_index
                )
                output_base_logits = self._representation2logits(representation)
                output_base_probits = self._logits2probits(output_base_logits)
                # get probability of action
                # TODO: only consider legal actions
                if action.type == Generative.REDUCE:
                    action_probability = self._reduce(output_base_probits)
                elif action.type == Generative.GEN:
                    action_probability = self._generate(output_base_probits, representation, action)
                    token_index = min(token_index + 1, sequence_tokens_length - 1)
                else:
                    action_probability = self._non_terminal(output_base_probits, representation, action)
                batch_action_probabilities[action_index, batch_index] = action_probability

        return batch_action_probabilities

    def next_state(self, previous_state):
        """
        Compute next state and action probabilities given the previous state.

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

    def _reset(self):
        self._action_history.reset()
        self._token_buffer.reset()
        self._stack.reset()

    def _push_to_stack(self, embeddings, action_index, batch_index):
        action = embeddings[action_index, batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
        return self._stack.push(action)

    def _reduce(self, output_base_probits):
        # TODO: reduce
        return output_base_probits[:, :, 0].unsqueeze(dim=0)

    def _generate(self, output_base_probits, representation, action):
        # TODO: generate
        return self._tmp_sigmoid(self._tmp_action(representation))

    def _non_terminal(self, output_base_probits, representation, action):
        # TODO: non terminal
        return self._tmp_sigmoid(self._tmp_action(representation))
