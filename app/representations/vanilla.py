from app.representations.representation import Representation
import torch
from torch import nn

class VanillaRepresentation(Representation):
    """
    Representation used by Dyer et. al. 2016.
    """

    def __init__(self, embedding_size, representation_size, dropout):
        """
        :type embedding_size: int
        :type representation_size: int
        :type dropout: float
        """
        super().__init__()
        # token, stack, action
        self._representation_size = representation_size
        input_size = 3 * embedding_size
        self._feedforward = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self._activation = nn.Tanh()
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, action_history, stack, token_buffer, action_timestep, token_timestep, batch_index):
        """
        :type action_history: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type token_buffer: app.memories.memory.Memory
        :type action_timestep: int
        :type token_timestep: int
        :type batch_index: int
        :rtype: torch.Tensor
        """
        action_embedding = action_history.get(action_timestep, batch_index)
        stack_embedding, _ = stack.top()
        token_embedding = token_buffer.get(token_timestep, batch_index)
        embeddings = [
            self._dropout(token_embedding),
            self._dropout(stack_embedding),
            self._dropout(action_embedding),
        ]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        feedforward_input = torch.cat(embeddings, dim=2)
        output = self._feedforward(feedforward_input)
        output = self._activation(output)
        return output

    def __str__(self):
        return f'Vanilla(size={self._representation_size})'
