from app.representations.representation import Representation
from torch import nn

class StackOnlyRepresentation(Representation):
    """
    Stack-only representation used by Kuncoro et al. (2017).
    """

    def __init__(self, embedding_size, representation_size, dropout):
        """
        :type embedding_size: int
        :type representation_size: int
        :type dropout: float
        """
        super().__init__()
        self._representation_size = representation_size
        self._feedforward = nn.Linear(in_features=embedding_size, out_features=representation_size, bias=True)
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
        stack_embedding, _ = stack.top()
        stack_embedding = self._dropout(stack_embedding)
        output = self._feedforward(stack_embedding)
        output = self._activation(output)
        return output

    def __str__(self):
        return f'StackOnly(size={self._representation_size})'
