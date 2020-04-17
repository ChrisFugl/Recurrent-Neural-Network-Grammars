from app.representations.representation import Representation
from torch import nn

class BufferOnlyRepresentation(Representation):
    """
    Buffer-only representation.
    """

    def __init__(self, embedding_size, representation_size, dropout):
        """
        :type embedding_size: int
        :type representation_size: int
        :type dropout: float
        """
        super().__init__()
        self.representation_size = representation_size
        self.feedforward = nn.Linear(in_features=embedding_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, action_history, action_history_lengths, stack, stack_lengths, token_buffer, token_buffer_lengths):
        """
        :type action_history: torch.Tensor
        :type action_history_lengths: torch.Tensor
        :type stack: torch.Tensor
        :type stack_lengths: torch.Tensor
        :type token_buffer: torch.Tensor
        :type token_buffer_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        output = token_buffer
        output = self.dropout(output)
        output = self.feedforward(output)
        output = self.activation(output)
        return output

    def top_only(self):
        """
        :rtype: bool
        """
        return True

    def uses_action_history(self):
        """
        :rtype: bool
        """
        return False

    def uses_stack(self):
        """
        :rtype: bool
        """
        return False

    def uses_token_buffer(self):
        """
        :rtype: bool
        """
        return True

    def __str__(self):
        return f'BufferOnly(size={self.representation_size})'
