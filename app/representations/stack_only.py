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
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, action_history, stack, token_buffer):
        """
        :type action_history: torch.Tensor
        :type stack: torch.Tensor
        :type token_buffer: torch.Tensor
        :rtype: torch.Tensor
        """
        output = self._pick_last(stack)
        output = self._dropout(output)
        output = self._feedforward(output)
        output = self._activation(output)
        return output

    def _pick_last(self, embeddings):
        length, batch_size, hidden_size = embeddings.shape
        last = embeddings[length - 1, :, :].view(1, batch_size, hidden_size)
        return last

    def __str__(self):
        return f'StackOnly(size={self._representation_size})'
