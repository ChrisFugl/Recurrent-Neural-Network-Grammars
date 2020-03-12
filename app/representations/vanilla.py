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
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, action_history, stack, token_buffer):
        """
        :type action_history: torch.Tensor
        :type stack: torch.Tensor
        :type token_buffer: torch.Tensor
        :rtype: torch.Tensor
        """
        embeddings = [
            self._dropout(self._pick_last(action_history)),
            self._dropout(self._pick_last(stack)),
            self._dropout(self._pick_last(token_buffer)),
        ]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        feedforward_input = torch.cat(embeddings, dim=2)
        output = self._feedforward(feedforward_input)
        output = self._activation(output)
        return output

    def _pick_last(self, embeddings):
        length, batch_size, hidden_size = embeddings.shape
        last = embeddings[length - 1, :, :].view(1, batch_size, hidden_size)
        return last

    def __str__(self):
        return f'Vanilla(size={self._representation_size})'
