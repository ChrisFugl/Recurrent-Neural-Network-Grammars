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
        self.representation_size = representation_size
        # token, stack, action
        input_size = 3 * embedding_size
        self.feedforward = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
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
        # print(stack.shape, stack_lengths)
        # print(token_buffer.shape, token_buffer_lengths)
        # print(action_history.shape, action_history_lengths)
        embeddings = [
            self.dropout(self.pick_last(action_history, action_history_lengths)),
            self.dropout(self.pick_last(stack, stack_lengths)),
            self.dropout(self.pick_last(token_buffer, token_buffer_lengths)),
        ]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        feedforward_input = torch.cat(embeddings, dim=2)
        output = self.feedforward(feedforward_input)
        output = self.activation(output)
        return output

    def pick_last(self, embeddings, lengths):
        _, batch_size, hidden_size = embeddings.shape
        last_index = lengths - 1
        top = last_index.view(1, batch_size, 1).expand(1, batch_size, hidden_size)
        last = torch.gather(embeddings, 0, top).view(1, batch_size, hidden_size)
        return last

    def __str__(self):
        return f'Vanilla(size={self.representation_size})'
