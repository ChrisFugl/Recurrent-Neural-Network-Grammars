from app.representations.attention import Attention
from app.representations.representation import Representation
import torch
from torch import nn

class WeightedRepresentation(Representation):
    """
    Representation used by Dyer et. al. 2016.
    """

    def __init__(self, device, embedding_size, representation_size, dropout_type, dropout):
        """
        :type device: torch.device
        :type embedding_size: int
        :type representation_size: int
        :type dropout_type: str
        :type dropout: float
        """
        super().__init__()
        self.device = device
        self.representation_size = representation_size
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_type = dropout_type
        self.dropout_p = dropout
        self.structures = Attention(device, embedding_size, dropout, dropout_type)

    def forward(self, action_history, action_history_lengths, stack, stack_lengths, token_buffer, token_buffer_lengths):
        """
        :type action_history: torch.Tensor
        :type action_history_lengths: torch.Tensor
        :type stack: torch.Tensor
        :type stack_lengths: torch.Tensor
        :type token_buffer: torch.Tensor
        :type token_buffer_lengths: torch.Tensor
        :rtype: torch.Tensor, dict
        """
        stack_embedding = stack
        buffer_embedding = token_buffer
        history_embedding = action_history
        embeddings = torch.cat([stack_embedding, buffer_embedding], dim=0)
        batch_size = action_history.size(1)
        lengths = torch.tensor([2 for _ in range(batch_size)], dtype=torch.long, device=self.device)
        output, weights = self.structures(embeddings, lengths, query=history_embedding)
        return output, {'weighted': weights}

    def top_only(self):
        """
        :rtype: bool
        """
        return True

    def uses_action_history(self):
        """
        :rtype: bool
        """
        return True

    def uses_stack(self):
        """
        :rtype: bool
        """
        return True

    def uses_token_buffer(self):
        """
        :rtype: bool
        """
        return True

    def reset(self, batch_size):
        self.structures.reset()

    def __str__(self):
        if self.dropout_p is None:
            return f'Weighted(size={self.representation_size})'
        else:
            return f'Weighted(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
