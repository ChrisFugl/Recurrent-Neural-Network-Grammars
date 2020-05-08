from app.representations.attention import Attention
from app.representations.representation import Representation
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class AttentiveNoHistoryRepresentation(Representation):

    def __init__(self, embedding_size, representation_size, dropout):
        """
        :type embedding_size: int
        :type representation_size: int
        :type dropout: float
        """
        super().__init__()
        self.representation_size = representation_size
        self.buffer = Attention(embedding_size)
        self.stack = Attention(embedding_size)
        # token, stack
        input_size = 2 * embedding_size
        self.embedding2representation = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, his, his_lengths, stack, stack_lengths, buf, buf_lengths):
        """
        :type his: torch.Tensor
        :type his_lengths: torch.Tensor
        :type stack: torch.Tensor
        :type stack_lengths: torch.Tensor
        :type buf: torch.Tensor
        :type buf_lengths: torch.Tensor
        :rtype: torch.Tensor, dict
        """
        stack_embedding, stack_weights = self.embed(self.stack, stack, stack_lengths)
        buf_embedding, buf_weights = self.embed(self.buffer, buf, buf_lengths)
        embeddings = [stack_embedding, buf_embedding]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        output = torch.cat(embeddings, dim=2)
        output = self.embedding2representation(output)
        output = self.activation(output)
        info = {'buffer': buf_weights, 'stack': stack_weights}
        return output, info

    def top_only(self):
        """
        :rtype: bool
        """
        return False

    def uses_action_history(self):
        """
        :rtype: bool
        """
        return False

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

    def embed(self, attention, inputs, lengths):
        embedding, attention_weights = attention(inputs, lengths)
        output = self.activation(embedding)
        output = self.dropout(output)
        return output, attention_weights

    def __str__(self):
        if self.dropout_p is None:
            return f'AttentiveNoHistory(size={self.representation_size})'
        else:
            return f'AttentiveNoHistory(size={self.representation_size}, dropout={self.dropout_p})'
