from app.representations.attention import Attention
from app.representations.representation import Representation
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class AttentiveRepresentation(Representation):

    def __init__(self, device, embedding_size, representation_size, dropout_type, dropout):
        """
        :type device: torch.device
        :type embedding_size: int
        :type representation_size: int
        :type dropout_type: str
        :type dropout: float
        """
        super().__init__()
        self.representation_size = representation_size
        self.history = Attention(device, embedding_size, dropout, dropout_type)
        self.buffer = Attention(device, embedding_size, dropout, dropout_type)
        self.stack = Attention(device, embedding_size, dropout, dropout_type)
        # token, stack, action
        input_size = 3 * embedding_size
        self.embedding2representation = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout_p = dropout
        self.dropout_type = dropout_type

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
        his_embedding, his_weights = self.history(his, his_lengths)
        stack_embedding, stack_weights = self.stack(stack, stack_lengths)
        buf_embedding, buf_weights = self.buffer(buf, buf_lengths)
        embeddings = [his_embedding, stack_embedding, buf_embedding]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        output = torch.cat(embeddings, dim=2)
        output = self.embedding2representation(output)
        output = self.activation(output)
        info = {'history': his_weights, 'buffer': buf_weights, 'stack': stack_weights}
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
        self.history.reset()
        self.stack.reset()
        self.buffer.reset()

    def __str__(self):
        if self.dropout_p is None:
            return f'Attentive(size={self.representation_size})'
        else:
            return f'Attentive(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
