from app.dropout.variational import variational_dropout_mask
from app.representations.attention import Attention
from app.representations.representation import Representation
from app.utils import batched_index_select
import torch
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class AttentiveBufferRepresentation(Representation):

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
        self.buffer = Attention(device, embedding_size, dropout, dropout_type)
        # token, stack, action
        input_size = 3 * embedding_size
        self.embedding2representation = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_type = dropout_type
        self.dropout_p = dropout
        self.use_normal_dropout = dropout is not None and dropout_type == 'normal'
        self.use_variational_dropout = dropout is not None and dropout_type == 'variational'

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
        buf_embedding, buf_weights = self.buffer(buf, buf_lengths)
        his_embedding = batched_index_select(his, his_lengths - 1)
        stack_embedding = batched_index_select(stack, stack_lengths - 1)
        if self.use_normal_dropout:
            his_embedding = self.dropout(his_embedding)
            stack_embedding = self.dropout(stack_embedding)
        embeddings = [his_embedding, stack_embedding, buf_embedding]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        output = torch.cat(embeddings, dim=2)
        output = self.embedding2representation(output)
        output = self.activation(output)
        if self.use_variational_dropout and self.training:
            batch_size = output.size(1)
            output = output * self.dropout_mask.expand(1, batch_size, -1)
        info = {'buffer': buf_weights}
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
        self.buffer.reset()
        if self.use_variational_dropout and self.training:
            self.dropout_mask = variational_dropout_mask((1, 1, self.representation_size), self.dropout_p, device=self.device)

    def __str__(self):
        if self.dropout_p is None:
            return f'AttentiveBuffer(size={self.representation_size})'
        else:
            return f'AttentiveBuffer(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
