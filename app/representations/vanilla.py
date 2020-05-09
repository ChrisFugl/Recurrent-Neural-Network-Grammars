from app.dropout.variational import variational_dropout_mask
from app.representations.representation import Representation
import torch
from torch import nn

class VanillaRepresentation(Representation):
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
        # token, stack, action
        input_size = 3 * embedding_size
        self.affine = nn.Linear(in_features=input_size, out_features=representation_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_type = dropout_type
        self.dropout_p = dropout
        self.use_normal_dropout = dropout is not None and dropout_type == 'normal'
        self.use_variational_dropout = dropout is not None and dropout_type == 'variational'

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
        if self.use_normal_dropout:
            history_embedding = self.dropout(action_history)
            stack_embedding = self.dropout(stack)
            buffer_embedding = self.dropout(token_buffer)
        else:
            history_embedding = action_history
            stack_embedding = stack
            buffer_embedding = token_buffer
        embeddings = [history_embedding, stack_embedding, buffer_embedding]
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        affine_input = torch.cat(embeddings, dim=2)
        output = self.affine(affine_input)
        output = self.activation(output)
        if self.use_variational_dropout and self.training:
            batch_size = output.size(1)
            output = output * self.dropout_mask.expand(1, batch_size, -1)
        return output, {}

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
        if self.use_variational_dropout and self.training:
            self.dropout_mask = variational_dropout_mask((1, 1, self.representation_size), self.dropout_p, device=self.device)

    def __str__(self):
        if self.dropout_p is None:
            return f'Vanilla(size={self.representation_size})'
        else:
            return f'Vanilla(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
