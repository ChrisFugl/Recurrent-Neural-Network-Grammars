from app.dropout.variational import variational_dropout_mask
from app.representations.representation import Representation
from torch import nn

class StackOnlyRepresentation(Representation):
    """
    Stack-only representation used by Kuncoro et al. (2017).
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
        self.feedforward = nn.Linear(in_features=embedding_size, out_features=representation_size, bias=True)
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
        output = stack
        if self.use_normal_dropout:
            output = self.dropout(output)
        output = self.feedforward(output)
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
        return False

    def reset(self, batch_size):
        if self.use_variational_dropout and self.training:
            self.dropout_mask = variational_dropout_mask((1, 1, self.representation_size), self.dropout_p, device=self.device)

    def __str__(self):
        if self.dropout_p is None:
            return f'StackOnly(size={self.representation_size})'
        else:
            return f'StackOnly(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
