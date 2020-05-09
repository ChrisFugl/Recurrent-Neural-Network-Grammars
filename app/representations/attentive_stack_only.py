from app.representations.attention import Attention
from app.representations.representation import Representation
from torch import nn

MASKED_SCORE_VALUE = - 1e10

class AttentiveStackOnlyRepresentation(Representation):

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
        self.stack = Attention(device, embedding_size, dropout, dropout_type)
        self.embedding2representation = nn.Linear(in_features=embedding_size, out_features=representation_size, bias=True)
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
        stack_embedding, stack_weights = self.stack(stack, stack_lengths)
        # concatenate along last dimension, as inputs have shape S, B, H (sequence length, batch size, hidden size)
        output = self.embedding2representation(stack_embedding)
        output = self.activation(output)
        info = {'stack': stack_weights}
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
        return False

    def reset(self, batch_size):
        self.stack.reset()

    def __str__(self):
        if self.dropout_p is None:
            return f'AttentiveStackOnly(size={self.representation_size})'
        else:
            return f'AttentiveStackOnly(size={self.representation_size}, dropout={self.dropout_p}, dropout_type={self.dropout_type})'
