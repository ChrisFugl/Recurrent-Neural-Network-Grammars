from app.composers.composer import Composer
from app.utils import batched_index_select, padded_reverse
import torch
from torch import nn

class BiRNNComposer(Composer):

    def __init__(self, rnn_forward, rnn_backward, stack_hidden_size, stack_input_size, dropout):
        """
        :type rnn_forward: app.rnn.rnn.RNN
        :type rnn_backward: app.rnn.rnn.RNN
        :type stack_hidden_size: int
        :type stack_input_size: int
        :type dropout: float
        """
        super().__init__()
        self.rnn_forward = rnn_forward
        self.rnn_backward = rnn_backward
        self.affine = nn.Linear(in_features=2 * stack_hidden_size, out_features=stack_input_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nt_embedding, popped_items, popped_lengths):
        """
        :type nt_embedding: torch.Tensor
        :type popped_items: list of torch.Tensor
        :type popped_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        backward_popped_items = padded_reverse(popped_items, popped_lengths)
        forward_output = self.input2output(self.rnn_forward, nt_embedding, popped_items, popped_lengths)
        backward_output = self.input2output(self.rnn_backward, nt_embedding, backward_popped_items, popped_lengths)
        affine_input = torch.cat((forward_output, backward_output), dim=2)
        output = self.affine(affine_input)
        output = self.activation(output)
        return output

    def input2output(self, rnn, nt_embedding, items, lengths):
        batch_size = nt_embedding.size(1)
        initial_state = rnn.initial_state(batch_size)
        _, state = rnn(nt_embedding, initial_state)
        outputs, _ = rnn(items, state)
        output = batched_index_select(outputs, lengths - 1)
        output = self.dropout(output)
        return output

    def __str__(self):
        return f'BiRNN(rnn_forward={self.rnn_forward}, rnn_backward={self.rnn_backward})'
