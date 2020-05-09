from app.composers.composer import Composer
from app.dropout.variational import variational_dropout_mask
from app.utils import batched_index_select, padded_reverse
import torch
from torch import nn

class BiRNNComposer(Composer):

    def __init__(self, device, rnn_forward, rnn_backward, stack_hidden_size, stack_input_size, dropout, dropout_type):
        """
        :type device: torch.device
        :type rnn_forward: app.rnn.rnn.RNN
        :type rnn_backward: app.rnn.rnn.RNN
        :type stack_hidden_size: int
        :type stack_input_size: int
        :type dropout: float
        :type dropout_type: str
        """
        super().__init__()
        self.device = device
        self.rnn_forward = rnn_forward
        self.rnn_backward = rnn_backward
        self.affine = nn.Linear(in_features=2 * stack_hidden_size, out_features=stack_input_size, bias=True)
        self.activation = nn.ReLU()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_type = dropout_type
        self.use_normal_dropout = dropout is not None and dropout_type == 'normal'
        self.use_variational = dropout is not None and dropout_type == 'variational'
        self.stack_input_size = stack_input_size

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
        if self.use_normal_dropout:
            output = self.dropout(output)
        elif self.use_variational and self.training:
            batch_size = nt_embedding.size(1)
            output = output * self.dropout_mask.expand(1, batch_size, -1)
        return output

    def input2output(self, rnn, nt_embedding, items, lengths):
        batch_size = nt_embedding.size(1)
        initial_state = rnn.initial_state(batch_size)
        _, state = rnn(nt_embedding, initial_state)
        outputs, _ = rnn(items, state)
        output = batched_index_select(outputs, lengths - 1)
        return output

    def reset(self, batch_size):
        self.rnn_forward.reset(batch_size)
        self.rnn_backward.reset(batch_size)
        if self.use_variational and self.training:
            self.dropout_mask = variational_dropout_mask((1, 1, self.stack_input_size), self.dropout_p, device=self.device)

    def __str__(self):
        large_ind = ' ' * 4
        small_ind = ' ' * 2
        base_args = f'\n{large_ind}forward={self.rnn_forward},\n{large_ind}backward={self.rnn_backward}'
        if self.dropout_p is None:
            return f'BiRNN({base_args}\n{small_ind})'
        else:
            return f'BiRNN({base_args},\n{large_ind}dropout={self.dropout_p}\n{large_ind}dropout_type={self.dropout_type}\n{small_ind})'
