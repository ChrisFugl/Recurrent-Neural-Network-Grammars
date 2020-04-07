from app.composers.composer import Composer
from app.utils import padded_reverse
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class BiRNNComposer(Composer):

    def __init__(self, rnn_forward, rnn_backward, stack_hidden_size, stack_input_size):
        """
        :type rnn_forward: app.rnn.rnn.RNN
        :type rnn_backward: app.rnn.rnn.RNN
        :type stack_hidden_size: int
        :type stack_input_size: int
        """
        super().__init__()
        self.rnn_forward = rnn_forward
        self.rnn_backward = rnn_backward
        self.affine = nn.Linear(in_features=2 * stack_hidden_size, out_features=stack_input_size, bias=True)
        self.activation = nn.ReLU()

    def forward(self, nt_embedding, popped_items, popped_lengths):
        """
        :type nt_embedding: torch.Tensor
        :type popped_items: torch.Tensor
        :type popped_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        # plus 1 to account for non-terminal embedding
        lengths = popped_lengths + 1
        backward_popped_items = padded_reverse(popped_items, popped_lengths)
        forward_output = self.input2output(self.rnn_forward, nt_embedding, popped_items, lengths)
        backward_output = self.input2output(self.rnn_backward, nt_embedding, backward_popped_items, lengths)
        affine_input = torch.cat((forward_output, backward_output), dim=2)
        output = self.activation(self.affine(affine_input))
        return output

    def input2output(self, rnn, nt_embedding, items, lengths):
        batch_size = items.size(1)
        rnn_input = torch.cat((nt_embedding, items), dim=0)
        packed_input = pack_padded_sequence(rnn_input, lengths, enforce_sorted=False)
        initial_state = rnn.initial_state(batch_size)
        _, (hidden_state, _) = rnn(packed_input, initial_state) # num layers, batch size, hidden size
        last_layer_state = hidden_state[-1]
        output = last_layer_state.unsqueeze(dim=0)
        return output

    def __str__(self):
        return f'BiRNN(rnn_forward={self.rnn_forward}, rnn_backward={self.rnn_backward})'
