from app.composers.composer import Composer
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class BiRNNComposer(Composer):

    def __init__(self, birnn, stack_hidden_size, stack_input_size):
        """
        :type birnn: app.rnn.rnn.RNN
        :type stack_hidden_size: int
        :type stack_input_size: int
        """
        super().__init__()
        self.birnn = birnn
        self.affine = nn.Linear(in_features=2 * stack_hidden_size, out_features=stack_input_size, bias=True)
        self.activation = nn.ReLU()

    def forward(self, non_terminal_embedding, popped_stack_items, lengths):
        """
        :type non_terminal_embedding: torch.Tensor
        :type popped_stack_items: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        tensors = torch.cat((non_terminal_embedding, popped_stack_items), dim=0)
        lengths_cat = lengths + 1
        packed = pack_padded_sequence(tensors, lengths_cat, enforce_sorted=False)
        _, batch_size, _ = tensors.shape
        initial_state = self.birnn.initial_state(batch_size)
        _, (hidden_state, _) = self.birnn(packed, initial_state)
        num_layers = hidden_state.size(0) // 2
        hidden_size = hidden_state.size(2)
        unpacked_state = hidden_state.view(num_layers, 2, batch_size, hidden_size)
        last_layer_states = unpacked_state[-1]
        forward_state = last_layer_states[0]
        backward_state = last_layer_states[1]
        affine_input = torch.cat((forward_state, backward_state), dim=1).unsqueeze(dim=0)
        output = self.activation(self.affine(affine_input))
        return output

    def __str__(self):
        return f'BiRNN(birnn={self.birnn})'
