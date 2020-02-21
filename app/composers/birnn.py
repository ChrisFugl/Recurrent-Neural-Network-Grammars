from app.composers.composer import Composer
import torch
from torch import nn

class BiRNNComposer(Composer):

    def __init__(self, birnn, output_size):
        """
        :type birnn: app.rnn.rnn.RNN
        :type birnn_output_size: int
        :
        """
        super().__init__()
        self._birnn = birnn
        self._affine = nn.Linear(in_features=2 * output_size, out_features=output_size, bias=True)
        self._activation = nn.Tanh()

    def forward(self, non_terminal_embedding, popped_stack_items):
        """
        :type non_terminal_embedding: torch.Tensor
        :type popped_stack_items: torch.Tensor
        :rtype: torch.Tensor
        """
        tensors = torch.cat((non_terminal_embedding, popped_stack_items), dim=0)
        sequence_length, _, _ = tensors.shape
        birnn_hidden_state = self._birnn.initial_state()
        birnn_output, _ = self._birnn(tensors, birnn_hidden_state)
        affine_input = birnn_output[sequence_length - 1].unsqueeze(dim=0)
        output = self._activation(self._affine(affine_input))
        return output
