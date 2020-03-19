from app.composers.composer import Composer
from app.constants import PAD_INDEX
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiRNNComposer(Composer):

    def __init__(self, birnn, output_size):
        """
        :type birnn: app.rnn.rnn.RNN
        :type birnn_output_size: int
        """
        super().__init__()
        self._birnn = birnn
        self._affine = nn.Linear(in_features=2 * output_size, out_features=output_size, bias=True)
        self._activation = nn.ReLU()

    def forward(self, non_terminal_embedding, popped_stack_items, lengths):
        """
        :type non_terminal_embedding: torch.Tensor
        :type popped_stack_items: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        tensors = torch.cat((non_terminal_embedding, popped_stack_items), dim=0)
        packed = pack_padded_sequence(tensors, lengths, enforce_sorted=False)
        _, batch_size, _ = tensors.shape
        state = self._birnn.initial_state(batch_size)
        packed_output, _ = self._birnn(packed, state)
        unpacked_output, _ = pad_packed_sequence(packed_output, padding_value=PAD_INDEX)
        affine_input = self._pick_last(unpacked_output, lengths)
        output = self._activation(self._affine(affine_input))
        return output

    def _pick_last(self, values, lengths):
        _, batch_size, hidden_size = values.shape
        last_index = lengths - 1
        top = last_index.view(1, batch_size, 1).expand(1, batch_size, hidden_size)
        last = torch.gather(values, 0, top).view(1, batch_size, hidden_size)
        return last

    def __str__(self):
        return f'BiRNN(birnn={self._birnn})'
