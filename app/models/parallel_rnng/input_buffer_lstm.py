from app.models.parallel_rnng.buffer_lstm import BufferLSTM
from app.utils import batched_index_select
import torch

class InputBufferLSTM(BufferLSTM):

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: buffer contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_length = torch.max(self.lengths)
        contents = self.buffer[:max_length]
        return contents, self.lengths

    def initialize(self, inputs, lengths):
        """
        :type inputs: torch.Tensor
        :type lengths: torch.Tensor
        """
        buffer, _ = self.lstm(inputs)
        self.buffer = buffer
        self.lengths = lengths

    def hold_or_pop(self, op):
        """
        :type op: torch.Tensor
        """
        self.lengths = self.lengths + op

    def top(self):
        """
        :rtype: torch.Tensor
        """
        return batched_index_select(self.buffer, self.lengths - 1)

    def __str__(self):
        return f'InputBufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
