from app.models.parallel_rnng.buffer_lstm import BufferLSTM
from app.utils import batched_index_select
import torch

class OutputBufferLSTM(BufferLSTM):

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: buffer contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_lengths = self.lengths.max()
        contents = self.buffer[:max_lengths]
        return contents, self.lengths

    def initialize(self, inputs):
        """
        :type inputs: torch.Tensor
        """
        batch_size = inputs.size(1)
        output, _ = self.lstm(inputs)
        self.buffer = output
        self.lengths = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    def hold_or_push(self, op):
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
        return f'OutputBufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
