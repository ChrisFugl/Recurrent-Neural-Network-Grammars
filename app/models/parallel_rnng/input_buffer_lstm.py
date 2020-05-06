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
        batch_size = lengths.size(0)
        initial_state = self.rnn.initial_state(batch_size)
        buffer, _ = self.rnn(inputs, initial_state)
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
        return f'InputBufferLSTM(rnn={self.rnn})'
