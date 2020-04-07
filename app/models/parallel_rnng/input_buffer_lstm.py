from app.models.parallel_rnng.buffer_lstm import BufferLSTM, BufferState
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

    def inference_initialize(self, inputs, lengths):
        """
        :type inputs: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        buffer, _ = self.lstm(inputs)
        return BufferState(buffer, inputs, lengths)

    def inference_hold_or_pop(self, state, op):
        """
        :type state: app.models.parallel_rnng.buffer_lstm.BufferState
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        next_lengths = state.lengths + op
        return BufferState(state.buffer, state.inputs, next_lengths)

    def inference_top_embeddings(self, state):
        """
        :type state: app.models.parallel_rnng.buffer_lstm.BufferState
        :rtype: torch.Tensor
        """
        batch_size = state.inputs.size(1)
        return batched_index_select(state.inputs, state.lengths - 1).view(batch_size, self.input_size)

    def __str__(self):
        return f'InputBufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
