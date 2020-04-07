from app.models.parallel_rnng.buffer_lstm import BufferLSTM, BufferState
from app.utils import batched_index_select
import torch

class OutputBufferLSTM(BufferLSTM):

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: buffer contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_pos = self.pos.max()
        contents = self.buffer[:max_pos]
        return contents, self.pos

    def initialize(self, inputs):
        """
        :type inputs: torch.Tensor
        """
        batch_size = inputs.size(1)
        output, _ = self.lstm(inputs)
        self.buffer = output
        self.pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    def hold_or_push(self, op):
        """
        :type op: torch.Tensor
        """
        self.pos = self.pos + op

    def inference_initialize(self, inputs):
        """
        :type inputs: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.BufferState
        """
        batch_size = inputs.size(1)
        buffer, _ = self.lstm(inputs)
        lengths = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        return BufferState(buffer, inputs, lengths)

    def inference_hold_or_push(self, state, op):
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
        return f'OutputBufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
