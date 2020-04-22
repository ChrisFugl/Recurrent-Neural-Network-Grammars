import torch
from torch import nn

class Buffer(nn.Module):

    def __init__(self, device, rnn):
        """
        :type device: torch.device
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.device = device
        self.rnn = rnn

    def contents(self, state):
        """
        :type state: app.models.rnng.buffer.BufferState
        :rtype: torch.Tensor, torch.Tensor
        """
        length = state.sequence_index + 1
        length_tensor = torch.tensor([length], device=self.device, dtype=torch.long)
        output = state.buffer[:length]
        return output, length_tensor

    def initialize(self, inputs, lengths, start_indices):
        """
        :type inputs: torch.Tensor
        :type lengths: torch.Tensor
        :type start_indices: list of int
        :rtype: list of app.models.rnng.buffer.BufferState
        """
        batch_size = len(start_indices)
        initial_rnn_state = self.rnn.initial_state(batch_size)
        batched_buffer, _ = self.rnn(inputs, initial_rnn_state)
        hidden_size = batched_buffer.size(2)
        states = []
        for batch_index, sequence_index in enumerate(start_indices):
            length = lengths[batch_index]
            buffer = batched_buffer[:length, batch_index]
            buffer = buffer.view(length, 1, hidden_size)
            state = BufferState(buffer, batch_index, sequence_index)
            states.append(state)
        return states

    def pop(self, state):
        """
        :type state: app.models.rnng.buffer.BufferState
        """
        next_state = BufferState(state.buffer, state.batch_index, state.sequence_index - 1)
        return next_state

    def push(self, state):
        """
        :type state: app.models.rnng.buffer.BufferState
        """
        next_state = BufferState(state.buffer, state.batch_index, state.sequence_index + 1)
        return next_state

    def top(self, state):
        """
        :type state: app.models.rnng.buffer.BufferState
        """
        hidden_size = state.buffer.size(2)
        output = state.buffer[state.sequence_index]
        output = output.view(1, 1, hidden_size)
        return output

    def __str__(self):
        return f'Buffer(rnn={self.rnn})'

class BufferState:

    def __init__(self, buffer, batch_index, sequence_index):
        """
        :type buffer: torch.Tensor
        :type batch_index: int
        :type sequence_index: int
        """
        self.buffer = buffer
        self.batch_index = batch_index
        self.sequence_index = sequence_index
