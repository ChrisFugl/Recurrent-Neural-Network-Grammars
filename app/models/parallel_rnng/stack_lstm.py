from app.models.parallel_rnng.memory_lstm import MemoryLSTM
import torch

class StackLSTM(MemoryLSTM):

    def __init__(self, device, stack_size, input_size, hidden_size, num_layers, bias, dropout):
        """
        :type device: torch.device
        :type stack_size: int
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        """
        super().__init__(device, input_size, hidden_size, num_layers, bias, dropout)
        self._stack_size = stack_size

    def contents(self, stack):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :returns: stack contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_index = torch.max(stack.indices)
        last_layer_state = stack.hidden_state[:, self._num_layers - 1, :, :]
        # do not include the initial state
        contents = last_layer_state[1:max_index + 1, :, :]
        return contents, stack.indices

    def initialize(self, batch_size):
        """
        :type batch_size: int
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        shape = (self._stack_size + 1, self._num_layers, batch_size, self._hidden_size)
        hidden_state = torch.zeros(shape, device=self._device, dtype=torch.float)
        cell_state = torch.zeros(shape, device=self._device, dtype=torch.float)
        indices = torch.tensor([0] * batch_size, device=self._device, dtype=torch.long)
        return Stack(hidden_state, cell_state, indices)

    def hold_or_pop(self, stack, op):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :param op: tensor, (batch size), hold = 0, pop = -1
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack, torch.Tensor
        """
        output = self.top(stack)
        next_indices = stack.indices + op
        return Stack(stack.hidden_state, stack.cell_state, next_indices), output

    def hold_or_push(self, stack, input, op):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :param input: tensor, (sequence length, batch size, input size)
        :type input: torch.Tensor
        :param op: tensor, (batch size), push = 1, hold = 0
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        batch_size = input.size(1)
        top = stack.indices.view(1, 1, batch_size, 1).expand(1, self._num_layers, batch_size, self._hidden_size)
        top_hidden_state = torch.gather(stack.hidden_state, 0, top).squeeze()
        top_cell_state = torch.gather(stack.cell_state, 0, top).squeeze()
        top_state = (top_hidden_state, top_cell_state)
        _, (hidden_state, cell_state) = self._lstm(input, top_state)
        next_hidden_state = stack.hidden_state.clone()
        next_cell_state = stack.cell_state.clone()
        next_hidden_state[stack.indices + 1, :, :, :] = hidden_state
        next_cell_state[stack.indices + 1, :, :, :] = cell_state
        next_indices = stack.indices + op
        return Stack(next_hidden_state, next_cell_state, next_indices)

    def top(self, stack):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :rtype: torch.Tensor
        """
        batch_size = stack.hidden_state.size(2)
        last_layer_state = stack.hidden_state[:, self._num_layers - 1, :, :]
        top = stack.indices.view(1, batch_size, 1).expand(1, batch_size, self._hidden_size)
        output = torch.gather(last_layer_state, 0, top).squeeze()
        return output

    def __str__(self):
        return f'StackLSTM(stack_size={self._stack_size}, input_size={self._input_size}, hidden_size={self._hidden_size}, num_layers={self._num_layers})'

class Stack:

    def __init__(self, hidden_state, cell_state, indices):
        """
        :type hidden_state: torch.Tensor
        :type cell_state: torch.Tensor
        :type indices: torch.Tensor
        """
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.indices = indices
