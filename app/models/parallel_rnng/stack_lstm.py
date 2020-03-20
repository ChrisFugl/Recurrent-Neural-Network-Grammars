from app.models.parallel_rnng.multi_layer_lstm_cell import MultiLayerLSTMCell
import torch
from torch import nn

class StackLSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = MultiLayerLSTMCell(input_size, hidden_size, num_layers, bias, dropout)

    def contents(self, stack):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :returns: stack contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_index = torch.max(self.pos)
        last_layer_state = self.hidden_stack[:, :, :, self.num_layers - 1]
        # do not include the initial state
        contents = last_layer_state[1:max_index + 1, :, :]
        lengths = self.pos
        return contents, lengths

    def initialize(self, stack_size, batch_size):
        """
        :type stack_size: int
        :type batch_size: int
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        self.pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        shape = (stack_size + 1, batch_size, self.hidden_size, self.num_layers)
        self.hidden_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.cell_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.batch_indices = torch.arange(0, batch_size, device=self.device, dtype=torch.long)
        stack = None
        return stack

    def hold_or_pop(self, stack, op):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :param op: tensor, (batch size), hold = 0, pop = -1
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack, torch.Tensor
        """
        output = self.top(stack)
        self.pos = self.pos + op
        next_stack = None
        return next_stack, output

    def hold_or_push(self, stack, input, op):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :param input: tensor, (sequence length, batch size, input size)
        :type input: torch.Tensor
        :param op: tensor, (batch size), push = 1, hold = 0
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.stack_lstm.Stack
        """
        batch_size = input.size(0)
        top = self.pos.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, batch_size, self.hidden_size, self.num_layers)
        hidden_state = torch.gather(self.hidden_stack, 0, top).view(batch_size, self.hidden_size, self.num_layers)
        cell_state = torch.gather(self.cell_stack, 0, top).view(batch_size, self.hidden_size, self.num_layers)
        next_hidden, next_cell = self.lstm(input, (hidden_state, cell_state))
        next_hidden_stack = self.hidden_stack.clone()
        next_cell_stack = self.cell_stack.clone()
        next_hidden_stack[self.pos + 1, self.batch_indices, :, :] = next_hidden
        next_cell_stack[self.pos + 1, self.batch_indices, :, :] = next_cell
        self.hidden_stack = next_hidden_stack
        self.cell_stack = next_cell_stack
        self.pos = self.pos + op
        next_stack = None
        return next_stack

    def top(self, stack):
        """
        :type stack: app.models.parallel_rnng.stack_lstm.Stack
        :rtype: torch.Tensor
        """
        batch_size = len(self.pos)
        top = self.pos.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, batch_size, self.hidden_size, self.num_layers)
        output = torch.gather(self.hidden_stack, 0, top).view(batch_size, self.hidden_size, self.num_layers)
        output = output[:, :, self.num_layers - 1]
        return output

    def __str__(self):
        return f'StackLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'

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
