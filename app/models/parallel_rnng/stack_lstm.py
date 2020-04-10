from app.models.parallel_rnng.multi_layer_lstm_cell import MultiLayerLSTMCell
from app.utils import batched_index_select
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

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

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
        """
        self.pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        shape = (stack_size + 1, batch_size, self.hidden_size, self.num_layers)
        self.hidden_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.cell_stack = torch.zeros(shape, device=self.device, dtype=torch.float)
        self.batch_indices = torch.arange(0, batch_size, device=self.device, dtype=torch.long)

    def hold_or_pop(self, op):
        """
        :param op: tensor, (batch size), hold = 0, pop = -1
        :type op: torch.Tensor
        :rtype: torch.Tensor
        """
        output = self.top()
        self.pos = self.pos + op
        return output

    def hold_or_push(self, input, op):
        """
        :param input: tensor, (sequence length, batch size, input size)
        :type input: torch.Tensor
        :param op: tensor, (batch size), push = 1, hold = 0
        :type op: torch.Tensor
        """
        batch_size = input.size(0)
        hidden_state = batched_index_select(self.hidden_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        cell_state = batched_index_select(self.cell_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        next_hidden, next_cell = self.lstm(input, (hidden_state, cell_state))
        # only needs to clone stack state when training, as training requires gradient computations
        if self.training:
            self.hidden_stack = self.hidden_stack.clone()
            self.cell_stack = self.cell_stack.clone()
        self.hidden_stack[self.pos + 1, self.batch_indices, :, :] = next_hidden
        self.cell_stack[self.pos + 1, self.batch_indices, :, :] = next_cell
        self.pos = self.pos + op

    def top(self):
        """
        :rtype: torch.Tensor
        """
        batch_size = len(self.pos)
        output = batched_index_select(self.hidden_stack, self.pos).view(batch_size, self.hidden_size, self.num_layers)
        output = output[:, :, self.num_layers - 1]
        return output

    def __str__(self):
        return f'StackLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
