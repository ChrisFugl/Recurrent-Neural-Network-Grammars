from app.models.parallel_rnng.multi_layer_lstm_cell import MultiLayerLSTMCell
import torch
from torch import nn

class StackLSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, weight_drop, dropout_type, layer_norm):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type weight_drop: float
        :type dropout_type: str
        :type layer_norm: bool
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.weight_drop = weight_drop
        self.layer_norm = layer_norm
        self.lstm = MultiLayerLSTMCell(device, input_size, hidden_size, num_layers, bias, dropout, weight_drop, dropout_type, layer_norm)

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
        hidden_state = self.hidden_stack[self.pos, self.batch_indices]
        cell_state = self.cell_stack[self.pos, self.batch_indices]
        next_hidden, next_cell = self.lstm(input, (hidden_state, cell_state))
        self.hidden_stack[self.pos + 1, self.batch_indices] = next_hidden
        self.cell_stack[self.pos + 1, self.batch_indices] = next_cell
        self.pos = self.pos + op

    def top(self):
        """
        :rtype: torch.Tensor
        """
        output = self.hidden_stack[self.pos, self.batch_indices]
        output = output[:, :, self.num_layers - 1]
        return output

    def reset(self, batch_size):
        self.lstm.reset(batch_size)

    def __str__(self):
        base_args = f'input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, layer_norm={self.layer_norm}'
        if self.weight_drop is not None:
            if self.dropout is not None and self.dropout_type == 'variational':
                return f'StackLSTM({base_args}, weight_drop={self.weight_drop}, dropout={self.dropout}, dropout_type={self.dropout_type})'
            else:
                return f'StackLSTM({base_args}, weight_drop={self.weight_drop})'
        elif self.dropout is not None:
            return f'StackLSTM({base_args}, dropout={self.dropout}, dropout_type={self.dropout_type})'
        else:
            return f'StackLSTM({base_args})'
