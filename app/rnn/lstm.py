from app.dropout.variational import variational_dropout_mask
from app.dropout.weight_drop import WeightDrop
from app.rnn.rnn import RNN
import torch
from torch import nn

class LSTM(RNN):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, bidirectional, weight_drop, dropout_type):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type bidirectional: bool
        :type weight_drop: float
        :type dropout_type: str
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = self.hidden_size * self.num_directions
        self.state_dim0_size = self.num_layers * self.num_directions
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.is_variational = dropout_type == 'variational'
        self.weight_drop = weight_drop
        self.use_weight_drop = weight_drop is not None
        if dropout_type == 'normal' and dropout is not None:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional)
        if self.use_weight_drop:
            weights = [f'weight_hh_l{i}' for i in range(num_layers)]
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional)
            self.lstm = WeightDrop(self.lstm, weights, weight_drop)

    def forward(self, input, hidden_state):
        """
        :type input: torch.Tensor
        :type hidden_state: torch.Tensor, torch.Tensor
        :rtype: torch.Tensor, (torch.Tensor, torch.Tensor)
        """
        output, state = self.lstm(input, hidden_state)
        if self.is_variational and self.training:
            length, batch_size, _ = output.shape
            output = output * self.output_mask.expand(length, batch_size, -1)
            if not self.use_weight_drop:
                hidden = state[0] * self.hidden_mask.expand(-1, batch_size, -1)
                cell = state[1] * self.cell_mask.expand(-1, batch_size, -1)
                state = hidden, cell
        return output, state

    def initial_state(self, batch_size):
        """
        Get initial hidden state.

        :type batch_size: int
        :rtype: torch.Tensor, torch.Tensor
        """
        shape = (self.state_dim0_size, batch_size, self.hidden_size)
        cell = torch.zeros(shape, device=self.device, requires_grad=True)
        hidden = torch.zeros(shape, device=self.device, requires_grad=True)
        return cell, hidden

    def reset(self, batch_size):
        if self.use_weight_drop:
            self.lstm.reset(batch_size)
        if self.is_variational and self.training:
            self.output_mask = variational_dropout_mask((1, 1, self.output_size), self.dropout, device=self.device)
            self.hidden_mask = variational_dropout_mask((self.state_dim0_size, 1, self.hidden_size), self.dropout, device=self.device)
            self.cell_mask = variational_dropout_mask((self.state_dim0_size, 1, self.hidden_size), self.dropout, device=self.device)

    def get_output_size(self):
        """
        :rtype: int
        """
        return self.output_size

    def state2output(self, state):
        """
        :type state: torch.Tensor, torch.Tensor
        :rtype: torch.Tensor
        """
        hidden, _ = state # (num_layers * num_directions, batch, hidden_size)
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            forward = hidden[-1, 0]
            backward = hidden[-1, 1]
            output = torch.stack((forward, backward), dim=0)
        else:
            output = hidden[-1]
        return output

    def __str__(self):
        base_args = f'input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}'
        if self.use_weight_drop:
            if self.is_variational and self.dropout is not None:
                return f'LSTM({base_args}, weight_drop={self.weight_drop}, dropout={self.dropout}, dropout_type={self.dropout_type})'
            else:
                return f'LSTM({base_args}, weight_drop={self.weight_drop})'
        elif self.dropout is not None:
            return f'LSTM({base_args}, dropout={self.dropout}, dropout_type={self.dropout_type})'
        else:
            return f'LSTM({base_args})'
