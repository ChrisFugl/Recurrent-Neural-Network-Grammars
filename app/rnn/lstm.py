from app.dropout.weight_drop import WeightDrop
from app.rnn.rnn import RNN
import torch
from torch import nn

class LSTM(RNN):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, bidirectional, weight_drop):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type bidirectional: bool
        :type weight_drop: float
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        self.weight_drop = weight_drop
        self.use_weight_drop = weight_drop is not None
        if self.use_weight_drop:
            weights = [f'weight_hh_l{i}' for i in range(num_layers)]
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional)
            self.lstm = WeightDrop(self.lstm, weights, weight_drop)
        else:
            if dropout is None:
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional)
            else:
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden_state):
        """
        :type input: torch.Tensor
        :type hidden_state: torch.Tensor, torch.Tensor
        :rtype: torch.Tensor, (torch.Tensor, torch.Tensor)
        """
        return self.lstm(input, hidden_state)

    def initial_state(self, batch_size):
        """
        Get initial hidden state.

        :type batch_size: int
        :rtype: torch.Tensor, torch.Tensor
        """
        shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(shape, device=self.device, requires_grad=True)
        hidden = torch.zeros(shape, device=self.device, requires_grad=True)
        return cell, hidden

    def reset(self):
        if self.use_weight_drop:
            self.lstm.reset()

    def get_output_size(self):
        """
        :rtype: int
        """
        return self.hidden_size * self.num_directions

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
        if self.use_weight_drop:
            return f'LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, weight_drop={self.weight_drop})'
        elif self.dropout is not None:
            return f'LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout})'
        else:
            return f'LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
