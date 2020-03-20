from app.rnn.rnn import RNN
import torch
from torch import nn

class LSTM(RNN):

    def __init__(self, device, input_size, hidden_size, num_layers, bias, dropout, bidirectional):
        """
        :type device: torch.device
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type bidirectional: bool
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional
        )

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
        num_directions = 2 if self.bidirectional else 1
        shape = (self.num_layers * num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(shape, device=self.device, requires_grad=True)
        hidden = torch.zeros(shape, device=self.device, requires_grad=True)
        return cell, hidden

    def __str__(self):
        return f'LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
