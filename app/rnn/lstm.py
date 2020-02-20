from app.rnn.rnn import RNN
import torch
from torch import nn

class LSTM(RNN):

    def __init__(self, device, batch_size, input_size, hidden_size, num_layers, bias, dropout, bidirectional):
        """
        :type device: torch.device
        :type batch_size: int
        :type input_size: int
        :type hidden_size: int
        :type num_layers: int
        :type bias: bool
        :type dropout: float
        :type bidirectional: bool
        """
        super().__init__()
        self._device = device
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._lstm = nn.LSTM(
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
        return self._lstm(input, hidden_state)

    def initial_state(self):
        """
        Get initial hidden state.

        :rtype: torch.Tensor, torch.Tensor
        """
        num_directions = 2 if self._bidirectional else 1
        shape = (self._num_layers * num_directions, self._batch_size, self._hidden_size)
        cell = torch.zeros(shape, device=self._device, requires_grad=True)
        hidden = torch.zeros(shape, device=self._device, requires_grad=True)
        return cell, hidden
