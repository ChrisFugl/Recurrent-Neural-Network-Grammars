import torch
from torch import nn

class BufferLSTM(nn.Module):

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, dropout=dropout)

    def inference_contents(self, state):
        """
        :type state: app.models.parallel_rnng.buffer_lstm.BufferState
        :rtype: torch.Tensor, torch.Tensor
        """
        max_length = torch.max(state.lengths)
        contents = state.buffer[:max_length]
        return contents, state.lengths

class BufferState:

    def __init__(self, buffer, inputs, lengths):
        """
        :type buffer: torch.Tensor
        :type inputs: torch.Tensor
        :type lengths: torch.Tensor
        """
        self.buffer = buffer
        self.inputs = inputs
        self.lengths = lengths
