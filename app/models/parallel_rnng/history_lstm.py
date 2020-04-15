import torch
from torch import nn

class HistoryLSTM(nn.Module):
    """
    Similar to a StackLSTM, but only push is supported
    and it does not allow for hold. Instead it accepts
    a length argument to ensure that positions are not
    pushed beyond their respective lengths.
    """

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

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: history contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        contents = torch.cat(self.history, dim=0)
        return contents, self.lengths

    def initialize(self, batch_size):
        """
        :type batch_size: int
        """
        self.lengths = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        state_shape = (self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(state_shape, device=self.device, requires_grad=True)
        hidden = torch.zeros(state_shape, device=self.device, requires_grad=True)
        self.state = hidden, cell
        self.history = []

    def hold_or_push(self, input, op):
        """
        :type input: torch.Tensor
        :type op: torch.Tensor
        """
        output, next_state = self.lstm(input, self.state)
        self.lengths = self.lengths + op
        self.state = next_state
        self.history.append(output)

    def top(self):
        """
        :rtype: torch.Tensor
        """
        top = []
        for i, length in enumerate(self.lengths):
            output = self.history[length - 1][:, i, :]
            top.append(output)
        return torch.stack(top, dim=1)

    def __str__(self):
        return f'HistoryLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'

class HistoryState:

    def __init__(self, batch_size, previous, length, output, hidden_states):
        """
        :type batch_size: int
        :type previous: app.models.parallel_rnng.history_lstm.HistoryState
        :type length: int
        :type output: torch.Tensor
        :type hidden_states: (torch.Tensor, torch.Tensor)
        """
        self.batch_size = batch_size
        self.previous = previous
        self.length = length
        self.output = output
        self.hidden_states = hidden_states
