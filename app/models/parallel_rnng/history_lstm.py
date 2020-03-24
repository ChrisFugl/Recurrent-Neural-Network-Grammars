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

    def initialize(self, lengths):
        """
        :type lengths: torch.Tensor
        """
        self.lengths = torch.zeros_like(lengths, device=self.device, dtype=torch.long)
        self.max_lengths = lengths
        batch_size = lengths.size(0)
        state_shape = (self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(state_shape, device=self.device, requires_grad=True)
        hidden = torch.zeros(state_shape, device=self.device, requires_grad=True)
        self.state = hidden, cell
        self.history = []

    def push(self, input):
        """
        :type input: torch.Tensor
        """
        output, next_state = self.lstm(input, self.state)
        self.lengths = torch.min(self.lengths + 1, self.max_lengths)
        self.state = next_state
        self.history.append(output)

    def __str__(self):
        return f'HistoryLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
