import torch
from torch import nn

class BufferLSTM(nn.Module):
    """
    Similar to a StackLSTM, but only push and pop
    does not change the contents. Instead it only moves
    the indices. The GPU should be better able to
    allocate sufficient memory when it does not repeatedly
    have to copy a large buffer.
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
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            dropout=dropout,
        )

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: buffer contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_index = torch.max(self.pos)
        contents = self.buffer[0:max_index + 1, :, :]
        return contents, self.pos + 1

    def initialize(self, inputs, pos):
        """
        :type inputs: torch.Tensor
        :type pos: torch.Tensor
        """
        buffer, _ = self.lstm(inputs)
        self.buffer = buffer
        self.pos = pos

    def forward(self, op):
        """
        :type op: torch.Tensor
        """
        self.pos = self.pos + op

    def hold_or_pop(self, op):
        """
        :type op: torch.Tensor
        :rtype: torch.Tensor
        """
        top = self.top()
        self.pos = self.pos + op
        return top

    def hold_or_push(self, op):
        """
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        self.pos = self.pos + op

    def top(self):
        """
        :rtype: torch.Tensor
        """
        batch_size = self.buffer.size(1)
        top = self.pos.view(1, batch_size, 1).expand(1, batch_size, self.hidden_size)
        output = torch.gather(self.buffer, 0, top).squeeze()
        return output

    def __str__(self):
        return f'BufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'
