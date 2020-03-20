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

    def contents(self, buffer):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :type buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :returns: buffer contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        max_index = torch.max(buffer.indices)
        contents = buffer.output[0:max_index + 1, :, :]
        return contents, buffer.indices + 1

    def initialize(self, inputs, indices):
        """
        :type inputs: torch.Tensor
        :type indices: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        buffer, _ = self.lstm(inputs)
        return Buffer(buffer, indices)

    def forward(self, buffer, op):
        """
        :type buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        next_indices = buffer.indices + op
        return Buffer(buffer.output, next_indices)

    def hold_or_pop(self, buffer, op):
        """
        :type buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer, torch.Tensor
        """
        top = self.top(buffer)
        next_indices = buffer.indices + op
        return Buffer(buffer.output, next_indices), top

    def hold_or_push(self, buffer, op):
        """
        :type buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :type op: torch.Tensor
        :rtype: app.models.parallel_rnng.buffer_lstm.Buffer
        """
        next_indices = buffer.indices + op
        return Buffer(buffer.output, next_indices)

    def top(self, buffer):
        """
        :type buffer: app.models.parallel_rnng.buffer_lstm.Buffer
        :rtype: torch.Tensor
        """
        batch_size = buffer.output.size(1)
        top = buffer.indices.view(1, batch_size, 1).expand(1, batch_size, self.hidden_size)
        output = torch.gather(buffer.output, 0, top).squeeze()
        return output

    def __str__(self):
        return f'BufferLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})'

class Buffer:

    def __init__(self, output, indices):
        """
        :type output: torch.Tensor
        :type indices: torch.Tensor
        """
        self.output = output
        self.indices = indices
