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
        # length = contents.size(0)
        # batch_size = contents.size(1)
        # lengths = torch.tensor([length] * batch_size, device=self.device, dtype=torch.long)
        # return contents, lengths

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

    def inference_initialize(self, batch_size):
        """
        :type batch_size: int
        :rtype: app.models.parallel_rnng.history_lstm.HistoryState
        """
        previous = None
        length = 0
        output = None
        state_shape = (self.num_layers, batch_size, self.hidden_size)
        hidden = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        cell = torch.zeros(state_shape, device=self.device, dtype=torch.float)
        hidden_states = (hidden, cell)
        return HistoryState(batch_size, previous, length, output, hidden_states)

    def inference_push(self, state, input):
        """
        :type state: app.models.parallel_rnng.history_lstm.HistoryState
        :type input: torch.Tensor
        :rtype: app.models.parallel_rnng.history_lstm.HistoryState
        """
        next_previous = state
        next_length = state.length + 1
        next_output, next_hidden_states = self.lstm(input, state.hidden_states)
        return HistoryState(state.batch_size, next_previous, next_length, next_output, next_hidden_states)

    def inference_contents(self, state):
        """
        :type state: app.models.parallel_rnng.history_lstm.HistoryState
        :returns: history contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        history = []
        node = state
        while node.output is not None:
            history.append(node.output)
            node = node.previous
        history.reverse()
        contents = torch.cat(history, dim=0)
        lengths = torch.tensor([state.length] * state.batch_size, device=self.device, dtype=torch.long)
        return contents, lengths
        # length = contents.size(0)
        # batch_size = contents.size(1)
        # lengths = torch.tensor([length] * batch_size, device=self.device, dtype=torch.long)
        # return contents, lengths

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
