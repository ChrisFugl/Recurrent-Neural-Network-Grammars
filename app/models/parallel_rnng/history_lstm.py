from app.utils import batched_index_select
import torch
from torch import nn

class HistoryLSTM(nn.Module):
    """
    Similar to a StackLSTM, but only push is supported
    and it does not allow for hold. Instead it accepts
    a length argument to ensure that positions are not
    pushed beyond their respective lengths.
    """

    def __init__(self, device, rnn):
        """
        :type device: torch.device
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.device = device
        self.rnn = rnn

    def contents(self):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :returns: history contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        if self.prefilled:
            max_length = self.lengths.max()
            contents = self.history[:max_length]
            return contents, self.lengths
        else:
            contents = torch.cat(self.history, dim=0)
            return contents, self.lengths

    def initialize(self, batch_size, inputs=None):
        """
        :type batch_size: int
        :type inputs: torch.Tensor
        """
        if inputs is None:
            self.state = self.rnn.initial_state(batch_size)
            self.history = []
            self.lengths = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
            self.prefilled = False
        else:
            initial_state = self.rnn.initial_state(batch_size)
            self.history, _ = self.rnn(inputs, initial_state)
            # assumes that start embedding is implicitly given through inputs
            self.lengths = torch.ones((batch_size,), device=self.device, dtype=torch.long)
            self.prefilled = True

    def hold_or_push(self, op, input=None):
        """
        :type op: torch.Tensor
        :type input: torch.Tensor
        """
        self.lengths = self.lengths + op
        if not self.prefilled:
            output, next_state = self.rnn(input, self.state)
            self.state = next_state
            self.history.append(output)

    def top(self):
        """
        :rtype: torch.Tensor
        """
        if self.prefilled:
            return batched_index_select(self.history, self.lengths - 1)
        else:
            top = []
            for i, length in enumerate(self.lengths):
                output = self.history[length - 1][:, i, :]
                top.append(output)
            return torch.stack(top, dim=1)

    def __str__(self):
        return f'HistoryLSTM(rnn={self.rnn})'

    def reset(self, batch_size):
        self.rnn.reset(batch_size)
