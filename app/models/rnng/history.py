import torch
from torch import nn

class History(nn.Module):

    def __init__(self, device, rnn):
        """
        :type device: torch.device
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self.device = device
        self.rnn = rnn

    def contents(self, state):
        """
        :rtype: torch.Tensor, torch.Tensor
        """
        if state.prefilled:
            length = state.sequence_index + 1
            length_tensor = torch.tensor([length], device=self.device, dtype=torch.long)
            output = state.history[:length]
            return output, length_tensor
        else:
            contents = []
            node = state
            length = 0
            while node.output is not None:
                contents.append(node.output)
                node = node.previous
                length += 1
            contents = torch.cat(contents, dim=0)
            length_tensor = torch.tensor([length], device=self.device, dtype=torch.long)
            return contents, length_tensor

    def initialize(self, batch_size, inputs=None):
        """
        :type batch_size: int
        :type inputs: (torch.Tensor, torch.Tensor)
        :rtype: list of object
        """
        states = []
        if inputs is None:
            initial_rnn_state = self.rnn.initial_state(1)
            previous = None
            output = None
            for batch_index in range(batch_size):
                state = DynamicHistoryState(previous, output, initial_rnn_state)
                states.append(state)
        else:
            initial_rnn_state = self.rnn.initial_state(batch_size)
            tensors, lengths = inputs
            batched_history, _ = self.rnn(tensors, initial_rnn_state)
            hidden_size = batched_history.size(2)
            sequence_index = 0
            for batch_index in range(batch_size):
                length = lengths[batch_index]
                history = batched_history[:length, batch_index]
                history = history.view(length, 1, hidden_size)
                state = PrefilledHistoryState(history, sequence_index)
                states.append(state)
        return states

    def push(self, state, input=None):
        """
        :type input: torch.Tensor
        """
        if state.prefilled:
            next_state = PrefilledHistoryState(state.history, state.sequence_index + 1)
        else:
            next_output, next_rnn_state = self.rnn(input, state.rnn_state)
            next_state = DynamicHistoryState(state, next_output, next_rnn_state)
        return next_state

    def top(self, state):
        if state.prefilled:
            hidden_size = state.history.size(2)
            output = state.history[state.sequence_index]
            output = output.view(1, 1, hidden_size)
            return output
        else:
            return state.output

    def reset(self, batch_size):
        self.rnn.reset(batch_size)

    def __str__(self):
        return f'History(rnn={self.rnn})'

class PrefilledHistoryState:

    prefilled = True

    def __init__(self, history, sequence_index):
        """
        :type history: torch.Tensor
        :type sequence_index: int
        """
        self.history = history
        self.sequence_index = sequence_index

class DynamicHistoryState:

    prefilled = False

    def __init__(self, previous, output, rnn_state):
        """
        :type previous: app.models.rnng.history.DynamicHistoryState
        :type rnn_state: torch.Tensor
        :type output: torch.Tensor
        """
        self.previuos = previous
        self.rnn_state = rnn_state
        self.output = output
