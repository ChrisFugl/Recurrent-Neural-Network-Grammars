from app.models.parallel_rnng.memory_lstm import MemoryLSTM
import torch

class HistoryLSTM(MemoryLSTM):
    """
    Similar to a StackLSTM, but only push is allowed.
    This is done for performance reasons, since a
    push-only stack does not have to clone the entire
    stack contents every time an update is made.
    """

    def contents(self, history):
        """
        Retrieve content for all batches. Each batch will include states up to the largest index.

        :type history: app.models.parallel_rnng.history_lstm.History
        :returns: history contents and batch lengths
        :rtype: torch.Tensor, torch.Tensor
        """
        batch_size = history.hidden_state.size(1)
        index = 0
        node = history
        contents = []
        # do not include initial state
        while node.previous is not None:
            output = node.hidden_state[self._num_layers - 1]
            contents.append(output)
            index += 1
            node = node.previous
        contents.reverse()
        contents = torch.stack(contents, dim=0)
        indices = torch.tensor([index] * batch_size, device=self._device, dtype=torch.long)
        return contents, indices

    def initialize(self, batch_size):
        """
        :type batch_size: int
        :rtype: app.models.parallel_rnng.history_lstm.History
        """
        shape = (self._num_layers, batch_size, self._hidden_size)
        hidden_state = torch.zeros(shape, device=self._device, dtype=torch.float)
        cell_state = torch.zeros(shape, device=self._device, dtype=torch.float)
        return History(hidden_state, cell_state, None)

    def push(self, history, input):
        """
        :type history: app.models.parallel_rnng.history_lstm.History
        :param input: tensor, (sequence length, batch size, input size)
        :type input: torch.Tensor
        :rtype: app.models.parallel_rnng.history_lstm.History
        """
        top_state = (history.hidden_state, history.cell_state)
        _, (next_hidden_state, next_cell_state) = self._lstm(input, top_state)
        return History(next_hidden_state, next_cell_state, history)

    def top(self, history):
        """
        :type history: app.models.parallel_rnng.history_lstm.History
        :rtype: torch.Tensor
        """
        batch_size = history.hidden_state.size(2)
        last_layer_state = history.hidden_state[:, self._num_layers - 1, :, :]
        top = history.indices.view(1, batch_size, 1).expand(1, batch_size, self._hidden_size)
        output = torch.gather(last_layer_state, 0, top).squeeze()
        return output

    def __str__(self):
        return f'HistoryLSTM(input_size={self._input_size}, hidden_size={self._hidden_size}, num_layers={self._num_layers})'

class History:

    def __init__(self, hidden_state, cell_state, previous):
        """
        :type hidden_state: torch.Tensor
        :type cell_state: torch.Tensor
        :type previous: app.models.parallel_rnng.history_lstm.History
        """
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.previous = previous
