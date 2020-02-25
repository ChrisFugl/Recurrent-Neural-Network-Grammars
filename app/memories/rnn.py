from app.memories.memory import Memory
import torch

class RNNMemory(Memory):

    def __init__(self, rnn):
        """
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self._rnn = rnn
        self._items = []
        self._previous_state = rnn.initial_state()

    def add(self, items):
        """
        :type items: torch.Tensor
        :rtype: torch.Tensor
        :returns: last item of embedding
        """
        state = self._previous_state
        output, self._previous_state = self._rnn(items, state)
        sequence_length, _, _ = output.shape
        for i in range(sequence_length):
            item = output[i]
            self._items.append(item)
        return output[sequence_length - 1]

    def count(self):
        """
        :rtype: int
        """
        return len(self._items)

    def empty(self):
        """
        :rtype: bool
        """
        return len(self._items) == 0

    def get(self, sequence_index, batch_index):
        """
        :type sequence_index: int
        :type batch_index: int
        :rtype: torch.Tensor
        """
        return self._items[sequence_index][batch_index].unsqueeze(dim=0).unsqueeze(dim=0)

    def last(self):
        """
        Get last item of embedding.

        :rtype: torch.Tensor
        """
        items_count = len(self._items)
        return self._items[items_count - 1].unsqueeze(dim=0).unsqueeze(dim=0)

    def new(self):
        """
        :rtype: app.memories.rnn.RNNMemory
        """
        return RNNMemory(self._rnn)

    def upto(self, timestep):
        """
        Get embedding of every item in the embedding until a timestep.

        :type timestep: int
        :rtype: torch.Tensor
        """
        items_sliced = self._items[:timestep]
        items_tensor = torch.stack(items_sliced, dim=0)
        return items_tensor
