from app.stacks.stack import Stack
import torch

class StackRNN(Stack):

    def __init__(self, rnn):
        """
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self._rnn = rnn
        self._top = None

    def contents(self):
        """
        :rtype: torch.Tensor
        :returns: sequence of all states in stack
        """
        contents = []
        top = self._top
        while top is not None:
            contents.append(top.output)
            top = top.parent
        contents.reverse()
        return torch.cat(contents, dim=0)

    def empty(self):
        """
        :rtype: bool
        """
        return self._top is None

    def new(self):
        """
        :rtype: app.stacks.rnn.StackRNN
        """
        return StackRNN(self._rnn)

    def push(self, item, data=None):
        """
        :type item: torch.Tensor
        :type data: object
        :rtype: torch.Tensor
        :returns: output of the stack after push
        """
        if self.empty():
            previous_state = self._rnn.initial_state()
        else:
            previous_state = self._top.state
        output, next_state = self._rnn(item, previous_state)
        self._top = StackCell(output, next_state, self._top, data)
        return output

    def pop(self):
        """
        :rtype: torch.Tensor, object
        :returns: state of the stack after pop, data
        """
        if self.empty():
            raise Exception('Pop operation is impossible on an empty stack.')
        else:
            output = self._top.output
            data = self._top.data
            self._top = self._top.parent
            return output, data

    def top(self):
        """
        :rtype: torch.Tensor, object
        :returns: state of the top of the stack, data
        """
        return self._top.output, self._top.data

    def __str__(self):
        return f'StackRNN(rnn={self._rnn})'

class StackCell:

    def __init__(self, output, state, parent, data):
        self.output = output
        self.state = state
        self.parent = parent
        self.data = data
