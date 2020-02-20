from app.stacks.stack import Stack
import torch

class StackRNN(Stack):

    def __init__(self, rnn):
        """
        :type rnn: app.rnn.rnn.RNN
        """
        super().__init__()
        self._rnn = rnn
        self.reset()

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

    def push(self, item):
        """
        :type item: torch.Tensor
        :rtype: torch.Tensor
        :returns: output of the stack after push
        """
        if self.empty():
            previous_state = self._rnn.initial_state()
        else:
            previous_state = self._top.state
        output, next_state = self._rnn(item, previous_state)
        self._top = StackCell(output, next_state, self._top)
        return output

    def pop(self):
        """
        :rtype: torch.Tensor
        :returns: output of the stack after pop
        """
        if self.empty():
            raise Exception('Pop operation is impossible since stack is empty.')
        else:
            output = self._top.output
            self._top = self._top.parent
            return output

    def reset(self):
        """
        Empty the stack.
        """
        self._top = None

    def top(self):
        """
        :rtype: torch.Tensor
        """
        return self._top.output

class StackCell:

    def __init__(self, output, state, parent):
        self.output = output
        self.state = state
        self.parent = parent
