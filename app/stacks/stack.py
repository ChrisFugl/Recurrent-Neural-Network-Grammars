from torch import nn

class Stack(nn.Module):

    def contents(self):
        """
        :rtype: torch.Tensor
        :returns: sequence of all states in stack
        """
        raise NotImplementedError('must be implemented by subclass')

    def empty(self):
        """
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')

    def push(self, item):
        """
        :type item: torch.Tensor
        :rtype: torch.Tensor
        :returns: state of the stack after push
        """
        raise NotImplementedError('must be implemented by subclass')

    def pop(self):
        """
        :rtype: torch.Tensor
        :returns: state of the stack after pop
        """
        raise NotImplementedError('must be implemented by subclass')

    def reset(self):
        """
        Empty the stack.
        """
        raise NotImplementedError('must be implemented by subclass')

    def top(self):
        """
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
