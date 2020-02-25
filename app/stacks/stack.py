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

    def new(self):
        """
        :rtype: app.stacks.stack.Stack
        """
        raise NotImplementedError('must be implemented by subclass')

    def push(self, item, data=None):
        """
        :type item: torch.Tensor
        :type data: object
        :rtype: torch.Tensor
        :returns: state of the stack after push
        """
        raise NotImplementedError('must be implemented by subclass')

    def pop(self):
        """
        :rtype: torch.Tensor, object
        :returns: state of the stack after pop, data
        """
        raise NotImplementedError('must be implemented by subclass')

    def top(self):
        """
        :rtype: torch.Tensor, object
        :returns: state of the top of the stack, data
        """
        raise NotImplementedError('must be implemented by subclass')
