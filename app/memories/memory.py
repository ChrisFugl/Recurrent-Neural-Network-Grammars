from torch import nn

class Memory(nn.Module):

    def add(self, items):
        """
        :type items: torch.Tensor
        :rtype: torch.Tensor
        :returns: last item of memory
        """
        raise NotImplementedError('must be implemented by subclass')

    def count(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def empty(self):
        """
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')

    def get(self, sequence_index, batch_index):
        """
        :type sequence_index: int
        :type batch_index: int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def last(self):
        """
        Get last item of memory.

        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def upto(self, timestep):
        """
        Get memory of every item in the memory until a given timestep.

        :type timestep: int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def reset(self):
        """
        Reset memory state.
        """
        raise NotImplementedError('must be implemented by subclass')
