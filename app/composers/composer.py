from torch import nn

class Composer(nn.Module):

    def forward(self, non_terminal_embedding, popped_stack_items, lengths):
        """
        :type non_terminal_embedding: torch.Tensor
        :type popped_stack_items: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def reset(self):
        """
        Resets internal state.
        """
        raise NotImplementedError('must be implemented by subclass')
