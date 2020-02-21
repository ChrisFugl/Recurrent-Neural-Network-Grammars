from torch import nn

class Composer(nn.Module):

    def forward(self, non_terminal_embedding, popped_stack_items):
        """
        :type non_terminal_embedding: torch.Tensor
        :type popped_stack_items: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
