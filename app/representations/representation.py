from torch import nn

class Representation(nn.Module):
    """
    Forward pass computed the representation of the algorithm's state at the current timestep. This corresponds to u_t in Dyer et al. 2016.
    """

    def forward(self, action_history, action_history_lengths, stack, stack_lengths, token_buffer, token_buffer_lengths):
        """
        :type action_history: torch.Tensor
        :type action_history_lengths: torch.Tensor
        :type stack: torch.Tensor
        :type stack_lengths: torch.Tensor
        :type token_buffer: torch.Tensor
        :type token_buffer_lengths: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def top_only(self):
        """
        :rtype: bool
        """
        raise NotImplementedError('must be implemented by subclass')
