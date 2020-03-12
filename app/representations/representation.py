from torch import nn

class Representation(nn.Module):
    """
    Forward pass computed the representation of the algorithm's state at the current timestep. This corresponds to u_t in Dyer et al. 2016.
    """

    def forward(self, action_history, stack, token_buffer):
        """
        :type action_history: torch.Tensor
        :type stack: torch.Tensor
        :type token_buffer: torch.Tensor
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
