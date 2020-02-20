from torch import nn

class Representation(nn.Module):
    """
    Forward pass computed the representation of the algorithm's state at the current timestep. This corresponds to u_t in Dyer et al. 2016.
    """

    def forward(self, action_history, stack, token_buffer, action_timestep, token_timestep, batch_index):
        """
        :type action_history: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type token_buffer: app.memories.memory.Memory
        :type action_timestep: int
        :type token_timestep: int
        :type batch_index: int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
