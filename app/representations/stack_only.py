from app.representations.representation import Representation

class StackOnlyRepresentation(Representation):
    """
    Stack-only representation used by Kuncoro et al. (2018).
    """

    def forward(self, action_history, stack, token_buffer, timestep, batch_index):
        """
        :type action_history: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type token_buffer: app.memories.memory.Memory
        :type timestep: int
        :type batch_index: int
        :rtype: torch.Tensor
        """
        # TODO
        raise NotImplementedError('must be implemented by subclass')
