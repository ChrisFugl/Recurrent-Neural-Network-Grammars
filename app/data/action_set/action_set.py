class ActionSet:

    def valid_actions(self, token_buffer, token_index, stack, open_non_terminals_count):
        """
        :type token_buffer: app.memories.memory.Memory
        :type token_index: int
        :type stack: app.stacks.stack.Stack
        :type open_non_terminals_count: int
        :rtype: list of int, dict
        """
        raise NotImplementedError('must be implemented by subclass')
