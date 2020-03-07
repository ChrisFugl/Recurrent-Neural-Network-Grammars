class RNNGState:

    def __init__(self, stack, action_history, token_buffer, tokens_embedding, tokens_length, token_index, token_counter, open_non_terminals_count, action_index):
        """
        :type stack: app.stacks.stack.Stack
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type tokens_embedding: torch.Tensor
        :type tokens_length: int
        :type token_index: int
        :type token_counter: int
        :type open_non_terminals_count: int
        :type action_index: int
        """
        self.stack = stack
        self.action_history = action_history
        self.token_buffer = token_buffer
        self.tokens_embedding = tokens_embedding
        self.tokens_length = tokens_length
        self.token_index = token_index
        self.token_counter = token_counter
        self.open_non_terminals_count = open_non_terminals_count
        self.action_index = action_index

    def next(self, token_index, token_counter, open_non_terminals_count):
        """
        :type token_index: int
        :type token_counter: int
        :type open_non_terminals_count: int
        :rtype: RNNGState
        """
        return RNNGState(
            self.stack,
            self.action_history,
            self.token_buffer,
            self.tokens_embedding,
            self.tokens_length,
            token_index,
            token_counter,
            open_non_terminals_count,
            self.action_index + 1
        )
