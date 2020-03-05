class RNNGState:

    def __init__(self, stack, action_history, token_buffer, token_embeddings, tokens_length):
        """
        :type stack: app.stacks.stack.Stack
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type token_embeddings: torch.Tensor
        :type tokens_length: int
        """
        self.stack = stack
        self.action_history = action_history
        self.token_buffer = token_buffer
        self.token_embeddings = token_embeddings
        self.tokens_length = tokens_length
