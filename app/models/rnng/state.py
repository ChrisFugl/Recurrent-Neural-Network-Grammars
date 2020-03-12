class RNNGState:

    def __init__(self, stack_top, action_top, token_top, tokens, tokens_length, open_non_terminals_count, token_counter):
        """
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_top: app.models.rnng.stack.StackNode
        :type token_top: app.models.rnng.stack.StackNode
        :type tokens: torch.Tensor
        :type tokens_length: int
        :type open_non_terminals_count: int
        :type token_counter: int
        """
        self.stack_top = stack_top
        self.action_top = action_top
        self.token_top = token_top
        self.tokens = tokens
        self.tokens_length = tokens_length
        self.open_non_terminals_count = open_non_terminals_count
        self.token_counter = token_counter

    def next(self, stack_top, action_top, token_top, open_non_terminals_count, token_counter):
        """
        :type stack_top: app.models.rnng.stack.StackNode
        :type action_top: app.models.rnng.stack.StackNode
        :type token_top: app.models.rnng.stack.StackNode
        :type open_non_terminals_count: int
        :type token_counter: int
        :rtype: RNNGState
        """
        return RNNGState(
            stack_top,
            action_top,
            token_top,
            self.tokens,
            self.tokens_length,
            open_non_terminals_count,
            token_counter,
        )
