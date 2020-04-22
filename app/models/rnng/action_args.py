class ActionLogProbs:

    def __init__(self, representation, log_prob_base, action2index):
        """
        :type representation: torch.Tensor
        :type log_prob_base: torch.Tensor
        :type action2index: dict
        """
        self.representation = representation
        self.log_prob_base = log_prob_base
        self.action2index = action2index

class ActionOutputs:

    def __init__(self, stack_top, buffer_state, open_non_terminals_count, token_counter):
        """
        :type stack_top: app.models.rnng.stack.StackNode
        :type buffer_state: app.models.rnng.buffer.BufferState
        :type open_non_terminals_count: int
        :type token_counter: int
        """
        self.stack_top = stack_top
        self.buffer_state = buffer_state
        self.open_non_terminals_count = open_non_terminals_count
        self.token_counter = token_counter

    def update(self, **kwargs):
        action_log_prob = kwargs.get('action_log_prob', None)
        stack_top = kwargs.get('stack_top', self.stack_top)
        buffer_state = kwargs.get('buffer_state', self.buffer_state)
        open_non_terminals_count = kwargs.get('open_non_terminals_count', self.open_non_terminals_count)
        token_counter = kwargs.get('token_counter', self.token_counter)
        return action_log_prob, stack_top, buffer_state, open_non_terminals_count, token_counter
