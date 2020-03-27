class State:

    def __init__(self, history, stack, buffer, tokens_length, token_counter, last_action, open_nt_count):
        """
        :type history: app.models.parallel_rnng.history_lstm.HistoryState
        :type stack: app.models.parallel_rnng.stack_lstm.StackState
        :type buffer: app.models.parallel_rnng.buffer_lstm.BufferState
        :type tokens_length: int
        :type token_counter: int
        :type last_action: app.data.actions.action.Action
        :type open_nt_count: int
        """
        self.history = history
        self.stack = stack
        self.buffer = buffer
        self.tokens_length = tokens_length
        self.token_counter = token_counter
        self.last_action = last_action
        self.open_nt_count = open_nt_count
