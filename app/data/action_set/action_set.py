class ActionSet:

    def valid_actions(self, tokens_length, token_counter, last_action, open_non_terminals_count):
        """
        :type tokens_length: int
        :type token_counter: int
        :type last_action: app.actions.action.Action
        :type open_non_terminals_count: int
        :rtype: list of int, dict
        """
        raise NotImplementedError('must be implemented by subclass')
