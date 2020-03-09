from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_REDUCE_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE

class Sampler:

    def evaluate(self):
        """
        :returns: gold trees, predicted trees, log probability of each predicted tree
        :rtype: list of list of app.data.actions.action.Action, list of list of app.data.actions.action.Action, list of float
        """
        raise NotImplementedError('must be implemented by subclass')

    def sample(self, tokens):
        """
        :type tokens: list of str
        :rtype: list of app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def _is_finished_sampling(self, actions, tokens_length):
        """
        :type actions: list of app.data.actions.action.Action
        :rtype: bool
        """
        return (
                len(actions) > 2
            and self._count(actions, ACTION_NON_TERMINAL_TYPE) == self._count(actions, ACTION_REDUCE_TYPE)
            and (
                   self._count(actions, ACTION_SHIFT_TYPE) == tokens_length
                or self._count(actions, ACTION_GENERATE_TYPE) == tokens_length
            )
        )

    def _count(self, actions, type):
        filtered = filter(lambda action: action.type() == type, actions)
        return len(list(filtered))
