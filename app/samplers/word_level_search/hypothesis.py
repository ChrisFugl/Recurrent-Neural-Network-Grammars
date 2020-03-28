class Hypothesis:

    def __init__(self, device, state, action, log_prob, parent=None):
        """
        :type device: torch.device
        :param state: state of the model
        :type action: app.data.actions.action.Action
        :type log_prob: float
        :type parent: Hypothesis
        """
        self._device = device
        self._parent = parent
        self.state = state
        self.action = action
        self.log_prob = log_prob

    def actions(self):
        """
        :rtype: list of app.data.actions.action.Action
        """
        actions = []
        node = self
        while node is not None and node.action is not None:
            actions.append(node.action)
            node = node._parent
        actions.reverse()
        return actions

    def successors(self, model, action_converter, token, include_nt):
        """
        Find all successors of the current hypothesis.

        :type model: app.models.model.Model
        :type action_converter: app.data.converters.action.ActionConverter
        :type token: str
        :type include_nt: bool
        :rtype: list of Hypothesis
        """
        successors = []
        log_probs, index2action_index = model.next_action_log_probs(self.state, token=token, include_gen=False, include_nt=include_nt)
        for index, log_prob in enumerate(log_probs):
            action_index = index2action_index[index]
            action = action_converter.integer2action(action_index)
            next_state = model.next_state(self.state, action)
            next_log_prob = self.log_prob + float(log_prob)
            successor = Hypothesis(self._device, next_state, action, next_log_prob, parent=self)
            successors.append(successor)
        return successors
