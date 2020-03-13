class Sample:

    def __init__(self, gold_actions, gold_tokens, gold_tags, gold_log_prob, predicted_actions, predicted_log_prob, tokens_prob):
        """
        :type gold_actions: list of app.data.actions.action.Action
        :type gold_tokens: list of str
        :type gold_tags: list of str
        :type gold_log_prob: float
        :type predicted_actions: list of app.data.actions.action.Action
        :type predicted_log_prob: float
        :type tokens_prob: float
        """
        self.gold = Gold(gold_actions, gold_tokens, gold_tags, gold_log_prob)
        self.prediction = Prediction(predicted_actions, predicted_log_prob)
        self.tokens_prob = tokens_prob

class Gold:

    def __init__(self, actions, tokens, tags, log_prob):
        """
        :type actions: list of app.data.actions.action.Action
        :type tokens: list of str
        :type tags: list of str
        :type log_prob: float
        """
        self.actions = actions
        self.tokens = tokens
        self.tags = tags
        self.log_prob = log_prob

class Prediction:

    def __init__(self, actions, log_prob):
        """
        :type actions: list of app.data.actions.action.Action
        :type log_prob: float
        """
        self.actions = actions
        self.log_prob = log_prob
