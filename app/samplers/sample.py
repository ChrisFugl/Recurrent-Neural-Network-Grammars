class Sample:

    def __init__(self, gold, predictions):
        """
        :type gold: (list of app.data.actions.action.Action, list of str, list of str, list of str, list of float, float)
        :type predictions: list of (list of app.data.actions.action.Action, list of float, float)
        """
        self.gold = Gold(*gold)
        self.predictions = [Prediction(*prediction) for prediction in predictions]

class Gold:

    def __init__(self, actions, tokens, unknownified_tokens, tags, log_probs, log_likelihood):
        """
        :type actions: list of app.data.actions.action.Action
        :type tokens: list of str
        :type unknownified_tokens: list of str
        :type tags: list of str
        :type log_probs: list of float
        :type log_likelihood: float
        """
        self.actions = actions
        self.tokens = tokens
        self.unknownified_tokens = unknownified_tokens
        self.tags = tags
        self.log_probs = log_probs
        self.log_likelihood = log_likelihood

class Prediction:

    def __init__(self, actions, log_probs, log_likelihood):
        """
        :type actions: list of app.data.actions.action.Action
        :type log_probs: list of float
        :type log_likelihood: float
        """
        self.actions = actions
        self.log_probs = log_probs
        self.log_likelihood = log_likelihood
