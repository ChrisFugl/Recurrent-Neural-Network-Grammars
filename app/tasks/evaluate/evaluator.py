class Evaluator:

    def __init__(self, model, action_converter, token_converter, tag_converter, non_terminal_converter):
        """
        :type model: app.models.model.Model
        :type action_converter: app.data.converters.action.ActionConverter
        :type token_converter: app.data.converters.token.TokenConverter
        :type tag_converter: app.data.converters.tag.TagConverter
        :type non_terminal_converter: app.data.converters.non_terminal.NonTerminalConverter
        """
        self.model = model
        self.action_converter = action_converter
        self.token_converter = token_converter
        self.tag_converter = tag_converter
        self.non_terminal_converter = non_terminal_converter

    def evaluate_predictions(self, tokens, tags, predictions):
        raise NotImplementedError('must be implemented by subclass')

    def get_predicted_log_likelihoods(self, evaluations):
        raise NotImplementedError('must be implemented by subclass')

    def get_extra_evaluation_stats(self, evaluations):
        """
        :rtype: list of (str, str)
        """
        raise NotImplementedError('must be implemented by subclass')
