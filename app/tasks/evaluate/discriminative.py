from app.tasks.evaluate.evaluator import Evaluator

class DiscriminativeEvaluator(Evaluator):
    """
    Select greatest log likelihood without regard to the loaded model.
    """

    def evaluate_predictions(self, tokens, tags, predictions):
        """
        :type tokens: list of str
        :type tags: list of str
        :type predictions: list of object
        :rtype: list of app.data.actions.action.Action, float
        """
        best_prediction = None
        best_log_likelihood = None
        for prediction in predictions:
            log_likelihood = prediction['log_likelihood']
            if best_log_likelihood is None or best_log_likelihood < log_likelihood:
                best_prediction = prediction['actions']
                best_log_likelihood = log_likelihood
        best_prediction = list(map(self.action_converter.string2action, best_prediction))
        return best_prediction, best_log_likelihood

    def get_predicted_log_likelihoods(self, evaluations):
        return evaluations

    def get_extra_evaluation_stats(self, evaluations):
        """
        :rtype: list of (str, str)
        """
        return []
