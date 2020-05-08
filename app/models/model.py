from torch import nn

class Model(nn.Module):
    """
    Base model that all models should inherit from.
    """

    def batch_log_likelihood(self, batch):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor, dict
        """
        raise NotImplementedError('must be implemented by subclass')

    def initial_state(self, tokens, tokens_tensor, unknownified_tokens_tensor, singletons_tensor, tags_tensor, lengths):
        """
        Get initial state of model in a parse.

        :type tokens: list of list of str
        :type tokens_tensor: torch.Tensor
        :type unknownified_tokens_tensor: torch.Tensor
        :type singletons_tensor: torch.Tensor
        :type tags_tensor: torch.Tensor
        :type lengths: torch.Tensor
        :returns: initial state
        """
        raise NotImplementedError('must be implemented by subclass')

    def next_state(self, state, actions):
        """
        Advance state of the model to the next state.

        :param state: model specific state
        :type actions: list of app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def next_action_log_probs(self, state, posterior_scaling=1.0, token=None, include_gen=True, include_nt=True):
        """
        Compute log probability of every action given the current state.

        :param state: model specific state
        :type token: str
        :type include_gen: bool
        :type include_nt: bool
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def valid_actions(self, state):
        """
        :param state: model specific state
        :rtype: list of list of int
        """
        raise NotImplementedError('must be implemented by subclass')

    def save(self, path):
        """
        Save model parameters.

        :type path: str
        """
        raise NotImplementedError('must be implemented by subclass')

    def load(self, path):
        """
        Load model parameters from file.

        :type path: str
        """
        raise NotImplementedError('must be implemented by subclass')
