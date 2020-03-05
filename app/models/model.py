from torch import nn

class Model(nn.Module):
    """
    Base model that all models should inherit from.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: torch.Tensor
        """
        return self.log_likelihood(batch)

    def log_likelihood(self, batch, posterior_scaling=1.0):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        :type posterior_scaling: float
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def initial_state(self, tokens):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :returns: initial state
        """
        raise NotImplementedError('must be implemented by subclass')

    def next_state(self, previous_state, action):
        """
        Advance state of the model to the next state.

        :param previous_state: model specific previous state
        :type action: app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def next_action_log_probs(self, state, posterior_scaling=1.0):
        """
        Compute log probability of every action given the current state.

        :param state: state of a parse
        :rtype: torch.Tensor
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
