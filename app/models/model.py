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
        raise NotImplementedError('method must be implemented by a subclass')

    def next_state(self, previous_state):
        """
        Compute next state and action probabilities given the previous state.

        :returns: next state, next_actions
        """
        raise NotImplementedError('method must be implemented by a subclass')

    def parse(self, tokens):
        """
        Generate a parse of a sentence.

        :type tokens: torch.Tensor
        """
        raise NotImplementedError('method must be implemented by a subclass')

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
