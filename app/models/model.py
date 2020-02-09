import torch

class Model(nn.Module):
    """
    Base model that all models should inherit from.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def likelihood(self, data):
        """
        Compute likelihood of each sentence/tree in a batch.

        :param data: batch of words
        :type data: list of sentence/tree
        :rtype: list of float
        """
        raise NotImplementedError('method must be implemented by a subclass')

    def next_state(self, previous_state):
        """
        Compute next state and action probabilities given the previous state.

        :returns: next state, next_actions
        """
        raise NotImplementedError('method must be implemented by a subclass')

    def parse(self, sentence):
        """
        Generate a parse of a sentence.
        """
        raise NotImplementedError('method must be implemented by a subclass')
