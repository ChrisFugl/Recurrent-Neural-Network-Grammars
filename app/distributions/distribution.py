from torch import nn

class Distribution(nn.Module):

    def forward(self, value):
        return self.log_prob(value)

    def log_prob(self, representation, value, posterior_scaling=1.0):
        """
        Compute log probability of given value.

        :type representation: torch.Tensor
        :type posterior_scaling: float
        :type value: object
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def log_probs(self, representation, posterior_scaling=1.0):
        """
        Log probabilities of all elements in distribution.

        :type representation: torch.Tensor
        :type posterior_scaling: float
        :rtype: torch.Tensor, list of int
        :returns: log probabilities, log probability index to action index
        """
        raise NotImplementedError('must be implemented by subclass')
