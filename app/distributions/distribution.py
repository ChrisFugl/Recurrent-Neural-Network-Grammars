from torch import nn

class Distribution(nn.Module):

    def forward(self, value):
        return self.log_prob(value)

    def log_prob(self, representation, value):
        """
        Compute log probability of given value.

        :type representation: torch.Tensor
        :type value: object
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def sample(self, representation, n):
        """
        Sample n values from distribution.

        :type representation: torch.Tensor
        :type n: int
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
