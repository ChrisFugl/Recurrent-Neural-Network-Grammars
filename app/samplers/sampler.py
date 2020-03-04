class Sampler:

    def sample(self, tokens):
        """
        :param tokens: tokens length x 1 x hidden size
        :type tokens: torch.Tensor
        :returns: actions length x 1 x 1
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')
