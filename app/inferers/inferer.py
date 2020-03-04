class Inferer:

    def names(self):
        """
        :rtype: list of str
        :returns: name of each score returned by inferer
        """
        raise NotImplementedError('must be implemented by subclass')

    def infer(self, tokens):
        """
        :type tokens: torch.Tensor
        :rtype: torch.Tensor, dict
        :returns: highest ranking tree and dictionary of scores
        """
        raise NotImplementedError('must be implemented by subclass')
