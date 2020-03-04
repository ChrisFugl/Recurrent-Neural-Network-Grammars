from app.inferers.inferer import Inferer

class ImportanceInferer(Inferer):

    def names(self):
        """
        :rtype: list of str
        :returns: name of each score returned by inferer
        """
        raise NotImplementedError('importance sampling has not yet been implemented')

    def infer(self, tokens):
        """
        :type tokens: torch.Tensor
        :rtype: torch.Tensor, dict
        :returns: highest ranking tree and dictionary of scores
        """
        raise NotImplementedError('importance sampling has not yet been implemented')

    def __str__(self):
        return 'Importance'
