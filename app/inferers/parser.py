from app.inferers.inferer import Inferer

class ParserInferer(Inferer):

    _names = []

    def __init__(self, device, model, sampler, samples):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type sampler: app.samplers.sampler.Sampler
        :type samples: int
        """
        super().__init__()
        self._device = device
        self._model = model
        self._sampler = sampler
        self._samples = samples

    def names(self):
        """
        :rtype: list of str
        :returns: name of each score returned by inferer
        """
        return list(self._names)

    def infer(self, tokens):
        """
        :type tokens: torch.Tensor
        :rtype: torch.Tensor, dict
        :returns: highest ranking tree and dictionary of scores
        """
        tree = 'foo'
        scores = {}
        return tree, scores

    def __str__(self):
        return f'Parser(samples={self._samples})'
