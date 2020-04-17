"""
Slightly modified from https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
See link for details on this class.
"""
from torch import nn

class WeightDrop(nn.Module):

    def __init__(self, module, weights, dropout):
        """
        :type module: torch.nn.Module
        :type weights: list of str
        :type dropout: float
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.setup()

    def setup(self):
        for name in self.weights:
            name_raw = f'{name}_raw'
            weight = getattr(self.module, name)
            param = nn.Parameter(weight.data)
            del self.module._parameters[name]
            self.module.register_parameter(name_raw, param)

    def set_weights(self):
        for name in self.weights:
            name_raw = f'{name}_raw'
            weight_raw = getattr(self.module, name_raw)
            weight = nn.functional.dropout(weight_raw, p=self.dropout, training=self.training)
            param = nn.Parameter(weight.data)
            setattr(self.module, name, param)

    def forward(self, *args):
        self.set_weights()
        if issubclass(type(self.module), nn.RNNBase):
            self.module.flatten_parameters()
        return self.module.forward(*args)

    def __str__(self):
        return f'WeightDrop(dropout={self.dropout}, module={self.module})'
