"""
Modified version of https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
"""

def embedding_dropout_mask(weights, dropout):
    """
    :type weights: torch.Tensor
    :type dropout: float
    :rtype: torch.Tensor
    """
    embedding_size = weights.size(0)
    mask = weights.data.new()
    mask = mask.resize_((embedding_size, 1))
    mask = mask.bernoulli_(1 - dropout)
    mask = mask.expand_as(weights) / (1 - dropout)
    return mask
