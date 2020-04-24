"""
Modified version of https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
"""
from app.constants import PAD_INDEX
from torch import nn

class EmbeddingDropout(nn.Module):

    def __init__(self, embedding, dropout):
        """
        :type embedding: torch.nn.Embedding
        :type dropout: float
        """
        super().__init__()
        self.embedding = embedding
        self.embedding_size = embedding.weight.size(0)
        self.dropout = dropout
        self.reset()

    def forward(self, indices):
        """
        :type indices: torch.Tensor
        """
        if self.training:
            embedding = self.masked_embedding
        else:
            embedding = self.embedding.weight
        return nn.functional.embedding(
            indices, embedding, PAD_INDEX,
            self.embedding.max_norm, self.embedding.norm_type,
            self.embedding.scale_grad_by_freq, self.embedding.sparse
        )

    def reset(self):
        mask = self.embedding.weight.data.new()
        mask = mask.resize_((self.embedding_size, 1))
        mask = mask.bernoulli_(1 - self.dropout)
        mask = mask.expand_as(self.embedding.weight) / (1 - self.dropout)
        self.masked_embedding = mask * self.embedding.weight

    def __str__(self):
        return f'EmbeddingDropout(dropout={self.dropout})'
