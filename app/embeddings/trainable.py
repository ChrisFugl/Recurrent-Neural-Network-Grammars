from app.constants import PAD_INDEX
from app.dropout.embedding import embedding_dropout_mask
from app.embeddings.embedding import Embedding
from torch import nn

class TrainableEmbedding(Embedding):

    def __init__(self, num_embeddings, size, dropout, max_norm, norm_type, scale_grad_by_freq, sparse):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, size,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            padding_idx=PAD_INDEX,
        )
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.dropout = dropout
        self.use_dropout = dropout is not None
        self.reset()

    def forward(self, indices):
        """
        :type indices: torch.Tensor
        :rtype: torch.Tensor
        """
        if self.use_dropout and self.training:
            weights = self.mask * self.embedding.weight
        else:
            weights = self.embedding.weight
        return nn.functional.embedding(
            indices,
            weights,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            padding_idx=PAD_INDEX,
        )

    def reset(self):
        if self.use_dropout and self.training:
            self.mask = embedding_dropout_mask(self.embedding.weight, self.dropout)

    def __str__(self):
        if self.use_dropout:
            return f'Trainable(dropout={self.dropout})'
        else:
            return f'Trainable()'
