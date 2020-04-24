from app.constants import PAD_INDEX
from app.dropout.embedding import EmbeddingDropout
from torch import nn

class Embedding(nn.Module):

    def __init__(self, num_embeddings, size, dropout, max_norm, norm_type, scale_grad_by_freq, sparse):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            size,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            padding_idx=PAD_INDEX,
        )
        if dropout is not None:
            self.embedding = EmbeddingDropout(self.embedding, dropout)
            self.use_dropout = True
        else:
            self.use_dropout = False

    def forward(self, indices):
        """
        :type indices: torch.Tensor
        :rtype: torch.Tensor
        """
        return self.embedding(indices)

    def reset(self):
        if self.use_dropout:
            self.embedding.reset()
