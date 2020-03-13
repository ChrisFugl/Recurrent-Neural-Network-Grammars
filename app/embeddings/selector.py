from app.constants import PAD_INDEX
from torch.nn import Embedding

def get_embedding(num_embeddings, size, config):
    """
    :type num_embeddings: int
    :type size: int
    :type config: object
    :rtype: torch.nn.Embedding
    """
    return Embedding(
        num_embeddings,
        size,
        max_norm=config.max_norm,
        norm_type=config.norm_type,
        scale_grad_by_freq=config.scale_grad_by_freq,
        sparse=config.sparse,
        padding_idx=PAD_INDEX,
    )
