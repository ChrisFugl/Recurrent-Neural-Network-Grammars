from app.embeddings.embedding import Embedding

def get_embedding(num_embeddings, size, dropout, config):
    """
    :type num_embeddings: int
    :type size: int
    :type config: object
    :rtype: app.embeddings.embedding.Embedding
    """
    return Embedding(num_embeddings, size, dropout, config.max_norm, config.norm_type, config.scale_grad_by_freq, config.sparse)
