"""
Pooling module with all sorts of max, min and mean
"""
import numpy as np

def pool_embeddings(emb):
    """
    Perform mean, max, and min pooling on embeddings from a feature-extraction pipeline.

    Args:
        emb (np.ndarray or list): Shape (1, seq_len, hidden_dim)

    Returns:
        dict: {'mean': ..., 'max': ..., 'min': ...} each of shape (hidden_dim,)
    """
    emb = np.array(emb)  # ensure numpy
    if emb.ndim == 3:
        emb = emb[0]  # remove batch dim → shape (seq_len, hidden_dim)

    mean_pool = emb.mean(axis=0)  # average over tokens
    max_pool = emb.max(axis=0)    # max over tokens
    min_pool = emb.min(axis=0)    # min over tokens

    return {'mean': mean_pool, 'max': max_pool, 'min': min_pool}
