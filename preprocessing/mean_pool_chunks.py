"""
module for mean pooling embedding outputs
"""
import numpy as np

def mean_pool_chunks(embedded_chunks):
    """
    Meanpool embeddings from chunks
    """
    flat_embs = []
    for emb in embedded_chunks:
        arr = np.array(emb).squeeze()  # shape: (num_tokens, embedding_dim)
        if arr.ndim == 1:  # single token case
            arr = arr[np.newaxis, :]
        avg_emb = arr.mean(axis=0)  # mean over tokens
        flat_embs.append(avg_emb)
    return np.array(flat_embs)
