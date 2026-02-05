"""
Module for visializing vectors and vector-spaces
"""
import matplotlib.pyplot as plt  # pylint: disable=import-error
import numpy as np


def plot_vectors_as_wave(vectors):
    """
    Plots one or more vectors/embeddings as line graphs to visualize their fluctuation.

    Handles nested embeddings from feature-extraction pipelines.

    Parameters:
    vectors (list of list or np.ndarray): List of vectors to plot. Each vector can be:
        - 1D (shape: [dim])
        - 2D (shape: [seq_len, dim])
        - 3D (shape: [batch, seq_len, dim])
    """
    processed_vectors = []

    for v in vectors:
        v = np.array(v)

        # If batch dimension exists, take first batch
        if v.ndim == 3:
            v = v[0]

        # If sequence dimension exists, take mean over tokens
        if v.ndim == 2:
            v = v.mean(axis=0)

        # Now v should be 1D
        processed_vectors.append(v)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap('tab10', len(processed_vectors))  # distinct colors

    for idx, vector in enumerate(processed_vectors):
        x = np.arange(len(vector))
        plt.plot(x, vector, label=f"Vector {idx}", color=colors(idx))
        # Optional: label indices (can be commented out for long vectors)
        for i, y in enumerate(vector):
            if len(vector) <= 50:  # only label if vector is not huge
                plt.text(
                    i,
                    y,
                    str(i),
                    fontsize=8,
                    color=colors(idx),
                    alpha=0.7,
                    ha='center',
                    va='bottom'
                )

    plt.title("Vector Fluctuations Visualization")
    plt.xlabel("Dimension Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_vectors_by_pool(original_vectors, pooled_vectors_dict, labels=None):
    """
    Plots original vectors and pooled variants in separate overlapping graphs.

    Args:
        original_vectors (list of np.ndarray): Original embeddings (1D, 2D, or 3D)
        pooled_vectors_dict (dict of list): Keys are pooling names ('mean', 'max', 'min'),
                        values are lists of vectors (same length as original_vectors)
        labels (list of str, optional): Names of vectors (e.g., ['Code A', 'Code B'])
    """
    # Prepare all "categories": original + pooling types
    categories = {'Original': original_vectors}
    categories.update(pooled_vectors_dict)  # Add mean/max/min pools

    #colors = plt.cm.get_cmap('tab10', len(original_vectors))

    for _, (category, vectors) in enumerate(categories.items()):
        plt.figure(figsize=(12, 5))
        for i, v in enumerate(vectors):
            v = np.array(v)
            if v.ndim == 3:
                v = v[0]  # remove batch
            if v.ndim == 2:
                v = v.mean(axis=0)  # reduce sequence if needed

            x = np.arange(len(v))
            plt.plot(
                x,
                v,
                label=(labels[i] if labels else f"Vector {i}") + f" - {category}"
            )

        plt.title(f"{category} Vectors Overlapping")
        plt.xlabel("Dimension Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
