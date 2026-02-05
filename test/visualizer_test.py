"""
Test module for visualization
"""
from transformers import pipeline

from visualizer.visualize_vectorspace import (
    #plot_vectors_as_wave,
    plot_vectors_by_pool
)
from data.dataset_factory import get_dataset_generator
from src.pooling import pool_embeddings


def test_visualizer():
    """
    Small script for visualization test
    """
    model="microsoft/unixcoder-base"
    pipe = pipeline("feature-extraction", model=model)

    for code_sample in get_dataset_generator(
        dataset_name='codexglue',
        mode='pairs',
        **{'split': 'train'}
        ):
        code_a = code_sample.code_a
        code_b = code_sample.code_b
        emb_a = pipe(code_a)
        emb_b = pipe(code_b)
        #plot_vectors_as_wave([emb_a, emb_b])

        pooled_a = pool_embeddings(emb_a)
        pooled_b = pool_embeddings(emb_b)

        print("CODE A")
        print(code_a)
        print("CODE B")
        print(code_b)

        pooled_dict = {
            'mean': [pooled_a['mean'], pooled_b['mean']],
            'max': [pooled_a['max'], pooled_b['max']],
            'min': [pooled_a['min'], pooled_b['min']]
        }

        plot_vectors_by_pool([emb_a, emb_b], pooled_dict, labels=['Code A', 'Code B'])
