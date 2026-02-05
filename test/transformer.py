"""
Use a pipeline as a high-level helper
"""
import numpy as np

from transformers import pipeline

def get_feature_extraction_pipe(model_name: str) -> pipeline:
    """
    Loads a pipeline from huggingface with emphasis on feature extraction using model_name as model
    """
    return pipeline("feature-extraction", model=model_name)


def mean_pool(pipeline_output):
    """
    pipeline_output shape: (1, seq_len, hidden_size)
    returns: (hidden_size,)
    """
    return np.mean(pipeline_output[0], axis=0)


def print_cosine_similarity(vec1, vec2):
    """
    Prints cosine simliarity of two vectors
    Vectors must be of same dimensionality
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    cosine_sim = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

    print(f"Cosine similarity: {cosine_sim:.4f}")


if __name__ == '__main__':
    MODEL="microsoft/unixcoder-base"
    pipe_ = get_feature_extraction_pipe(MODEL)

    output_hello_world = mean_pool(pipe_("Hello world"))
    output_bye_world = mean_pool(pipe_("Bye world"))
    print_cosine_similarity(output_hello_world, output_bye_world)
