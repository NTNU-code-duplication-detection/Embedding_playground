from datasets.data_generators.sourcecodeplag_dataset_gen import original_non_plagiarized_generator, original_plagiarized_generator
from transformers import pipeline
import numpy as np
from pathlib import Path

def read_file(path: str) -> str:
    """
    Reads the entire content of a file and returns it as a string.

    Args:
        path (str): Path to the file.

    Returns:
        str: Content of the file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    return file_path.read_text(encoding="utf-8")


def get_feature_extraction_pipe(model_name: str) -> pipeline:
    """
    Loads a pipeline from huggingface with emphasis on feature extraction using model_name as model
    """
    pipe = pipeline("feature-extraction", model=model_name)
    return pipe


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

def test_ir_plag():
    """
    Load model and create a simple embedding
    """
    
    model="microsoft/unixcoder-base"
    pipe = get_feature_extraction_pipe(model)

    # Non-plagiarized
    print('---'*20)
    print('Non-plagiarized')
    for original, extern in original_non_plagiarized_generator('datasets/sourcecodeplagiarismdataset/IR-Plag-Dataset'):
        orig    = read_file(original)
        ext     = read_file(extern)
        f1 = mean_pool(pipe(orig))
        f2 = mean_pool(pipe(ext))
        print_cosine_similarity(f1, f2)
    
    # Non-plagiarized
    print('---'*20)
    print('Plagiarized')
    for original, extern in original_plagiarized_generator('datasets/sourcecodeplagiarismdataset/IR-Plag-Dataset'):
        orig    = read_file(original)
        ext     = read_file(extern)
        f1 = mean_pool(pipe(orig))
        f2 = mean_pool(pipe(ext))
        print_cosine_similarity(f1, f2)


def traverse_dataset_lazy():
    import os
    import time

    folder = "datasets/dataset/selected"

    for root, dirs, files in os.walk(folder):
        print(root)  # Print the current folder
        for f in files:
            time.sleep(1)
            print("   ", f)  # Print each file in that folder