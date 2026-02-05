"""
Module for code plagiarism detection using learned embeddings.

This module provides functionality to detect code plagiarism by comparing
code embeddings generated using transformers models.
"""

# pylint: disable=import-error

from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
class CodeEmbeddingPipeline:
    """
    Pipeline for creating embeddings from code functions and detecting plagiarism.
    """

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        """
        Initialize the pipeline with a specific model.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.pipe = pipeline("feature-extraction", model=model_name)

    def read_function(self, path: str) -> str:
        """
        Read a function from a file.

        Args:
            path: Path to the code file

        Returns:
            str: Content of the file
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        return file_path.read_text(encoding="utf-8")

    def create_embedding(self, code: str) -> np.ndarray:
        """
        Create embedding from code using mean pooling.

        Args:
            code: Source code as string

        Returns:
            np.ndarray: Mean-pooled embedding vector
        """
        # Get embeddings from pipeline (shape: 1, seq_len, hidden_size)
        output = self.pipe(code)
        # Mean pool over sequence length (shape: hidden_size,)
        embedding = np.mean(output[0], axis=0)
        return embedding

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            float: Cosine similarity score [0, 1]
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        cosine_sim = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        return float(cosine_sim)

    def function_to_embedding(self, function_path: str) -> np.ndarray:
        """
        Complete pipeline: Function file -> Embedding.

        @param function_path: Path to function file

        @return: np.ndarray: Embedding vector
        """
        code = self.read_function(function_path)
        embedding = self.create_embedding(code)
        return embedding

    def compare_functions(self, func1_path: str, func2_path: str) -> Tuple[float, bool]:
        """
        Compare two functions for plagiarism detection.

        Args:
            func1_path: Path to first function
            func2_path: Path to second function

        Returns:
            Tuple[float, bool]: (similarity_score, is_plagiarized)
        """
        emb1 = self.function_to_embedding(func1_path)
        emb2 = self.function_to_embedding(func2_path)

        similarity = self.compute_cosine_similarity(emb1, emb2)


        threshold = 0.90 # Treshold parameter. Definding what is plagiat and what is not
        is_plagiarized = similarity >= threshold

        return similarity, is_plagiarized


def original_non_plagiarized_generator(dataset_path: str):
    """
    Generator for non-plagiarized pairs - compares original files from different cases.
    """
    dataset_root = Path(dataset_path)
    original_files = []

    for case_folder in sorted(dataset_root.iterdir()):
        if not case_folder.is_dir() or not case_folder.name.startswith('case-'):
            continue

        original_folder = case_folder / 'original'
        if original_folder.exists():
            java_files = list(original_folder.glob('*.java'))
            if java_files:
                original_files.append(java_files[0])

    # Compare different original files (non-plagiarized pairs)
    for i, file1 in enumerate(original_files):
        for file2 in original_files[i + 1:min(i + 3, len(original_files))]:
            yield (str(file1), str(file2))


def original_plagiarized_generator(dataset_path: str):
    """
    Generator for plagiarized pairs - compares original with plagiarized versions.
    """
    dataset_root = Path(dataset_path)

    for case_folder in sorted(dataset_root.iterdir()):
        if not case_folder.is_dir() or not case_folder.name.startswith('case-'):
            continue

        # Find original file
        original_folder = case_folder / 'original'
        if not original_folder.exists():
            continue

        original_files = list(original_folder.glob('*.java'))
        if not original_files:
            continue

        original_file = original_files[0]

        # Find plagiarized files (recursively)
        plagiarized_folder = case_folder / 'plagiarized'
        if not plagiarized_folder.exists():
            continue

        plag_files = list(plagiarized_folder.rglob('*.java'))
        for plag_file in plag_files:
            yield (str(original_file), str(plag_file))


class PlagiarismDetectionAnalyzer:
    """
    Analyzer for evaluating plagiarism detection using histogram analysis.
    """

    def __init__(self, embedding_pipeline: CodeEmbeddingPipeline):
        self.pipeline = embedding_pipeline
        self.plagiarized_scores: List[float] = []
        self.non_plagiarized_scores: List[float] = []

    def analyze_dataset(self, dataset_path: str):
        """
        Analyze entire dataset and collect similarity scores.

        Args:
            dataset_path: Path to IR-Plag-Dataset
        """
        print('=' * 60)
        print('Analyzing Non-Plagiarized Pairs')
        print('=' * 60)

        for original, extern in original_non_plagiarized_generator(dataset_path):
            similarity, is_plag = self.pipeline.compare_functions(original, extern)
            self.non_plagiarized_scores.append(similarity)
            print(f"Similarity: {similarity:.4f} | Plagiarized: {is_plag}")

        print('\n' + '=' * 60)
        print('Analyzing Plagiarized Pairs')
        print('=' * 60)

        for original, extern in original_plagiarized_generator(dataset_path):
            similarity, is_plag = self.pipeline.compare_functions(original, extern)
            self.plagiarized_scores.append(similarity)
            print(f"Similarity: {similarity:.4f} | Plagiarized: {is_plag}")

    def plot_histogram(self, save_path: str = "similarity_histogram.png"):
        """
        Create histogram visualization of similarity scores.

        Args:
            save_path: Path to save the histogram plot
        """
        plt.figure(figsize=(12, 6))

        # Plot histograms
        plt.hist(self.non_plagiarized_scores, bins=20, alpha=0.5,
                 label='Non-Plagiarized', color='blue', density=True)
        plt.hist(self.plagiarized_scores, bins=20, alpha=0.5,
                 label='Plagiarized (Type-1)', color='red', density=True)

        # Add vertical line for threshold
        plt.axvline(x=0.95, color='green', linestyle='--',
                    linewidth=2, label='Threshold (0.95)')

        plt.xlabel('Cosine Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Similarity Scores: Plagiarized vs Non-Plagiarized',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\nHistogram saved to: {save_path}")
        plt.show()

    def compute_statistics(self):
        """
        Compute and print statistics for both groups.
        """
        print('\n' + '=' * 60)
        print('STATISTICS')
        print('=' * 60)

        print('\nNon-Plagiarized Pairs:')
        print(f"  Count: {len(self.non_plagiarized_scores)}")
        print(f"  Mean: {np.mean(self.non_plagiarized_scores):.4f}")
        print(f"  Std: {np.std(self.non_plagiarized_scores):.4f}")
        print(f"  Min: {np.min(self.non_plagiarized_scores):.4f}")
        print(f"  Max: {np.max(self.non_plagiarized_scores):.4f}")

        print('\nPlagiarized Pairs (Type-1):')
        print(f"  Count: {len(self.plagiarized_scores)}")
        print(f"  Mean: {np.mean(self.plagiarized_scores):.4f}")
        print(f"  Std: {np.std(self.plagiarized_scores):.4f}")
        print(f"  Min: {np.min(self.plagiarized_scores):.4f}")
        print(f"  Max: {np.max(self.plagiarized_scores):.4f}")

        # Calculate separation
        separation = np.mean(self.plagiarized_scores) - np.mean(self.non_plagiarized_scores)
        print(f"\nSeparation between groups: {separation:.4f}")


def main():
    """
    Main function to run the plagiarism detection pipeline.
    """
    # Initialize pipeline
    print("Initializing Code Embedding Pipeline...")
    embedding_pipeline = CodeEmbeddingPipeline(model_name="microsoft/unixcoder-base")

    # Initialize analyzer
    analyzer = PlagiarismDetectionAnalyzer(embedding_pipeline)

    # Analyze dataset
    dataset_path = (Path(__file__).parent.parent /'datasets'/
                    'sourcecodeplagiarismdataset'/'IR-Plag-Dataset')
    analyzer.analyze_dataset(str(dataset_path))

    # Compute statistics
    analyzer.compute_statistics()

    # Plot histogram
    analyzer.plot_histogram(save_path="type1_plagiarism_histogram.png")


if __name__ == "__main__":
    main()
