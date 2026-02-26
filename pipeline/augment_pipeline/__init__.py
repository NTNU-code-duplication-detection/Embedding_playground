"""
Augmentation pipeline for Java program-level self-supervision.

Exports:
- AugmentConfig
- build_positive_pair
- build_negative_pair
- iter_pairs
"""

from augment_pipeline.config import AugmentConfig
from augment_pipeline.generators import build_negative_pair, build_positive_pair, iter_pairs

__all__ = [
    "AugmentConfig",
    "build_positive_pair",
    "build_negative_pair",
    "iter_pairs",
]
