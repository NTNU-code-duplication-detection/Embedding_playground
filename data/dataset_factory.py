"""
Dataset generator factory for all datasets

Enables streaming java code from all datasets with parameters
"""
from data.data_generators.bigclonebench_dataset_gen import samples_generator, default_generator
from data.data_generators.codexglue_dataset_gen import bigclonebench_hf_generator
from data.data_generators.sourcecodeplag_dataset_gen import (
    original_non_plagiarized_generator,
    original_plagiarized_generator
)
from data.data_generators.code_clone_dataset_gen import code_clone_dataset_generator


_DATASET_REGISTRY = {
    "bigclonebench": {
        "non_plagiarized": samples_generator,
        "plagiarized": default_generator
    },
    "codexglue": {
        "pairs": bigclonebench_hf_generator
    },
    "sourcecodeplag": {
        "plagiarized": original_plagiarized_generator,
        "non_plagiarized": original_non_plagiarized_generator
    },
    "codeclonedataset": {
        "pairs": code_clone_dataset_generator
        # **kw: dataset_root, clone_type
    },
}


def get_dataset_generator(dataset_name: str, mode: str = "plagiarized", **kwargs):
    """
    Dataset generator to all datasets.

    @param dataset_name: Duh
    @param mode: Either plagiarized or non_plagiarized, or pairs in case of codexglue
    @param **kwargs: Custom path to dataset for BCB and SCP
    """
    # bigclonebench mode: negative, default

    dataset_name = dataset_name.lower()

    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(_DATASET_REGISTRY.keys())}"
        )

    dataset_modes = _DATASET_REGISTRY[dataset_name]

    if mode not in dataset_modes:
        raise ValueError(
            f"Unknown mode '{mode}' for dataset '{dataset_name}'. "
            f"Available modes: {list(dataset_modes.keys())}"
        )

    return dataset_modes[mode](**kwargs)
