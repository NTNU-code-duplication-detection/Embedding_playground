"""Module pipeline/gnn_train_program/encode.py."""

from __future__ import annotations

import torch
from gnn_train.model import MethodEncoder


def encode_program_mean(
    enc: MethodEncoder,
    methods: list[dict],
    device: str,
) -> torch.Tensor:
    """
    methods: list of dicts, each with x, edge_index, edge_type.
    returns: program embedding [H] = mean(method_embeddings)
    """
    if not methods:
        # infer hidden dim from encoder output by running a dummy if needed
        # but simplest fallback: use encoder.hidden_dim if you store it; else 0-vector of 128
        # Here we assume hidden_dim=enc.hidden_dim if present, else 128.
        H = getattr(enc, "hidden_dim", 128)
        return torch.zeros((H,), device=device)

    hs = []
    for m in methods:
        x = m["x"].to(device)
        ei = m["edge_index"].to(device)
        et = m["edge_type"].to(device)
        h = enc(x, ei, et)         # [H]
        hs.append(h)

    return torch.stack(hs, dim=0).mean(dim=0)
