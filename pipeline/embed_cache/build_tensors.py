"""
Module for building tensors
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

EDGE_TYPE_TO_ID = {
    "SEQ": 0,
    "AST": 1,
    "IF_THEN": 2,
    "IF_ELSE": 3,
}

def edges_to_tensors(edges: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    edges: list of {"src": int, "dst": int, "type": str}
    returns:
      edge_index: [2, E] long
      edge_type:  [E] long
    """
    src, dst, et = [], [], []
    for e in edges:
        t = e.get("type")
        if t not in EDGE_TYPE_TO_ID:
            continue
        src.append(int(e["src"]))
        dst.append(int(e["dst"]))
        et.append(int(EDGE_TYPE_TO_ID[t]))

    if len(src) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(et, dtype=torch.long)
    return edge_index, edge_type
