"""
Module for training
"""
from __future__ import annotations

import random
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gnn_train.model import MethodEncoder, PairClassifier
from gnn_train.shard_index import ShardStore, MethodLoc

Pair = Tuple[str, str, int]


def set_seed(seed: int):
    """
    Set seed for random and torch based on input
    """
    random.seed(seed)
    torch.manual_seed(seed)


def train_loop(
    *,
    pair_iter: Iterator[Pair],
    index: dict[str, MethodLoc],
    store: ShardStore,
    device: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    steps: int,
    batch_pairs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    log_every: int,
    num_edge_types: int = 4,
):
    """
    Main training loop
    """
    enc = MethodEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        dropout=dropout
        ).to(device)
    clf = PairClassifier(hidden_dim=hidden_dim, dropout=dropout).to(device)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(clf.parameters()),
        lr=lr, weight_decay=weight_decay
        )

    enc.train()
    clf.train()

    ema_loss = None

    for step in tqdm(range(1, steps + 1), desc="Training"):
        opt.zero_grad(set_to_none=True)

        losses = []
        for _ in range(batch_pairs):
            a_id, b_id, y = next(pair_iter)

            loc_a = index.get(a_id)
            loc_b = index.get(b_id)
            if loc_a is None or loc_b is None:
                continue

            ga = store.get_method(loc_a)
            gb = store.get_method(loc_b)

            xa = ga["x"].to(device)
            eia = ga["edge_index"].to(device)
            eta = ga["edge_type"].to(device)

            xb = gb["x"].to(device)
            eib = gb["edge_index"].to(device)
            etb = gb["edge_type"].to(device)

            ha = enc(xa, eia, eta)
            hb = enc(xb, eib, etb)
            logit = clf(ha, hb)

            target = torch.tensor(float(y), device=device)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            losses.append(loss)

        if not losses:
            continue

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        v = float(loss.detach().cpu())
        ema_loss = v if ema_loss is None else (0.95 * ema_loss + 0.05 * v)

        if step % log_every == 0:
            print(f"step={step} loss={v:.4f} ema={ema_loss:.4f}")

    return enc, clf
