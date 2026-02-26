"""
Train module
"""
from __future__ import annotations

import random
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gnn_train.model import MethodEncoder, PairClassifier
from gnn_train_program.program_store import ProgramStore
from gnn_train_program.encode import encode_program_mean
from gnn_train_program.eval import eval_pairs

Pair = Tuple[str, str, int]  # (program_path_a, program_path_b, label)


def set_seed(seed: int):
    """
    Set seed for random and torch
    """
    random.seed(seed)
    torch.manual_seed(seed)


def train_loop_program(
    *,
    pair_iter: Iterator[Pair],
    store: ProgramStore,
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
    seed: int = 0,
    eval_every: int = 0,
    val_iter: Iterator[Pair] | None = None,
    val_pairs: int = 200,
):
    """
    Main train loop
    """
    set_seed(seed)

    enc = MethodEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_edge_types=4, dropout=dropout).to(device)
    clf = PairClassifier(hidden_dim=hidden_dim, dropout=dropout).to(device)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(clf.parameters()), lr=lr, weight_decay=weight_decay)

    enc.train()
    clf.train()

    ema_loss = None

    for step in tqdm(range(1, steps + 1), desc="Training(program)"):
        opt.zero_grad(set_to_none=True)

        losses = []
        for _ in range(batch_pairs):
            a_path, b_path, y = next(pair_iter)

            ma = store.load_program_methods(a_path)
            mb = store.load_program_methods(b_path)
            if ma is None or mb is None:
                continue

            ha = encode_program_mean(enc, ma, device)
            hb = encode_program_mean(enc, mb, device)

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
            if eval_every and val_iter is not None and (step % eval_every == 0):

                metrics = eval_pairs(
                    enc=enc,
                    clf=clf,
                    store=store,
                    pair_iter=val_iter,
                    device=device,
                    num_pairs=val_pairs,
                )
                print(f"val_loss={metrics['val_loss']} val_acc={metrics['val_acc']}")
                enc.train()
                clf.train()

    return enc, clf
