"""Module pipeline/gnn_train_program/eval.py."""

from __future__ import annotations

from typing import Iterator, Tuple
import torch
import torch.nn.functional as F

from gnn_train_program.program_store import ProgramStore
from gnn_train_program.encode import encode_program_mean

Pair = Tuple[str, str, int]


@torch.no_grad()
def eval_pairs(
    *,
    enc,
    clf,
    store: ProgramStore,
    pair_iter: Iterator[Pair],
    device: str,
    num_pairs: int,
    max_tries: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate on a streaming iterator of (path_a, path_b, y).
    We keep drawing until we collect `num_pairs` valid examples, or we hit `max_tries`.
    """
    enc.eval()
    clf.eval()

    if max_tries is None:
        max_tries = max(num_pairs * 5, 1000)

    losses = []
    correct = 0
    used = 0
    pos = 0

    tries = 0
    while used < num_pairs and tries < max_tries:
        tries += 1
        a, b, y_int = next(pair_iter)

        ma = store.load_program_methods(a)
        mb = store.load_program_methods(b)
        if not ma or not mb:
            continue

        ha = encode_program_mean(enc, ma, device)
        hb = encode_program_mean(enc, mb, device)

        # clf is expected to accept (ha, hb) and return a scalar logit
        logit = clf(ha, hb).view(1)  # shape [1]

        target = torch.tensor([float(y_int)], device=device)  # shape [1]

        loss = F.binary_cross_entropy_with_logits(logit, target)
        prob = torch.sigmoid(logit)

        losses.append(loss.detach())
        pred = 1 if float(prob.item()) >= 0.5 else 0

        correct += int(pred == int(y_int))
        pos += int(int(y_int) == 1)
        used += 1

    if used == 0:
        if verbose:
            print("[VAL] used=0 (no valid pairs found)")
        return {"val_loss": None, "val_acc": None, "used": 0, "pos": 0, "neg": 0}

    val_loss = float(torch.stack(losses).mean().item())
    val_acc = correct / used
    neg = used - pos

    if verbose:
        print(f"[VAL] used={used} pos={pos} neg={neg} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    return {"val_loss": val_loss, "val_acc": val_acc, "used": used, "pos": pos, "neg": neg}
