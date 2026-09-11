"""Leakage-aware splitting: stratified by class, atomic by capture group.

Frames that share a ``source id`` (``0042.jpg`` and ``0042 (3).jpg``) come from one
capture and are near-duplicates of each other. If they straddle two splits, every
metric becomes optimistic, so groups - not images - are the unit of assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    groups_train: List[str]
    groups_val: List[str]
    groups_test: List[str]

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train_idx), "val": len(self.val_idx), "test": len(self.test_idx)}


def stratified_grouped_split(
    labels: Sequence[int],
    groups: Sequence[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 13,
) -> Split:
    """Deterministic group-wise stratified split (70/15/15 by default).

    For each class we shuffle its groups and hand whole groups to test, then val, then
    train, so class proportions and the per-class group structure are preserved.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=object)
    n = len(labels)
    if n != len(groups):
        raise ValueError("labels and groups must be the same length")
    if train_frac <= 0 or val_frac < 0:
        raise ValueError("train_frac must be > 0 and val_frac >= 0")
    if train_frac + val_frac >= 1.0:
        raise ValueError(f"train_frac + val_frac must stay below 1.0 (got {train_frac + val_frac:.2f})")
    tr, va = float(train_frac), float(val_frac)

    rng = np.random.default_rng(seed)
    idx_train: List[np.ndarray] = []
    idx_val: List[np.ndarray] = []
    idx_test: List[np.ndarray] = []
    g_train: List[str] = []
    g_val: List[str] = []
    g_test: List[str] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        cls_groups = groups[cls_idx]
        uniq = {}
        for i, g in zip(cls_idx, cls_groups):
            uniq.setdefault(str(g), []).append(int(i))
        keys = np.array(sorted(uniq), dtype=object)
        rng.shuffle(keys)

        n_groups = len(keys)
        fr_test, fr_val = 1.0 - tr - va, va
        n_test, n_val = _allocate_groups(n_groups, fr_test, fr_val)

        test_keys, val_keys, train_keys = keys[:n_test], keys[n_test : n_test + n_val], keys[n_test + n_val :]
        for name, keyset, buckets in (
            ("test", test_keys, (idx_test, g_test)),
            ("val", val_keys, (idx_val, g_val)),
            ("train", train_keys, (idx_train, g_train)),
        ):
            for k in keyset:
                rows = np.asarray(uniq[str(k)], dtype=np.int64)
                buckets[0].append(rows)
                buckets[1].extend([str(k)] * len(rows))

    def _cat(lst):
        return np.concatenate(lst) if lst else np.empty(0, dtype=np.int64)

    split = Split(
        train_idx=_cat(idx_train),
        val_idx=_cat(idx_val),
        test_idx=_cat(idx_test),
        groups_train=g_train,
        groups_val=g_val,
        groups_test=g_test,
    )
    _assert_disjoint(split)
    return split


def _allocate_groups(n_groups: int, fr_test: float, fr_val: float) -> Tuple[int, int]:
    """How many whole groups go to test / val (the rest to train).

    Priority: train always keeps at least one group, then test, then validation - so tiny
    classes degrade gracefully instead of producing an empty split.
    """
    if n_groups <= 1:
        return 0, 0
    if n_groups == 2:      # only test is possible without starving validation *and* training
        return 1, 0
    n_test = max(1, int(round(n_groups * fr_test)))
    n_val = max(1, int(round(n_groups * fr_val))) if fr_val > 0 else 0
    while n_test + n_val > n_groups - 1:      # keep >=1 group for training
        if n_val > n_test:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            n_val = 0
            break
    return min(n_test, n_groups - 1 - max(n_val, 0)), max(n_val, 0)


def _assert_disjoint(split: Split) -> None:
    a, b, c = set(map(int, split.train_idx)), set(map(int, split.val_idx)), set(map(int, split.test_idx))
    if a & b or a & c or b & c:
        raise AssertionError("index overlap between splits")
    gt, gv, gs = set(split.groups_train), set(split.groups_val), set(split.groups_test)
    if (gt & gv) or (gt & gs) or (gv & gs):
        raise AssertionError("a capture group leaked across two splits")


def group_leakage_report(groups: Sequence[str], labels: Sequence[int], split: Split) -> Dict[str, float]:
    """Sanity metrics recorded in the report: shared sources, class mix per split."""
    labels = np.asarray(labels)
    out: Dict[str, float] = {}
    for name, idx in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
        out[f"{name}_n"] = float(len(idx))
        out[f"{name}_pos_rate"] = float(labels[idx].mean()) if len(idx) else 0.0
    out["shared_groups_train_test"] = float(len(set(split.groups_train) & set(split.groups_test)))
    out["unique_groups"] = float(len(set(groups)))
    return out
