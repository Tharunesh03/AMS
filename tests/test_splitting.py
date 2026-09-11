"""Split integrity: no capture group may straddle splits, class ratios must hold."""

from __future__ import annotations

import numpy as np
import pytest

from ams.splitting import group_leakage_report, stratified_grouped_split


def _synthetic(n_per_class=2000, groups_per_class=200, frames=10):
    labels, groups = [], []
    for cls in (0, 1):
        for g in range(groups_per_class):
            for f in range(frames):
                labels.append(cls)
                groups.append(f"{'real' if cls == 0 else 'fake'}/{g:04d}")
    return np.array(labels), np.array(groups, dtype=object)


def test_sizes_follow_the_requested_ratio():
    labels, groups = _synthetic()
    sp = stratified_grouped_split(labels, groups, 0.70, 0.15, seed=11)
    sizes = sp.sizes()
    assert abs(sizes["train"] / 4000 - 0.70) < 0.03
    assert abs(sizes["val"] / 4000 - 0.15) < 0.03
    assert abs(sizes["test"] / 4000 - 0.15) < 0.03
    assert sizes["train"] + sizes["val"] + sizes["test"] == 4000


def test_no_index_or_group_overlap_between_splits():
    labels, groups = _synthetic()
    sp = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=5)
    a, b, c = set(sp.train_idx.tolist()), set(sp.val_idx.tolist()), set(sp.test_idx.tolist())
    assert not (a & b) and not (a & c) and not (b & c)
    assert not set(sp.groups_train) & set(sp.groups_test)
    assert not set(sp.groups_val) & set(sp.groups_test)
    rep = group_leakage_report(groups, labels, sp)
    assert rep["shared_groups_train_test"] == 0.0


def test_a_whole_group_moves_together():
    """Every image of a capture group lands in exactly one split."""
    labels, groups = _synthetic(n_per_class=500, groups_per_class=50, frames=10)
    sp = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=1)
    idx_group = {}
    for i, g in enumerate(groups):
        idx_group.setdefault(str(g), set()).add(i)
    parts = {k: set(int(i) for i in getattr(sp, f"{k}_idx")) for k in ("train", "val", "test")}
    for g, members in idx_group.items():
        hits = [len(members & p) for p in parts.values()]
        assert sum(h for h in hits if h) == len(members)
        assert sum(1 for h in hits if h) == 1, f"group {g} leaked across splits: {hits}"


def test_stratification_keeps_class_balance_per_split():
    labels = np.array([0] * 3000 + [1] * 1000)          # imbalanced corpus
    groups = np.array([f"c{int(l)}/s{int(l)*100 + i // 10:04d}" for i, l in enumerate(labels)], dtype=object)
    sp = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=2)
    for split in ("train_idx", "val_idx", "test_idx"):
        rate = float(labels[getattr(sp, split)].mean())
        assert abs(rate - 0.25) < 0.05, f"{split} positive rate drifted to {rate}"


def test_deterministic_for_a_given_seed():
    labels, groups = _synthetic(n_per_class=400, groups_per_class=40)
    a = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=42)
    b = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=42)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert a.groups_test == b.groups_test


def test_small_dataset_is_still_partitioned_exactly_once():
    """With a single group per class the group cannot be split - it must land in train."""
    labels = np.array([0, 0, 1, 1])
    groups = np.array(["a/1", "a/1", "b/1", "b/1"], dtype=object)
    sp = stratified_grouped_split(labels, groups, 0.7, 0.15, seed=0)
    allidx = np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx])
    assert sorted(allidx.tolist()) == [0, 1, 2, 3]
    assert len(set(allidx.tolist())) == 4
    assert len(sp.train_idx) == 4


def test_modest_dataset_gets_all_three_splits():
    """3 capture groups per class is the minimum to carve test + val + train."""
    labels = np.array([0] * 6 + [1] * 6)
    groups = np.array([f"a/{i // 2 + 1}" for i in range(6)] + [f"b/{i // 2 + 1}" for i in range(6)], dtype=object)
    sp = stratified_grouped_split(labels, groups, 0.5, 0.25, seed=0)
    assert len(sp.test_idx) >= 2 and len(sp.val_idx) >= 2 and len(sp.train_idx) >= 2
    assert sorted(np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx]).tolist()) == list(range(12))


def test_two_groups_per_class_prefers_test_over_validation():
    labels = np.array([0, 0, 0, 0])
    groups = np.array(["a/1", "a/1", "a/2", "a/2"], dtype=object)
    sp = stratified_grouped_split(labels, groups, 0.5, 0.25, seed=0)
    assert len(sp.test_idx) == 2 and len(sp.train_idx) == 2 and len(sp.val_idx) == 0


def test_rejects_bad_fractions():
    labels, groups = _synthetic(n_per_class=100, groups_per_class=10)
    with pytest.raises(ValueError):
        stratified_grouped_split(labels, groups, 0.9, 0.5, seed=0)
