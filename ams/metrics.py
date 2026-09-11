"""Metric helpers: binary + multiclass scores, threshold selection, calibration, timing."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)


def binary_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Full binary scorecard. ``score`` is the probability of the positive (AI/synthetic) class."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    yhat = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    n_neg = max(tn + fp, 1)
    n_pos = max(tp + fn, 1)
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "specificity": float(tn / n_neg),
        "false_positive_rate": float(fp / n_neg),
        "false_negative_rate": float(fn / n_pos),
        "mcc": float(matthews_corrcoef(y, yhat)) if (tn + fp + fn + tp) else 0.0,
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, s))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y, s))
        out["average_precision"] = out["pr_auc"]
    except ValueError:
        out["pr_auc"] = float("nan")
    if len(np.unique(y)) > 1 and np.all((s >= 0) & (s <= 1)):
        out["log_loss"] = float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6)))
        out["brier"] = float(brier_score_loss(y, s))
    eer, thr = equal_error_rate(y, s)
    out["eer"] = float(eer)
    out["eer_threshold"] = float(thr)
    return out


def multiclass_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str],
    top_k: int = 3,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=float)
    yhat = P.argmax(axis=1)
    k = min(top_k, P.shape[1])
    out: Dict[str, float] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_precision": float(precision_score(y, yhat, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, yhat, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yhat, average="weighted", zero_division=0)),
    }
    if P.shape[1] > 1:
        if P.shape[1] > top_k:                 # with <=3 classes a top-3 score always hits
            try:
                out["top3_accuracy"] = float(top_k_accuracy_score(y, P, k=k, labels=np.arange(len(classes))))
            except Exception:  # noqa: BLE001
                out["top3_accuracy"] = float("nan")
        else:
            out["top3_accuracy"] = None
            out["top3_note"] = (
                f"not reported: the task has {P.shape[1]} classes, so every prediction is inside the "
                "top 3 by construction and the metric would be a constant 1.0"
            )
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y, P, multi_class="ovr"))
        except Exception:  # noqa: BLE001
            out["roc_auc_ovr"] = float("nan")
        out["mean_top_prob"] = float(P.max(axis=1).mean())
    out["confusion_matrix"] = confusion_matrix(y, yhat, labels=list(range(len(classes)))).tolist()
    out["per_class_f1"] = {
        str(classes[i]): float(v) for i, v in enumerate(f1_score(y, yhat, average=None, zero_division=0))
    }
    return out


def equal_error_rate(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    """Threshold where FPR == FNR (the biometric operating point used for the UI gate)."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, score)
    idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[idx] + (1 - tpr[idx])) / 2.0)
    thr_v = float(thr[idx]) if idx < len(thr) else 0.5
    return eer, thr_v


def select_threshold(y_val: np.ndarray, score_val: np.ndarray, policy: str = "f1") -> float:
    """Pick the operating threshold on *validation* data only (never on test).

    Candidates are the midpoints between distinct validation scores rather than the score
    values themselves: with distance-weighted k-NN the scores are exactly {0, 1} and a
    candidate taken from a score value can land on 1.0, which a logistic model would never
    exceed at inference time. Midpoints sit inside the gap, are equivalent on validation and
    generalise better at serving time.
    """
    y = np.asarray(y_val).astype(int).ravel()
    sc = np.asarray(score_val, dtype=np.float64).ravel()
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(-sc, kind="mergesort")
    ys, ss = y[order], sc[order]
    cuts = np.flatnonzero(ss[:-1] > ss[1:])               # valid split points (strictly different scores)
    if cuts.size == 0:
        return 0.5
    tp = np.cumsum(ys)[cuts]
    n_pos = float(ys.sum())
    fp = np.arange(1, ss.size + 1)[cuts] - tp
    n_neg = float(ys.size - n_pos)
    tpr = tp / max(n_pos, 1.0)
    fpr = fp / max(n_neg, 1.0)
    fnr = 1.0 - tpr
    precision = tp / np.clip(tp + fp, 1e-9, None)
    recall = tpr
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    mid = 0.5 * (ss[cuts] + ss[cuts + 1])
    if policy == "youden":
        i = int(np.argmax(tpr - fpr))
    elif policy == "eer":
        i = int(np.argmin(np.abs(fpr - fnr)))
    else:
        i = int(np.argmax(f1))
    return float(mid[i])


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    """Calibration: predicted vs observed rate per probability bin."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "observed": float(y[m].mean()),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    rows = reliability_table(y, p, bins)
    n = sum(r["n"] for r in rows) or 1
    return float(sum(r["n"] / n * abs(r["observed"] - r["predicted"]) for r in rows))


def timed(fn, repeats: int = 3, warmup: int = 1, **kwargs) -> Tuple[object, Dict[str, float]]:
    """Run ``fn`` and return (result, timing) - used for the deployment score."""
    import time

    for _ in range(warmup):
        fn(**kwargs)
    ts = []
    res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(**kwargs)
        ts.append(time.perf_counter() - t0)
    return res, {"ms": float(np.mean(ts) * 1000), "ms_sd": float(np.std(ts) * 1000), "repeats": int(repeats)}
