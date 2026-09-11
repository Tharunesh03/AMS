"""Model selection: weighted deployment score, never accuracy alone.

    Final Score = 0.40*F1 + 0.25*ROC-AUC + 0.15*Accuracy + 0.10*Speed + 0.10*Efficiency

``Speed`` and ``Efficiency`` are relative scores over the candidate pool:

    Speed       = min(1, median_throughput / best_throughput)
    Efficiency  = 0.5*small-size score + 0.5*light-pipeline score
                  small-size score   = min(1, size_mb_limit / size_mb)  (log-scaled)
                  light-pipeline     = 1 for classical / no-torch models, 0.85 otherwise
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_WEIGHTS = {
    "f1": 0.40,
    "roc_auc": 0.25,
    "accuracy": 0.15,
    "speed": 0.10,
    "efficiency": 0.10,
}

#: Multi-class tasks (generator fingerprinting) have no single ROC-AUC, so the pool is
#: scored on macro F1 + weighted F1 + accuracy + top-3, per the task's own evaluation rule.
GENERATOR_WEIGHTS = {
    "f1": 0.35,
    "weighted_f1": 0.20,
    "accuracy": 0.15,
    "top3_accuracy": 0.10,
    "speed": 0.10,
    "efficiency": 0.10,
}

TASK_WEIGHTS = {"ai_detector": DEFAULT_WEIGHTS, "generator": GENERATOR_WEIGHTS}


def _metric_for(key: str, m: Dict[str, object]) -> float:
    if key == "f1":
        return float(m.get("f1", m.get("macro_f1", 0.0)))
    if key == "roc_auc":
        return float(m.get("roc_auc", m.get("roc_auc_ovr", 0.5)))
    if key == "weighted_f1":
        return float(m.get("weighted_f1", m.get("macro_f1", 0.0)))
    if key == "top3_accuracy":
        v = m.get("top3_accuracy")
        return 0.0 if v is None else float(v)
    if key == "accuracy":
        return float(m.get("accuracy", 0.0))
    return float(m.get(key, 0.0) or 0.0)


@dataclass
class Candidate:
    """One row of the comparison table (all values measured, never assumed)."""

    name: str
    family: str                      # classical | deep
    task: str                        # ai_detector | generator | face
    metrics: Dict[str, float]        # accuracy / f1 / roc_auc / ...
    size_mb: float
    inference_ms: float
    pipeline: str = "torch"          # torch | sklearn
    params: Optional[int] = None
    notes: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        return 1000.0 / max(self.inference_ms, 1e-6)


def _norm_size(size_mb: float, ref_mb: float = 1.0) -> float:
    """Log-scaled size score: 1.0 at/below ``ref_mb``, decaying ~1 point per decade."""
    if size_mb <= 0:
        return 1.0
    return float(np.clip(1.0 - math.log10(max(size_mb, 1e-6) / ref_mb) / 3.0, 0.0, 1.0))


def score_pool(
    candidates: Sequence[Candidate],
    weights: Optional[Dict[str, float]] = None,
    task: str = "ai_detector",
) -> List[Dict[str, object]]:
    """Return every candidate with sub-scores and the weighted final score."""
    w = dict(TASK_WEIGHTS.get(task, DEFAULT_WEIGHTS))
    if weights:
        w.update(weights)
    pool = [c for c in candidates if c.task == task] or list(candidates)
    if not pool:
        return []
    best_tp = max(c.throughput for c in pool)
    rows: List[Dict[str, object]] = []
    for c in pool:
        speed = min(1.0, c.throughput / max(best_tp, 1e-9))
        size_score = _norm_size(c.size_mb)
        light = 1.0 if c.pipeline == "sklearn" else 0.85
        efficiency = 0.5 * size_score + 0.5 * light
        f1 = _metric_for("f1", c.metrics)
        auc = _metric_for("roc_auc", c.metrics)
        acc = _metric_for("accuracy", c.metrics)
        final = sum(w[k] * _metric_for(k, c.metrics) for k in w if k not in ("speed", "efficiency"))
        final += w.get("speed", 0.0) * speed + w.get("efficiency", 0.0) * efficiency
        rows.append(
            {
                "name": c.name,
                "family": c.family,
                "task": c.task,
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
                "accuracy": round(acc, 4),
                "weighted_f1": round(_metric_for("weighted_f1", c.metrics), 4),
                "top3_accuracy": (None if c.metrics.get("top3_accuracy") is None
                                  else round(_metric_for("top3_accuracy", c.metrics), 4)),
                "recall": float(c.metrics.get("recall", c.metrics.get("macro_recall", float("nan")))),
                "precision": float(c.metrics.get("precision", c.metrics.get("macro_precision", float("nan")))),
                "inference_ms": round(c.inference_ms, 3),
                "throughput_img_s": round(c.throughput, 1),
                "size_mb": round(c.size_mb, 3),
                "params": c.params,
                "subscores": {
                    "speed": round(speed, 4),
                    "size": round(size_score, 4),
                    "efficiency": round(efficiency, 4),
                },
                "final_score": round(final, 4),
                "metrics": c.metrics,
                "notes": c.notes,
                "artifacts": c.artifacts,
            }
        )
    rows.sort(key=lambda r: -float(r["final_score"]))  # type: ignore[arg-type]

    # Spec rule: near-identical performance -> prefer the smaller/faster model. The weighted
    # score alone rarely produces this because cost carries weight, so the tie-break is applied
    # explicitly on the *performance* gap and recorded in the row.
    for i, r in enumerate(rows):
        r["rank"] = i + 1                                   # rank is strictly by weighted score
    winner = rows[0]
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        perf_keys = [k for k in ("f1", "roc_auc", "accuracy", "weighted_f1", "top3_accuracy")
                     if k in a and k in b and a.get(k) is not None and b.get(k) is not None]
        perf_gap = max(abs(float(a[k]) - float(b[k])) for k in perf_keys[:3]) if perf_keys else 1.0
        if perf_gap < 0.01:                                  # statistically equal accuracy -> cheap wins
            cost = lambda r: float(r["size_mb"]) + float(r["inference_ms"]) / 100.0
            cheap, pricey = (a, b) if cost(a) <= cost(b) else (b, a)
            if cheap is not a:
                cheap["tied_with"] = a["name"]
                cheap["perf_gap_to_leader"] = round(perf_gap, 5)
                cheap["tie_note"] = (
                    f"{'/'.join(k for k in perf_keys[:3])} within {perf_gap:.4f} of the score leader "
                    f"{a['name']}, while being the cheaper deployment ({cheap['size_mb']} MB @ "
                    f"{cheap['inference_ms']} ms vs {a['size_mb']} MB @ {a['inference_ms']} ms)"
                )
                winner = cheap
    for r in rows:
        r["selected"] = r is winner
    return rows


def winner_row(rows: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """The row the selection rule picks (explicit ``selected`` flag, else rank 1)."""
    for r in rows:
        if r.get("selected"):
            return r
    return rows[0] if rows else None


def selection_rationale(
    rows: Sequence[Dict[str, object]], weights: Optional[Dict[str, float]] = None
) -> Dict[str, object]:
    """Human-readable justification, including the "tie -> smaller/faster" rule."""
    if not rows:
        return {"selected": None, "reason": "no candidates"}
    best = winner_row(rows)
    assert best is not None
    second = next((r for r in rows if r is not best), None)
    w = dict(weights or DEFAULT_WEIGHTS)
    shown = [k for k in w if k not in ("speed", "efficiency")]
    if shown == ["f1", "roc_auc", "accuracy"] or not shown:
        parts = f"F1={best['f1']:.4f}, ROC-AUC={best['roc_auc']:.4f}, accuracy={best['accuracy']:.4f}"
    else:
        parts = ", ".join(f"{k}={float(best.get(k, 0.0) or 0.0):.4f}" for k in shown)
    reason = (
        f"deployment score {float(best['final_score']):.4f} from {parts}, "
        f"{best['inference_ms']:.1f} ms/image, {best['size_mb']:.2f} MB"
    )
    if second is not None:
        gap = float(best["final_score"]) - float(second["final_score"])
        perf_keys = [k for k in ("f1", "roc_auc", "accuracy", "weighted_f1", "top3_accuracy")
                     if k in best and k in second and best.get(k) is not None and second.get(k) is not None]
        perf_gap = max(abs(float(best[k]) - float(second[k])) for k in perf_keys[:3]) if perf_keys else 1.0
        if best.get("tie_note"):
            reason += f"; selected by tie-break: {best['tie_note']}."
        elif perf_gap < 0.01:
            reason += (f"; performance within {perf_gap:.4f} of {second['name']} and this is already the "
                       f"cheaper artifact, so no tie-break was needed")
        elif gap > 0:
            reason += f"; beats runner-up {second['name']} ({float(second['final_score']):.4f}) by {gap:.4f}."
        else:
            reason += f"; ranked {best['rank']} of {len(rows)} on the weighted score."
    return {
        "selected": best["name"],
        "selected_family": best["family"],
        "reason": reason,
        "score_gap_to_runner_up": round(float(best["final_score"]) - float(rows[0]["final_score"]), 4)
        if rows and rows[0] is not best
        else (round(float(best["final_score"]) - float(second["final_score"]), 4) if second else None),
        "selected_rank": int(best.get("rank", 1)),
        "tie_break_applied": bool(best.get("tie_note")),
    }
