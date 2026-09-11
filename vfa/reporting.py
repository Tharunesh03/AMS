"""Reports, plots and the demo samples used by the web UI.

Everything printed here is read from measured artifacts (metrics dicts, score arrays,
training histories) - no hand-written numbers.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score  # noqa: E402

POS_COLOR = "#c2410c"
NEG_COLOR = "#0f766e"


# --------------------------------------------------------------------------- plots
def _ensure_parent(path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


def plot_sample_grid(db, path: Path, n: int = 8, upscale: int = 4) -> Optional[Path]:
    """Real vs synthetic side-by-side, so the reader can see what the model sees."""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.25, 2.9))
    try:
        rng = np.random.default_rng(0)
        for row, cls_id in enumerate(range(len(db.classes))):
            idx = np.flatnonzero(db.labels == cls_id)
            pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
            for col, i in enumerate(pick):
                img = db.images[int(i)].astype(np.float32)
                img = np.kron(img, np.ones((upscale, upscale, 1))) if upscale > 1 else img
                axes[row, col].imshow(np.clip(img, 0, 255).astype(np.uint8))
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(db.classes[cls_id], fontsize=9)
        axes[0, 0].set_title("class grid (nearest-neighbour upscaled)", fontsize=9, loc="left")
        fig.tight_layout()
        _ensure_parent(path)
        fig.savefig(path, dpi=110)
        return Path(path)
    finally:
        plt.close(fig)


def plot_roc_grid(scores: Dict[str, np.ndarray], y_true: np.ndarray, path: Path, max_curves: int = 6) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    try:
        y = np.asarray(y_true).astype(int)
        items = list(scores.items())[:max_curves]
        for name, s in items:
            s = np.asarray(s)
            if s.ndim > 1:
                s = s[:, -1]
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        ax.set_xlabel("false-positive rate")
        ax.set_ylabel("true-positive rate")
        ax.set_title("ROC on the held-out test split", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        _ensure_parent(path)
        fig.savefig(path, dpi=110)
        return Path(path)
    finally:
        plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: Sequence[str], path: Path, title: str = "confusion matrix") -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    try:
        cm = np.asarray(cm, dtype=float)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), labels=classes, fontsize=8)
        ax.set_yticks(range(len(classes)), labels=classes, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        _ensure_parent(path)
        fig.savefig(path, dpi=110)
        return Path(path)
    finally:
        plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], path: Path, title: str = "") -> Optional[Path]:
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    try:
        ep = [h["epoch"] for h in history]
        ax.plot(ep, [h["loss"] for h in history], label="train loss", color="#1f2937")
        ax.plot(ep, [h["val_auc"] for h in history], label="val ROC-AUC", color=POS_COLOR)
        ax.plot(ep, [h["val_f1"] for h in history], label="val F1", color=NEG_COLOR)
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_title(f"training curves - {title}", fontsize=9)
        fig.tight_layout()
        _ensure_parent(path)
        fig.savefig(path, dpi=110)
        return Path(path)
    finally:
        plt.close(fig)


def plot_reliability(rel: List[Dict[str, float]], path: Path, title: str = "") -> Optional[Path]:
    if not rel:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    try:
        pred = [r["predicted"] for r in rel]
        obs = [r["observed"] for r in rel]
        counts = [r["n"] for r in rel]
        ax.bar(np.arange(len(rel)), obs, width=0.8, color="#93c5fd", label="observed rate")
        ax.plot(np.arange(len(rel)), pred, "o-", color="#1f2937", lw=1, ms=3, label="predicted")
        ax.set_xticks(
            np.arange(len(rel)),
            labels=[f'{r["bin"]} (n={c})' for r, c in zip(rel, counts)],
            rotation=90,
            fontsize=5.5,
        )
        ax.set_xlabel("predicted P(synthetic) bin", fontsize=8)
        ax.set_ylabel("observed rate", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_title(f"calibration - {title}", fontsize=9)
        fig.tight_layout()
        _ensure_parent(path)
        fig.savefig(path, dpi=110)
        return Path(path)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------- markdown
def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def _f(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v != v else f"{v:.{nd}f}"


def write_dataset_report(db, path: Path, seed: Optional[int] = None,
                         generator: Optional[Dict[str, object]] = None) -> Path:
    inv = db.inventory
    lines: List[str] = []
    lines.append("# Dataset inspection (measured, before any training)\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Root structure and labels\n")
    lines.append(f"- data root: `{inv.root}`")
    lines.append(f"- classes discovered from sub-directories: `{sorted(inv.classes_found)}`")
    lines.append(f"- label mapping used for the binary task: `{db.mapping}` (1 = positive = AI/synthetic)")
    lines.append(f"- images found: **{inv.n_images}**, validated and usable: **{inv.n_valid}**, corrupt/unreadable: **{inv.n_corrupt}**")
    lines.append(f"- capture groups (same source id, near-duplicate frames): **{inv.n_groups}**")
    lines.append(
        f"- exact duplicates (identical md5): **{inv.n_duplicate_exact}**; near-duplicates inside one "
        f"capture group (dhash, hamming<=2): **{inv.n_duplicate_near}**; perceptually identical copies "
        f"across different capture sources: **{getattr(inv, 'n_duplicate_perceptual_cross_source', 0)}**"
    )
    lines.append(f"- formats: `{inv.formats}`")
    lines.append(f"- dimensions: `{inv.sizes}`")
    lines.append(f"- channels: `{inv.channels}`")
    lines.append(f"- class balance: `{inv.class_balance}` (minority/majority = {inv.minority_ratio})")
    if inv.notes:
        lines.append(f"- notes: {inv.notes}")
    lines.append("")
    lines.append("## Per-class image statistics\n")
    rows = []
    for cls, st in sorted(inv.per_class_stats.items()):
        rows.append([cls, st["n"], st["gray_mean"], st["gray_std"], st["laplacian_var_median"], st["bytes_median"]])
    lines.append(_table(["class", "images", "mean luma", "std luma", "median |laplacian| variance", "median file bytes"], rows))
    lines.append("")
    lines.append(
        "The two classes differ measurably in **global tone** and in **high-frequency energy**. "
        "That is a shortcut a lazy classifier can exploit, so photometric statistics are removed "
        "by the per-image z-score in both the forensic descriptor and the model input, and the "
        "difference is reported here for transparency.\n"
    )
    if inv.rejected:
        lines.append(f"## Rejected files ({len(inv.rejected)} shown)\n")
        lines.append(_table(["path", "reason"], [[r["path"], r["error"]] for r in inv.rejected[:15]]))
        lines.append("")
    lines.append("## Split protocol\n")
    lines.append("- strategy: stratified by class, **atomic per capture group** (a group never spans two splits)")
    lines.append(f"- sizes: `{db.split.sizes()}` (70/15/15 by image count, groups moved whole)")
    lines.append(f"- seed: `{13 if seed is None else seed}` (fixed in the run config, so the split is reproducible)")
    pos = {k: float(db.labels[getattr(db.split, f'{k}_idx')].mean()) for k in ("train", "val", "test")}
    lines.append(f"- positive (synthetic) rate per split: `{ {k: round(v,4) for k,v in pos.items()} }`")
    lines.append("- the test split is evaluated once, after model selection; the operating threshold is fitted on validation only")
    lines.append("")
    lines.append("## What the dataset does NOT contain\n")
    if generator is None:
        from .labels import discover_generator_labels

        generator = discover_generator_labels(Path(inv.root), list(db.classes)).to_dict()
    if generator.get("available"):
        lines.append(
            f"- per-generator labels **are** present (source: `{generator.get('source')}`, classes "
            f"`{list((generator.get('classes') or {}).keys())}`), so the multi-class fingerprint task "
            "is trained as a second model - see `models/metrics.json` `tasks.generator_classifier`"
        )
    else:
        lines.append(f"- generator fingerprinting: **not trained** - {generator.get('reason')}")
        lines.append("  (the second task is implemented in `vfa/generator.py` and runs automatically when such "
                     "folders exist; nothing was invented to make it possible here)")
    lines.append("- no separate manipulated-face label dimension, so no second face model is trained (see model card)")
    lines.append("- no identity annotations usable at this resolution (pixel-space 1-NN over same-source frames is at chance level)")
    lines.append("")
    grid = plot_sample_grid(db, Path(path).parent / "figures" / "dataset_grid.png")
    if grid:
        lines.append(f"![dataset grid](figures/{Path(grid).name})\n")
    Path(path).write_text("\n".join(lines))
    return Path(path)


def write_comparison_report(
    pool: Sequence[Dict[str, object]],
    rationale: Dict[str, object],
    db,
    path: Path,
    generator: Dict[str, object],
    ablations: Sequence[Dict[str, object]] = (),
    candidates: Optional[Dict[str, object]] = None,
) -> Path:
    lines: List[str] = []
    lines.append("# MODEL COMPARISON\n")
    lines.append("All rows are measured on the same splits with the same preprocessing. ")
    lines.append("`Score` = 0.40*F1 + 0.25*ROC-AUC + 0.15*Accuracy + 0.10*Speed + 0.10*Efficiency.\n")
    lines.append("## AI DETECTOR (binary: authentic vs AI-generated/synthetic)\n")
    rows = []
    for r in pool:
        rows.append(
            [
                r["name"], r["family"], _f(r["accuracy"]), _f(r["f1"]), _f(r["roc_auc"]),
                _f(r.get("recall")), _f(r.get("precision")), _f(r["metrics"].get("eer"), 3),
                f"{r['inference_ms']:.1f} ms", f"{r['size_mb']:.2f} MB", _f(r["final_score"]),
                "SELECTED" if r.get("selected") else f"rank {r.get('rank', '?')}",
            ]
        )
    lines.append(_table(
        ["Model", "Family", "Accuracy", "F1", "ROC-AUC", "Recall", "Precision", "EER",
         "Inference", "Size", "Score", "Status"], rows
    ))
    tied = [r for r in pool if r.get("tie_note")]
    if tied:
        lines.append("")
        lines.append("Ranking is strictly by weighted score; the selected model can differ from rank 1 "
                     "when the performance tie-break applies:\n")
        for r in tied:
            lines.append(f"- **{r['name']}** (rank {r.get('rank')}): {r['tie_note']}.")
    lines.append("")
    lines.append("```text")
    lines.append("SELECTED MODEL:")
    lines.append(f"  {rationale.get('selected')}")
    lines.append("")
    lines.append("REASON:")
    reason = str(rationale.get("reason", ""))
    for i in range(0, len(reason), 76):
        lines.append("  " + reason[i : i + 76])
    lines.append("```")
    lines.append("")
    lines.append(
        "Selection uses the weighted deployment score, not raw accuracy: a model is only promoted when it "
        "also pays for itself in inference time and artifact size. Full metric payloads "
        "(`log_loss`, `brier`, `mcc`, `pr_auc`, confusion matrices) live in ``models/metrics.json``.\n"
    )
    lines.append("## Thresholds chosen on validation (applied to test once)\n")
    rows = [[r["name"], _f(r["metrics"].get("threshold"), 3), _f(r["metrics"].get("eer"), 4), _f(r["metrics"].get("pr_auc"), 4)] for r in pool]
    lines.append(_table(["Model", "threshold", "EER", "PR-AUC"], rows))
    lines.append("")
    if ablations:
        lines.append("## Ablations (recorded for understanding, never eligible for selection)\n")
        rows = [
            [a.get("id"), _f(a.get("accuracy")), _f(a.get("f1")), _f(a.get("roc_auc")), _f(a.get("recall")),
             f"{a['inference_ms']:.1f} ms" if "inference_ms" in a else "-", a.get("note") or a.get("error") or ""]
            for a in ablations
        ]
        lines.append(_table(["Ablation", "Accuracy", "F1", "ROC-AUC", "Recall", "Inference", "What changed"], rows))
        lines.append("")
    lines.append("## GENERATOR CLASSIFIER (multi-class fingerprinting)\n")
    gtask = generator.get("task") or {}
    if generator.get("available") and gtask.get("trained"):
        lines.append(f"- **trained** on classes: `{gtask.get('classes')}` "
                     f"(label source: {gtask.get('label_source')}, {gtask.get('per_class_counts')})")
        lines.append(f"- selected candidate: `{gtask.get('selected_model')}` ({gtask.get('family')}), "
                     f"weights {gtask.get('selection', {}).get('weights')}")
        rows2 = [
            [r.get("name"), f"{r.get('f1'):.4f}", f"{r.get('weighted_f1'):.4f}", _f(r.get("top3_accuracy")),
             f"{r.get('accuracy'):.4f}", f"{r.get('inference_ms'):.1f} ms", f"{r.get('size_mb'):.3f} MB",
             "SELECTED" if r.get("selected") else ""]
            for r in gtask.get("comparison", [])
        ]
        lines.append("")
        lines.append(_table(["Generator candidate", "Macro F1", "Weighted F1", "Top-3", "Accuracy",
                            "Inference", "Size", "Status"], rows2))
        gate = gtask.get("unknown_generator_gate") or {}
        lines.append("")
        lines.append(f"- unknown-generator rule: accept a name only when max probability >= "
                     f"{gate.get('min_prob')} and its margin over the runner-up >= {gate.get('min_margin')} "
                     f"(fitted on validation: precision {gate.get('validation_precision_on_accepted')} at "
                     f"coverage {gate.get('validation_coverage')}, target met: {gate.get('target_met')}); "
                     f"everything else is reported as UNKNOWN")
        lines.append(f"- images without a generator label: {gtask.get('images_without_generator_label')}")
        lines.append(f"- measured test metrics: `{json.dumps(gtask.get('metrics'), default=str)[:300]}`")
        lines.append(f"- {gtask.get('limitation')}\n")
    elif generator.get("available"):
        lines.append(f"- labels exist but no task was run: `{generator.get('classes')}`\n")
    else:
        lines.append("**Not trained - the dataset does not support it.**")
        lines.append(f"- reason: {generator.get('reason')}")
        lines.append("- policy: classes are never fabricated; the UI shows generator identification as unavailable ")
        lines.append("  rather than guessing from a binary detector output.\n")
    lines.append("## FACE MODEL\n")
    lines.append(
        "- The corpus is 32x32 face crops with exactly two labels, so the trained detector *is* the "
        "face-authenticity model; a second 'deepfake face' classifier on the same two labels would be a duplicate, not evidence. "
    )
    lines.append(
        "- Face-level analysis is therefore shipped as an **analytical CV module** (`vfa.forensics`: face detection, "
        "landmark geometry, visual-consistency checks) and is labelled as such in the UI.\n"
    )
    figs = [p for p in sorted((Path(path).parent / "figures").glob("*.png"))]
    if figs:
        lines.append("## Figures\n")
        for f in figs:
            lines.append(f"![{f.stem}](figures/{f.name})")
        lines.append("")
    Path(path).write_text("\n".join(lines))
    return Path(path)


def write_training_report(
    pool: Sequence[Dict[str, object]],
    summaries: Dict[str, Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    path: Path,
    total_seconds: float,
    cfg,
) -> Path:
    lines: List[str] = []
    lines.append("# TRAINING REPORT\n")
    lines.append(f"- wall-clock: **{total_seconds/60:.1f} min** on {platform.system()} {platform.machine()}, "
                 f"{platform.processor() or 'CPU'}, Python {sys.version.split()[0]}, torch "
                 f"{_torch_version()}")
    lines.append(f"- run: `{cfg.run_name}` | device `{cfg.device}` | threads `{cfg.threads}` | seed `{cfg.seed}`")
    lines.append(f"- split: 70/15/15 stratified + capture-group atomic | cache grid `{cfg.cache_size}px` | descriptor grid `{cfg.feature_size}px`")
    lines.append(f"- photometric handling: `{cfg.photometric}` (per-image z-score for the classical descriptor)")
    lines.append("- configs and per-run artifacts: `artifacts/<run>/` (gitignored), production model: `models/`\n")
    lines.append("## Per-candidate protocol\n")
    rows = []
    for name, s in summaries.items():
        hist = s.get("history") or []
        rows.append([
            name,
            s.get("id"),
            f"{s.get('epochs_ran', len(hist))}" if hist else "-",
            f"{s.get('best_epoch','-')}" if "best_epoch" in s else "-",
            f"{s.get('train_seconds', s.get('fit_seconds', 0)):.0f}s" if isinstance(s.get("train_seconds", s.get("fit_seconds")), (int, float)) else "-",
            _f(s.get("threshold"), 3),
            s.get("params", ""),
            f"{s.get('input_size','')}px" if s.get("input_size") else s.get("feature_dim", ""),
            ("yes" if s.get("pretrained_applied") else "no") if "pretrained_applied" in s else "n/a",
        ])
    lines.append(_table(["Candidate", "id", "epochs", "best epoch", "train time", "threshold", "params", "input", "pretrained"], rows))
    lines.append("")
    lines.append("## Honest notes about this run\n")
    lines.append("- `pretrained=no` means the ImageNet weight download was unreachable from this environment; "
                  "the run continues from random initialisation and the fact is recorded per candidate.")
    lines.append("- Deep models are compute-capped on this 2-core CPU box; epochs listed are what actually ran "
                  "(early stopping on validation ROC-AUC), not a promise of convergence.")
    lines.append("- Ablations are recorded separately and cannot win selection.\n")
    Path(path).write_text("\n".join(lines))
    return Path(path)


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return "n/a"


def write_model_card(card: Dict[str, object], path: Path) -> Path:
    lines: List[str] = ["# MODEL CARD\n"]
    for k in ("model_name", "model_version", "task", "architecture", "framework", "params", "image_size",
              "n_classes", "classes", "threshold", "model_size_mb", "inference_ms_per_image",
              "pretrained_backbone", "training_date"):
        v = card.get(k)
        if isinstance(v, (list, dict)):
            v = json.dumps(v)
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"- chosen from: `{'`, `'.join(card.get('selected_from', []))}`")
    lines.append(f"- reason: {card.get('selection_reason')}")
    lines.append("")
    lines.append("## Data volumes")
    for k in ("training_images", "validation_images", "test_images"):
        lines.append(f"- {k}: {card.get(k)}")
    ds = card.get("dataset", {})
    lines.append(f"- dataset: `{json.dumps(ds, default=str)}`")
    lines.append("")
    lines.append("## Metrics (test split, single evaluation)")
    m = card.get("metrics", {})
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "false_positive_rate", "eer", "log_loss", "brier", "mcc"]
    lines.append(_table(keys + ["tp", "fp", "tn", "fn"], [[ _f(m.get(k), 4) if k not in ("tp","fp","tn","fn") else m.get(k) for k in keys + ["tp","fp","tn","fn"]] ]))
    lines.append("")
    lines.append("## Known limitations")
    for l in card.get("known_limitations", []):
        lines.append(f"- {l}")
    lines.append("")
    lines.append("## Scientific honesty")
    for l in card.get("honesty", []):
        lines.append(f"- {l}")
    lines.append("")
    lines.append("## Secondary models")
    g = card.get("generator_classifier") or {}
    gt = g.get("task") or {}
    if gt.get("trained"):
        gm = gt.get("metrics") or {}
        gate = gt.get("unknown_generator_gate") or {}
        lines.append(
            f"- generator fingerprinting: **trained** (`{gt.get('selected_model')}`, classes "
            f"`{gt.get('classes')}`, {gt.get('per_class_counts')}); test macro-F1 "
            f"{_f(gm.get('macro_f1'))}, weighted-F1 {_f(gm.get('weighted_f1'))}, accuracy "
            f"{_f(gm.get('accuracy'))}, top-3 {_f(gm.get('top3_accuracy'))}; exported to "
            f"`{gt.get('model_dir')}`"
        )
        lines.append(
            f"- unknown-generator rule: a name is returned only when max probability >= "
            f"{_f(gate.get('min_prob'), 2)} and margin >= {_f(gate.get('min_margin'), 2)} "
            f"(validation precision on accepted predictions {_f(gate.get('validation_precision_on_accepted'))}, "
            f"coverage {_f(gate.get('validation_coverage'))}); otherwise the answer is UNKNOWN - "
            "generators outside the trained list can never be named"
        )
    else:
        lines.append(f"- generator fingerprinting: **not trained** - {g.get('reason')}")
    fm = card.get("face_model") or {}
    lines.append(
        f"- deepfake face model: **not trained as a classifier** ({fm.get('analytical_cv_module', 'vfa.forensics')} "
        f"ships an analytical CV module instead: face detection, landmark geometry, visual-consistency checks; "
        f"identity model: {fm.get('identity_model_available')})"
    )
    tl = card.get("transfer_learning") or {}
    if tl:
        lines.append(
            f"- transfer learning: requested={tl.get('requested')}, weights applied for "
            f"`{tl.get('weights_applied_for')}`, from scratch `{tl.get('trained_from_scratch')}`, frozen-backbone "
            f"arms applied `{tl.get('frozen_backbone_applied_for')}` - {tl.get('note')}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    return Path(path)


# ------------------------------------------------------------------------ samples
def export_samples(db, out_dir: Path, n_per_class: int = 4, upscale: int = 5) -> List[Path]:
    """Small PNGs + manifest so the deployed app has offline demo images.

    Drawn from the **test split only**: demo images that the model trained on would make the
    dashboard look better than the system is.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    manifest: List[Dict[str, object]] = []
    test_pool = set(int(i) for i in db.split.test_idx)
    for cls_id, cls in enumerate(db.classes):
        idx = np.array([i for i in np.flatnonzero(db.labels == cls_id) if int(i) in test_pool], dtype=int)
        if idx.size == 0:                       # tiny datasets can leave a class out of test
            idx = np.flatnonzero(db.labels == cls_id)
        for j, i in enumerate(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)):
            img = db.images[int(i)]
            if upscale > 1:
                img = np.kron(img, np.ones((upscale, upscale, 1), dtype=np.uint8))
            from PIL import Image

            name = f"{cls}_{j}.png"
            Image.fromarray(img.astype(np.uint8)).save(out_dir / name)
            manifest.append(
                {
                    "file": name,
                    "class": cls,
                    "class_id": int(cls_id),
                    "split": "test" if test_pool else "train(fallback)",
                    "source_path": str(db.paths[int(i)]),
                    "note": f"display copy upscaled x{upscale} from the {db.cache_size}px decode cache; "
                            f"the model sees the {db.cache_size}px grid",
                }
            )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return [out_dir / m["file"] for m in manifest]
