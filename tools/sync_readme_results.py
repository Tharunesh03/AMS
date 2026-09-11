#!/usr/bin/env python3
"""Regenerate the results section of README.md from ``models/metrics.json``.

The README promises that its numbers come from the artifacts, so they are literally copied from
them: run this after ``python train.py`` (and after ``python -m ams evaluate``) and the section
between ``<!-- RESULTS:BEGIN -->`` / ``<!-- RESULTS:END -->`` is rewritten. Nothing is typed by
hand, so a stale number cannot survive a re-run.

    python tools/sync_readme_results.py [--models-dir models] [--readme README.md] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def _extra_arms(tl: dict) -> str:
    """Ablation arms are trained too but were never eligible - keep them out of the pool count."""
    all_arms = tl.get("trained_from_scratch_including_ablations") or []
    pool = tl.get("trained_from_scratch") or []
    extra = len(all_arms) - len(pool)
    return f" (plus {extra} ablation arm{'s' if extra != 1 else ''}, reported only)" if extra > 0 else ""


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_block(models_dir: Path) -> str:
    metrics = json.loads((models_dir / "metrics.json").read_text())
    info = json.loads((models_dir / "model_info.json").read_text())
    task = metrics["tasks"]["ai_detector"]
    m = task["metrics"]
    sel = metrics["selection"]
    ds = metrics["dataset"].get("inspection") or metrics["dataset"]
    rows = metrics["comparison"]

    lines: list[str] = []
    n_img = ds.get("n_valid") or 0
    scope = "full corpus" if int(n_img) >= 10_000 else f"{int(n_img)} images (reduced run)"
    lines.append(f"Measured on this machine in a single run (seed {metrics['split'].get('seed')}), {scope}, "
                 f"{metrics.get('generated_at', '')}. Regenerated from `models/metrics.json` by "
                 f"`python tools/sync_readme_results.py` - no number here is typed by hand.\n")
    lines.append(f"* dataset: {_fmt(ds.get('n_valid'))} usable images of {_fmt(ds.get('n_images'))} found, "
                 f"classes {', '.join(f'{k} {_fmt(v)}' for k, v in (ds.get('classes_found') or {}).items())}, "
                 f"all {', '.join(ds.get('sizes') or {}) or '?'} px; "
                 f"corrupt {_fmt(ds.get('n_corrupt'))}, exact duplicates {_fmt(ds.get('n_duplicate_exact'))}, "
                 f"near-duplicates {_fmt(ds.get('n_duplicate_near'))}, capture groups {_fmt(ds.get('n_groups'))}")
    sizes = metrics["split"]["sizes"]
    lines.append(f"* splits: train {_fmt(sizes['train'])} / val {_fmt(sizes['val'])} / test {_fmt(sizes['test'])} "
                 f"({metrics['split']['strategy']})")
    params = info.get("params") or {}
    if isinstance(params, dict):
        params = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ("feature_dim", "train_samples"))
    lines.append(f"* selected production model: **{info['architecture']}** ({info['framework']}; {params}) — "
                 f"{info['image_size']} px input, {_fmt(info.get('model_size_mb'), 2)} MB, "
                 f"{_fmt(info.get('inference_ms_per_image'), 1)} ms/image on CPU")
    tl0 = metrics.get("transfer_learning") or {}
    if tl0:
        n_deep_pool = sum(1 for r in rows if r.get("family") == "deep")
        scratch = list(tl0.get("trained_from_scratch") or [])
        extra = [k for k in (tl0.get("trained_from_scratch_including_ablations") or []) if k not in scratch]
        if tl0.get("weights_applied_for"):
            pre = "ImageNet weights loaded for " + ", ".join(tl0["weights_applied_for"])
        elif tl0.get("requested"):
            pre = (f"requested, but no weights could be loaded in the training environment: all {n_deep_pool} "
                   f"deep candidates in the pool trained from scratch"
                   + (f" (plus {len(extra)} ablation arm{'s' if len(extra) != 1 else ''}, reported only)"
                      if extra else ""))
        else:
            pre = "not requested for this run (--no-pretrained)"
        pre += (f"; frozen-backbone arms actually run: {', '.join(tl0['frozen_backbone_applied_for'])}"
                if tl0.get("frozen_backbone_applied_for")
                else "; no backbone was frozen, since freezing a randomly initialised one is not transfer learning")
    else:
        pre = "ImageNet transfer applied" if info.get("pretrained_backbone") else \
            f"no transfer learning ({info.get('pretrained_note') or 'weights unavailable'})"
    lines.append(f"* transfer learning: {pre} (full record in §9.3)")
    lines.append(f"* decision threshold {task['threshold']:.3f} ({task.get('threshold_policy', 'selected on validation')}), "
                 f"test split evaluated {task.get('test_evaluations', 1)} time(s)\n")

    lines.append("## 9.1 Comparison (test split, all candidates)\n")
    lines.append("| rank | candidate | family | accuracy | F1 | ROC-AUC | recall | precision | FPR | EER | ms/img | MB | score |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mm = r.get("metrics") or {}
        mark = "**" if r.get("selected") else ""
        lines.append(
            f"| {r.get('rank', '-')} | {mark}{r['name']}{mark} | {r['family']} | {_fmt(r.get('accuracy'))} | "
            f"{_fmt(r.get('f1'))} | {_fmt(r.get('roc_auc'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('precision'))} | "
            f"{_fmt(mm.get('false_positive_rate'))} | {_fmt(mm.get('eer'))} | {_fmt(r.get('inference_ms'), 1)} | "
            f"{_fmt(r.get('size_mb'), 2)} | {mark}{_fmt(r.get('final_score'))}{mark} |"
        )
    lines.append("")
    lines.append(f"Selection: {str(sel['reason']).rstrip('.')}.")
    best = max(rows, key=lambda r: float(r.get("f1") or 0.0))
    sel_row = next((r for r in rows if r.get("selected")), rows[0])
    if best.get("name") != sel_row.get("name"):
        lines.append(
            f"The most accurate candidate on the test split was **{best['name']}** "
            f"(F1 {_fmt(best.get('f1'))}, ROC-AUC {_fmt(best.get('roc_auc'))}, "
            f"accuracy {_fmt(best.get('accuracy'))}), and it was *not* selected: the deployment score "
            f"also weighs latency and artifact size ({_fmt(best.get('size_mb'), 2)} MB @ "
            f"{_fmt(best.get('inference_ms'), 1)} ms versus {_fmt(sel_row.get('size_mb'), 2)} MB @ "
            f"{_fmt(sel_row.get('inference_ms'), 1)} ms for {sel_row['name']}). Deep models were more "
            f"accurate here and a shallow one still won the stated rule - the table is the evidence, "
            f"and picking the larger model would have meant ignoring the published weights.\n"
        )
    if "margin_scores_mapped_to_p" in str(info.get("params")):
        lines.append(
            f"The selected model is a linear margin classifier whose decision values are squashed through "
            f"a logistic link to obtain probabilities, so the numbers are indicative rather than calibrated: "
            f"measured ECE {_fmt(task.get('calibration_ece'))} on the test split, brier "
            f"{_fmt(m.get('brier'))} (per-bin reliability table in `models/metrics.json`). The ranking of "
            f"scores - what the threshold acts on - is unaffected.\n"
        )

    lines.append("## 9.2 Selected model on the held-out test split\n")
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "specificity", "false_positive_rate", "false_negative_rate", "eer", "log_loss", "brier", "mcc"]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---:|" * len(keys))
    lines.append("| " + " | ".join(_fmt(m.get(k)) for k in keys) + " |")
    cm = m.get("confusion_matrix")
    if cm:
        cls = task.get("classes") or ["authentic", "synthetic"]
        lines.append(
            f"\nConfusion over {sum(sum(r) for r in cm)} test images, rows = true class, columns = predicted "
            f"(`{cls[0]}`, `{cls[1]}`): TN {cm[0][0]}, FP {cm[0][1]}, FN {cm[1][0]}, TP {cm[1][1]}. "
            f"Calibration error (ECE) {_fmt(task.get('calibration_ece'))}.\n"
        )

    gen = metrics.get("generator_classifier") or {}
    gtask = (metrics.get("tasks", {}) or {}).get("generator_classifier") or gen.get("task") or {}
    lines.append("## 9.3 Secondary tasks\n")
    if gtask.get("trained"):
        gm = gtask.get("metrics") or {}
        gate = gtask.get("unknown_generator_gate") or {}
        lines.append(
            f"* generator fingerprinting: **trained** (`{gtask.get('selected_model')}`, "
            f"{len(gtask.get('classes') or [])} classes `{gtask.get('classes')}`, per-class counts "
            f"{gtask.get('per_class_counts')}) — test macro-F1 {_fmt(gm.get('macro_f1'))}, weighted-F1 "
            f"{_fmt(gm.get('weighted_f1'))}, accuracy {_fmt(gm.get('accuracy'))}, top-3 "
            f"{_fmt(gm.get('top3_accuracy'))} on "
            f"{_fmt(int(gm['n'])) if gm.get('n') is not None else '-'} held-out images. Unknown-generator rule "
            f"(fitted on validation): a name is returned only at max-prob ≥ {_fmt(gate.get('min_prob'))} "
            f"with margin ≥ {_fmt(gate.get('min_margin'))}, which measured "
            f"{_fmt(gate.get('validation_precision_on_accepted'))} precision at "
            f"{_fmt(gate.get('validation_coverage'))} coverage; anything else is UNKNOWN."
        )
    else:
        lines.append(
            f"* generator fingerprinting: **not trainable on this dataset** — {gen.get('reason', '')} "
            f"({gen.get('n_classes', 0)} generator classes found). The second task itself is implemented and "
            f"tested (`ams/generator.py`, `tests/test_generator_task.py`); only the labels are missing, and "
            f"no classes were invented to work around that."
        )
    face = metrics.get("face_model") or {}
    face_reason = str(face.get("reason", "")).rstrip(".")
    face_extra = "" if "analytical CV module" in face_reason else \
        " An analytical CV module ships instead (`ams/forensics.py`), labelled as heuristics."
    lines.append(f"* face/deepfake model: trained={face.get('trained')} — {face_reason}.{face_extra}")
    tl = metrics.get("transfer_learning") or {}
    if tl:
        lines.append(
            f"* transfer learning: requested={tl.get('requested')}, weights actually loaded for "
            f"`{tl.get('weights_applied_for') or 'none'}`, every deep candidate in the selection pool "
            f"({len(tl.get('trained_from_scratch') or [])}) trained from scratch{_extra_arms(tl)}, "
            f"frozen-backbone arms applied `{tl.get('frozen_backbone_applied_for') or 'none'}` — "
            f"{tl.get('note')}\n"
        )

    abl = metrics.get("ablations")
    if abl:
        lines.append("## 9.4 Ablations (reported, never eligible for selection)\n")
        lines.append(json.dumps(abl, indent=2, default=str))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--check", action="store_true", help="exit 1 if README is out of date (no write)")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not (models_dir / "metrics.json").exists():
        print(f"error: {models_dir / 'metrics.json'} not found - run `python train.py` first", file=sys.stderr)
        return 2
    block = build_block(models_dir)
    path = Path(args.readme)
    text = path.read_text() if path.exists() else ""
    if BEGIN not in text or END not in text:
        print(f"error: {path} is missing the {BEGIN} / {END} markers", file=sys.stderr)
        return 2
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n" + block + END + tail
    if args.check:
        if new.strip() != text.strip():
            print("README results section is out of date - run tools/sync_readme_results.py", file=sys.stderr)
            return 1
        print("README results section matches models/metrics.json")
        return 0
    path.write_text(new)
    print(f"updated {path.name} results section from {models_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
