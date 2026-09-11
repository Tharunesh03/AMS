"""End-to-end orchestrator used by ``train.py``.

    dataset inspection -> validation -> preprocessing -> candidate training
    -> evaluation -> comparison -> selection -> production artifacts -> report

Every step writes machine-readable output under ``models/`` and human-readable output
under ``reports/``. Nothing here retrains when the app serves predictions: the app only
loads what this pipeline produced.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import FeatureOptions
from .labels import discover_generator_labels
from .metrics import binary_metrics, reliability_table, expected_calibration_error
from .models import CANDIDATE_PRESETS, classical_presets
from .reporting import (
    write_comparison_report,
    write_dataset_report,
    write_model_card,
    write_training_report,
    plot_roc_grid,
    plot_confusion,
    plot_training_curves,
    export_samples,
)
from .selection import Candidate, score_pool, selection_rationale
from .trainer import DataBundle, build_data, forensic_matrix, train_classical, train_deep

VERSION = "0.1.0"


@dataclass
class PipelineConfig:
    root: Path = Path("data")
    archive: Optional[Path] = Path("archive (4).zip")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    samples_dir: Path = Path("samples")
    cache_size: int = 64
    feature_size: int = 64
    max_images_per_class: Optional[int] = None
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 13
    run_name: str = "full"
    deep: bool = True
    classical: bool = True
    pretrained: bool = True
    deep_ids: Optional[Sequence[str]] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    patience: int = 5
    device: str = "cpu"
    threads: int = 2
    rebuild_cache: bool = False
    export_samples_n: int = 4
    verbose: bool = True
    # ablations recorded in the report (never used for selection)
    ablations: Sequence[Dict[str, object]] = ()
    photometric: str = "gray"
    generator_task: bool = True   # §9/§10: multi-class fingerprinting, only if the labels exist

    def to_json(self) -> str:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        return json.dumps(d, indent=2)


@dataclass
class RunResult:
    selected: Dict[str, object]
    candidates: List[Dict[str, object]]
    metrics: Dict[str, object]
    paths: Dict[str, str]
    generator: Dict[str, object]
    inventory: Dict[str, object]
    generator_task: Optional[Dict[str, object]] = None


def _log(cfg: PipelineConfig, msg: str) -> None:
    if cfg.verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_training(cfg: PipelineConfig) -> RunResult:
    t_start = time.time()
    cfg.artifacts_dir = Path(cfg.artifacts_dir)
    run_dir = cfg.artifacts_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_config.json").write_text(cfg.to_json())

    _log(cfg, f"step 1/8: inspecting dataset at {cfg.root}")
    db = build_data(
        root=cfg.root,
        cache_dir=cfg.artifacts_dir / "cache",
        archive=cfg.archive if cfg.archive and Path(cfg.archive).exists() else None,
        cache_size=cfg.cache_size,
        max_images_per_class=cfg.max_images_per_class,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        rebuild_cache=cfg.rebuild_cache,
        verbose=cfg.verbose,
        inventory_path=Path(cfg.models_dir) / "dataset_inspection.json",
    )
    inv = db.inventory
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)

    _log(cfg, f"step 2/8: label dimensions -> binary={db.binary}")
    gen = discover_generator_labels(cfg.root, db.classes)
    if not gen.available:
        _log(cfg, f"        generator fingerprinting: SKIPPED ({gen.reason[:90]}...)")

    write_dataset_report(db, Path(cfg.reports_dir) / "eda.md", seed=cfg.seed, generator=gen.to_dict())

    candidates: List[Candidate] = []
    summaries: Dict[str, Dict[str, object]] = {}

    X_forensic: Optional[np.ndarray] = None
    names: List[str] = []
    if cfg.classical:
        _log(cfg, f"step 3/8: forensic descriptor @ {cfg.feature_size}px for {len(classical_presets())} classical candidates")
        X_forensic, names = forensic_matrix(
            db, target_size=cfg.feature_size, work_dir=run_dir,
            opts=FeatureOptions(photometric=cfg.photometric), progress=cfg.verbose,
        )

    # ---- classical candidates
    if cfg.classical:
        for preset in classical_presets():
            _log(cfg, f"step 4/8: classical candidate {preset['id']}")
            work = run_dir / "classical" / str(preset["id"])
            cand, summ = train_classical(db, preset, work, X_forensic=X_forensic, feature_names=names)
            candidates.append(cand)
            summaries[cand.name] = summ

    # ---- deep candidates
    if cfg.deep:
        presets = [p for p in CANDIDATE_PRESETS if (cfg.deep_ids is None or p["id"] in list(cfg.deep_ids))]
        for preset in presets:
            _log(cfg, f"step 5/8: deep candidate {preset['id']} (pretrained={cfg.pretrained})")
            work = run_dir / "deep" / str(preset["id"])
            try:
                cand, summ = train_deep(
                    db, preset, work, epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                    pretrained=cfg.pretrained, device=cfg.device, torch_threads=cfg.threads,
                    patience=cfg.patience, progress=cfg.verbose,
                )
            except Exception as exc:  # noqa: BLE001 - never lose the whole run to one candidate
                _log(cfg, f"        !! {preset['id']} failed: {type(exc).__name__}: {exc}")
                continue
            candidates.append(cand)
            summaries[cand.name] = summ
            hist = summ.get("history") or []
            if hist:
                plot_training_curves(hist, Path(cfg.reports_dir) / "figures" / f"curves_{cand.name}.png", title=cand.name)

    if not candidates:
        raise RuntimeError("no candidate model trained successfully - nothing to select")

    # ---- ablations (recorded, not eligible for selection)
    ablation_rows: List[Dict[str, object]] = []
    for abl in cfg.ablations:
        _log(cfg, f"        ablation: {abl.get('id')}")
        preset = dict(abl)
        try:
            cand, summ = train_deep(db, preset, run_dir / "ablation" / str(preset["id"]),
                                    pretrained=cfg.pretrained, device=cfg.device, torch_threads=cfg.threads,
                                    patience=cfg.patience, progress=False)
            row = {"id": cand.name, **{k: cand.metrics.get(k) for k in ("accuracy", "f1", "roc_auc", "recall")},
                   "inference_ms": cand.inference_ms, "size_mb": cand.size_mb, "note": preset.get("note", "")}
            ablation_rows.append(row)
            summaries[cand.name] = summ
        except Exception as exc:  # noqa: BLE001
            ablation_rows.append({"id": preset.get("id"), "error": f"{type(exc).__name__}: {exc}"})

    # ---- comparison + selection
    _log(cfg, "step 6/8: scoring candidates (weighted deployment score)")
    pool = score_pool(candidates, task="ai_detector" if db.binary else "multiclass")
    rationale = selection_rationale(pool)
    winner_name = rationale["selected"]
    winner = next(c for c in candidates if c.name == winner_name)

    gen_task: Optional[Dict[str, object]] = None
    if cfg.generator_task:
        _log(cfg, "step 6b: generator fingerprinting (only if the dataset labels support it)")
        try:
            gen_task = run_generator_task(cfg, db, gen.to_dict(), run_dir)
        except Exception as exc:  # noqa: BLE001 - a second task must never lose the first one
            _log(cfg, f"        !! generator task failed: {type(exc).__name__}: {exc}")
            gen_task = {"task": "generator_fingerprinting", "trained": False,
                        "error": f"{type(exc).__name__}: {exc}"}
        if gen_task is None:
            _log(cfg, "        generator task: not trainable on this dataset (recorded, nothing invented)")
    gen_payload = gen.to_dict()
    if gen_task:
        gen_payload["task"] = gen_task

    _log(cfg, f"step 7/8: exporting production model -> {winner_name}")
    prod = promote_to_production(winner, db, Path(cfg.models_dir), pool, rationale, gen_payload,
                                 inv.to_dict(), summaries, cfg, ablation_rows, gen_task)

    _log(cfg, "step 8/8: writing reports, plots and metrics")
    test_scores = np.load(winner.artifacts["test_scores"]) if "test_scores" in winner.artifacts else None
    figures: List[Path] = []
    if test_scores is not None and db.binary:
        y_te = db.labels[db.split.test_idx]
        ref_scores = {}
        for c in candidates[:6]:
            p = Path(c.artifacts.get("test_scores", ""))
            if p.exists():
                ref_scores[c.name] = np.load(p)
        if ref_scores:
            figures.append(plot_roc_grid(ref_scores, y_te, Path(cfg.reports_dir) / "figures" / "roc_test.png"))
            best_score = ref_scores.get(winner.name)
            if best_score is not None:
                m = binary_metrics(y_te, best_score, threshold=float(prod["threshold"]))
                figures.append(
                    plot_confusion(
                        np.array(m["confusion_matrix"]), db.classes,
                        Path(cfg.reports_dir) / "figures" / "confusion_selected.png",
                        title=f"{winner.name} (test, thr={prod['threshold']:.2f})",
                    )
                )
    rows_with_rows = {c.name: c for c in candidates}
    write_comparison_report(pool, rationale, db, Path(cfg.reports_dir) / "model_comparison.md", gen_payload, ablation_rows, rows_with_rows)
    write_training_report(pool, summaries, ablation_rows, Path(cfg.reports_dir) / "training_report.md", time.time() - t_start, cfg)
    if cfg.export_samples_n:
        export_samples(db, Path(cfg.samples_dir), n_per_class=cfg.export_samples_n)

    total_s = time.time() - t_start
    _log(cfg, f"done in {total_s/60:.1f} min -> selected {winner_name}")
    return RunResult(
        selected={**rationale, "candidate": winner.name, "artifacts": prod["paths"]},
        candidates=pool,
        metrics=json.loads((Path(cfg.models_dir) / "metrics.json").read_text()),
        paths=prod["paths"],
        generator=gen_payload,
        inventory=inv.to_dict(),
        generator_task=gen_task,
    )


def promote_to_production(
    winner: Candidate,
    db: DataBundle,
    models_dir: Path,
    pool: Sequence[Dict[str, object]],
    rationale: Dict[str, object],
    generator: Dict[str, object],
    inventory: Dict[str, object],
    summaries: Dict[str, Dict[str, object]],
    cfg: PipelineConfig,
    ablations: Optional[Sequence[Dict[str, object]]] = (),
    generator_task: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Copy ONLY the selected model into ``models/`` (lightweight, deployment-ready)."""
    models_dir = Path(models_dir)
    out = models_dir / "ai_detector"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(winner.artifacts.get("checkpoint") or winner.artifacts.get("model"))
    suffix = src.suffix
    shutil.copy2(src, out / f"model{suffix}")
    if winner.family == "deep":
        (out / "model_config.json").write_text(json.dumps(summaries.get(winner.name, {}), indent=2, default=str))

    from .dataset import POSITIVE_LABELS

    def meaning_of(name: str) -> str:
        n = str(name).lower()
        if n in POSITIVE_LABELS or any(k in n for k in ("fake", "gen", "synth", "ai", "deepfake", "spoof")):
            return "ai_generated or synthetic"
        if any(k in n for k in ("real", "genuine", "authentic", "bona", "original")):
            return "authentic / camera-captured"
        return f"class '{name}' (meaning not derivable from the folder name)"

    classes_json = {
        "task": "ai_detector",
        "classes": db.classes,                       # indexed by class id, i.e. classes[labels[i]]
        "label_map": db.mapping,
        "positive_class": db.positive_class if db.binary else None,
        "positive_index": int(db.positive_index) if db.binary else None,
        "n_classes": db.n_classes,
        "binary": db.binary,
        "labels_meaning": {str(i): meaning_of(c) for i, c in enumerate(db.classes)},
    }
    (out / "classes.json").write_text(json.dumps(classes_json, indent=2))

    # preprocessing spec shared with the app
    pre = {
        "cache_size": db.cache_size,
        "resize_to": int(summaries.get(winner.name, {}).get("input_size", db.cache_size)),
        "mean": list(db.mean),
        "std": list(db.std),
        "photometric_for_features": cfg.photometric,
        "feature_size": cfg.feature_size if winner.family == "classical" else None,
        "augmented_fields": "train only - inference uses resize+normalise",
    }
    (models_dir / "preprocessing.json").write_text(json.dumps(pre, indent=2))

    # reference distributions for the explanation engine (same training split only)
    try:
        from .explain import build_reference

        build_reference(db, models_dir / "reference_stats.json", progress=False)
    except Exception as exc:  # noqa: BLE001 - explanation must never break training
        (models_dir / "reference_stats.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
        )

    test_scores_path = Path(winner.artifacts["test_scores"])
    y_te = db.labels[db.split.test_idx]
    p_te = np.load(test_scores_path)
    thr = float(winner.metrics.get("threshold", 0.5) or 0.5)
    if db.binary:
        p_te = p_te if p_te.ndim == 1 else p_te[:, db.positive_index]
        # keep exactly the threshold that produced the reported metrics (validated on val only)
        thr = float(winner.metrics.get("threshold", 0.5))
        m = binary_metrics(y_te, p_te, threshold=thr)
    else:
        m = {"accuracy": float(np.mean(p_te.argmax(1) == y_te))}

    calibration = expected_calibration_error(y_te, p_te) if db.binary else None
    rel = reliability_table(y_te, p_te) if db.binary else None

    pool_names = [r["name"] for r in pool]
    metrics_payload = {
        "schema_version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {"root": str(cfg.root), "inspection": inventory},
        "split": {
            "strategy": "stratified by class, atomic by capture group (70/15/15)",
            "sizes": db.split.sizes(),
            "seed": cfg.seed,
            "train_pos_rate": float(db.labels[db.split.train_idx].mean()),
            "test_pos_rate": float(y_te.mean()),
        },
        "selection": {"weights": {"f1": 0.40, "roc_auc": 0.25, "accuracy": 0.15, "speed": 0.10, "efficiency": 0.10}, **rationale},
        "comparison": list(pool),
        "tasks": {
            "ai_detector": {
                "task": "ai_detector",
                "selected_model": winner.name,
                "family": winner.family,
                "classes": db.classes,
                "threshold": thr,
                "threshold_policy": "max-F1 on the validation split (midpoint between distinct validation scores)",
                "eval_protocol": "train on train, threshold + early stopping on validation, test evaluated once",
                "test_evaluations": 1,
                "metrics": m,
                "calibration_ece": calibration,
                "reliability": rel,
                "train_images": db.split.sizes()["train"],
                "val_images": db.split.sizes()["val"],
                "test_images": db.split.sizes()["test"],
                "inference_ms_per_image": round(winner.inference_ms, 3),
                "model_size_mb": winner.size_mb,
            },
            **({"generator_classifier": generator_task} if generator_task else {}),
        },
        "generator_classifier": generator,
        "face_model": face_model_status(db, winner),
        "ablations": list(ablations or []),
        "transfer_learning": transfer_learning_record(summaries, cfg, pool_names),
        "reproducibility": {
            "config": json.loads(cfg.to_json()),
            "seed": cfg.seed,
            "cache_size": cfg.cache_size,
            "feature_size": cfg.feature_size,
            "photometric": cfg.photometric,
            "device": cfg.device,
            "threads": cfg.threads,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=str))

    card = {
        "model_name": f"ams-ai-detector-{winner.name}",
        "model_version": VERSION,
        "task": "binary AI/synthetic image detection (real vs generated)",
        "selected_from": [r["name"] for r in pool],
        "selection_reason": rationale["reason"],
        "architecture": winner.name,
        "framework": "pytorch" if winner.family == "deep" else "scikit-learn",
        "params": winner.params,
        "image_size": pre["resize_to"],
        "classes": db.classes,
        "n_classes": db.n_classes,
        "training_images": db.split.sizes()["train"],
        "validation_images": db.split.sizes()["val"],
        "test_images": db.split.sizes()["test"],
        "metrics": m,
        "threshold": thr,
        "model_size_mb": winner.size_mb,
        "inference_ms_per_image": round(winner.inference_ms, 3),
        "pretrained_backbone": summaries.get(winner.name, {}).get("pretrained_applied"),
        "pretrained_requested": summaries.get(winner.name, {}).get("pretrained_requested"),
        "pretrained_note": summaries.get(winner.name, {}).get("pretrained_note"),
        "transfer_learning": transfer_learning_record(summaries, cfg),
        "threshold_policy": "max-F1 on the validation split; midpoint between distinct validation scores",
        "eval_protocol": "train on train, threshold + early stopping on validation, test evaluated once",
        "test_evaluations": 1,
        "descriptor": summaries.get(winner.name, {}).get("descriptor"),
        "feature_dim": summaries.get(winner.name, {}).get("feature_dim"),
        "load_with": "python -m ams predict <image>   |   from ams.predictor import Detector; Detector(PredictorConfig()).predict_path(...)",
        "training_date": time.strftime("%Y-%m-%d"),
        "dataset": {
            "root": str(cfg.root),
            "classes_found": inventory.get("classes_found"),
            "n_valid": inventory.get("n_valid"),
            "n_corrupt": inventory.get("n_corrupt"),
            "class_balance": inventory.get("class_balance"),
            "source_groups": inventory.get("n_groups"),
            "near_duplicates": inventory.get("n_duplicate_near"),
            "perceptual_duplicates_across_sources": inventory.get("n_duplicate_perceptual_cross_source"),
            "formats": inventory.get("formats"),
            "sizes": inventory.get("sizes"),
        },
        "known_limitations": LIMITATIONS,
        "honesty": HONESTY,
        "generator_classifier": generator,
        "face_model": face_model_status(db, winner),
    }
    (models_dir / "model_info.json").write_text(json.dumps(card, indent=2, default=str))
    write_model_card(card, models_dir / "MODEL_CARD.md")

    return {"paths": {"model": str(out / f"model{suffix}"), "classes": str(out / "classes.json"),
                     "metrics": str(models_dir / "metrics.json"), "model_info": str(models_dir / "model_info.json"),
                     "preprocessing": str(models_dir / "preprocessing.json")},
            "threshold": thr, "card": card}


LIMITATIONS = [
    "Trained on 32x32 upscaled crops of one corpus: it learns the artifacts of *this* data (compression, texture, spectrum), not a universal notion of 'AI-generated'.",
    "Performance degrades on generators, resolutions, cameras and post-processing pipelines that are not represented in the training set.",
    "Heavy re-compression, screenshots, resizing, watermarks and upscaling change the measured evidence and can flip predictions.",
    "Class balance here is 50/50; in production the real rate is far higher, so the false-positive count matters more than accuracy suggests.",
    "The crops are faces at very low resolution: identity is not recoverable and no identity model was trained.",
    "In this corpus the two classes differ measurably in sharpness and tone, which is a shortcut a "
    "classifier can exploit instead of learning synthesis artefacts. Photometric statistics are "
    "per-image z-scored in both the descriptor and the model input to limit it, and the raw-pixel "
    "baselines plus the colour/photometric ablation in reports/model_comparison.md quantify how much "
    "of the performance that shortcut could explain; a residual correlation cannot be excluded.",
]
HONESTY = [
    "Prediction is probabilistic and is not proof of image origin.",
    "Generator identification is limited to patterns represented in the training dataset.",
    "Performance may decrease on unseen generators.",
    "Image editing, compression, screenshots and resizing can affect predictions.",
    "Face analysis does not identify people.",
    "No detector is 100% accurate and this one makes no such claim: an individual prediction can be wrong, and 'authentic' does not prove human authorship.",
]


def run_generator_task(
    cfg: PipelineConfig, db: DataBundle, gen_dim: Dict[str, object], run_dir: Path
) -> Optional[Dict[str, object]]:
    """Train the second supervised task (which generator made a synthetic image).

    Returns ``None`` when the dataset carries no per-generator labels: the caller then
    reports the task as untrainable. Classes are never invented to make it possible.
    """
    from .generator import build_generator_bundle, export_generator_model, fit_unknown_gate
    from .selection import GENERATOR_WEIGHTS

    gb = build_generator_bundle(db, gen_dim, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    if gb is None:
        return None
    gdb = gb.bundle
    _log(cfg, f"        {len(gb.generator_classes)} generator classes {gb.counts} on {len(gdb.labels)} images")

    work = Path(run_dir) / "generator"
    work.mkdir(parents=True, exist_ok=True)
    Xg, names = forensic_matrix(
        gdb, target_size=cfg.feature_size, work_dir=work,
        opts=FeatureOptions(photometric=cfg.photometric), progress=False,
    )

    cands: List[Candidate] = []
    gsum: Dict[str, Dict[str, object]] = {}
    if cfg.classical:
        for preset in classical_presets():
            if str(preset.get("feature")) != "forensic":
                continue
            p2 = dict(preset)
            p2["id"] = f"gen_{preset['id']}"
            p2["note"] = f"generator task | {preset.get('note', '')}".strip(" |")
            try:
                cand, summ = train_classical(gdb, p2, work / str(p2["id"]), X_forensic=Xg, feature_names=names)
            except Exception as exc:  # noqa: BLE001
                _log(cfg, f"        !! generator candidate {p2['id']} failed: {type(exc).__name__}: {exc}")
                continue
            cand.task = "generator"
            cands.append(cand)
            gsum[cand.name] = summ
    if cfg.deep:
        for preset in [p for p in CANDIDATE_PRESETS if (cfg.deep_ids is None or p["id"] in list(cfg.deep_ids))][:1]:
            p2 = dict(preset)
            p2["id"] = f"gen_{preset['id']}"
            try:
                cand, summ = train_deep(
                    gdb, p2, work / str(p2["id"]), epochs=min(int(cfg.epochs or 12), 12),
                    batch_size=cfg.batch_size, lr=cfg.lr, pretrained=cfg.pretrained,
                    device=cfg.device, torch_threads=cfg.threads, patience=cfg.patience, progress=False,
                )
            except Exception as exc:  # noqa: BLE001
                _log(cfg, f"        !! generator deep candidate failed: {type(exc).__name__}: {exc}")
                continue
            cand.task = "generator"
            cands.append(cand)
            gsum[cand.name] = summ
    if not cands:
        return {"task": "generator_fingerprinting", "trained": False,
                "error": "no generator candidate trained successfully on the labelled subset"}

    weights = dict(GENERATOR_WEIGHTS)
    weights_note = ""
    if len(gb.generator_classes) <= 3:
        weights["f1"] = round(weights.pop("top3_accuracy") + weights["f1"], 4)
        weights_note = (
            f"top-3 accuracy was dropped from the score because with "
            f"{len(gb.generator_classes)} classes it is a constant 1.0; its weight went to macro F1"
        )
    rows = score_pool(cands, weights=weights, task="generator")
    rationale = selection_rationale(rows, weights=weights)
    win = next(c for c in cands if c.name == rationale["selected"])

    val_scores = np.load(win.artifacts["val_scores"])
    y_val = gdb.labels[gdb.split.val_idx]
    if gdb.binary and val_scores.ndim == 1:      # 2 generator classes: rebuild the 2-vector
        val_scores = np.stack([1.0 - val_scores, val_scores], axis=1)
    gate = fit_unknown_gate(val_scores, y_val)

    model_path = win.artifacts["model"] if win.family == "classical" else win.artifacts["checkpoint"]
    out_dir = Path(cfg.models_dir) / "generator_classifier"
    meta = {
        "id": win.name,
        "feature_size": int(Xg.shape[1]),
        "input_size": int(gsum[win.name].get("input_size") or gdb.cache_size),
        "photometric": cfg.photometric,
    }
    export_generator_model(Path(model_path), out_dir, gb, gate, meta)
    _log(cfg, f"        generator model exported -> {out_dir} ({win.name}, "
              f"macro F1 {float(win.metrics.get('macro_f1', win.metrics.get('f1', 0.0))):.4f})")
    return {
        "task": "generator_fingerprinting",
        "trained": True,
        "selected_model": win.name,
        "family": win.family,
        "classes": gb.generator_classes,
        "per_class_counts": gb.counts,
        "label_source": gb.label_source,
        "images_without_generator_label": gb.n_unlabelled,
        "split_sizes": gdb.split.sizes(),
        "selection": {"weights": weights, "weights_declared": dict(GENERATOR_WEIGHTS),
                      "weights_note": weights_note, **rationale},
        "comparison": list(rows),
        "metrics": dict(win.metrics),
        "unknown_generator_gate": gate,
        "model_dir": str(out_dir),
        "model_kind": "torch" if win.family == "deep" else "sklearn",
        "inference_ms_per_image": round(win.inference_ms, 3),
        "model_size_mb": win.size_mb,
        "eval_protocol": "train on train, gate + selection on validation, test evaluated once",
        "test_evaluations": 1,
        "notes": list(gb.notes),
        "limitation": (
            "this model can only name the generators listed in `classes`; images from any other "
            "generator are returned as UNKNOWN by the acceptance rule, never relabelled to the "
            "closest known generator"
        ),
    }


def transfer_learning_record(
    summaries: Dict[str, Dict[str, object]],
    cfg: PipelineConfig,
    pool_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """What §7 asked for versus what this environment could actually deliver."""
    deep = {k: v for k, v in summaries.items() if isinstance(v, dict) and "pretrained_requested" in v}
    pool = set(pool_names or [])
    in_pool = {k: v for k, v in deep.items() if not pool or k in pool}
    ablation_arms = sorted(k for k in deep if pool and k not in pool)
    applied = sorted(k for k, v in deep.items() if v.get("pretrained_applied"))
    scratch = sorted(k for k in deep if k not in applied)
    frozen = sorted(k for k, v in deep.items() if (v.get("spec") or {}).get("freeze_backbone"))
    truly_frozen = sorted(k for k, v in deep.items() if v.get("frozen_backbone"))
    if not deep:
        note = "no deep candidate was part of this run"
    elif not in_pool:
        note = ("no deep candidate was eligible for selection in this run; the trained deep arms listed "
                "under ablation_arms are reported only")
    elif applied:
        note = (
            "ImageNet weights loaded for " + ", ".join(applied) + "; the frozen-backbone stage ran first "
            "where requested and the head was trained on top"
            if truly_frozen else
            "ImageNet weights loaded; every arm trained its backbone as well (no frozen-backbone arm in this run)"
        )
    elif cfg.pretrained:
        notes = " ; ".join(sorted({str(v.get("pretrained_note")) for v in deep.values() if v.get("pretrained_note")}))
        if "unreachable" in notes or "URLError" in notes or "Download" in notes.lower():
            why = (
                "the ImageNet weight host was unreachable from this environment "
                f"({notes or 'URLError'})"
            )
        else:
            why = f"these architectures have no loadable pretrained weights here ({notes or 'n/a'})"
        note = (
            f"transfer learning was requested but {why}; freezing a randomly initialised backbone would only "
            "train a head on random features, so no arm was frozen and every deep candidate trained from scratch "
            "with all layers trainable. This is recorded per candidate in pretrained_note rather than hidden."
        )
    else:
        note = "transfer learning was disabled by configuration (--no-pretrained)"
    return {
        "requested": bool(cfg.pretrained),
        "deep_candidates": len(in_pool),
        "deep_candidates_including_ablations": len(deep),
        "ablation_arms": ablation_arms,
        "weights_applied_for": applied,
        "trained_from_scratch": sorted(k for k in scratch if k in in_pool),
        "trained_from_scratch_including_ablations": scratch,
        "frozen_backbone_requested_for": frozen,
        "frozen_backbone_applied_for": truly_frozen,
        "trainable_parameter_ratios": {
            k: [v.get("trainable_parameters"), v.get("total_parameters")]
            for k, v in sorted(deep.items())
            if v.get("total_parameters")
        },
        "note": note,
    }


def face_model_status(db: DataBundle, winner: Candidate) -> Dict[str, object]:
    """Third-task bookkeeping: what the dataset does and does not support."""
    return {
        "trained": False,
        "requested": True,
        "reason": (
            "the corpus holds exactly two classes (real / synthetic face crops) and no 'manipulated' "
            "label dimension, so a separate REAL-FACE vs DEEPFAKE-FACE classifier would be a copy of the "
            "AI detector with a new name. Rather than duplicate it, the detector is trained on the face crops "
            "themselves and face-level checks are provided by the analytical CV module "
            "(ams.forensics: face detection, landmark geometry, visual consistency), which is explicitly "
            "NOT a trained deepfake classifier."
        ),
        "detector_trained_on_face_crops": True,
        "identity_model_available": False,
        "analytical_cv_module": "ams.forensics",
    }
