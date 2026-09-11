"""Command line: ``python -m ams <command>`` (see ``python -m ams --help``).

    inspect     dataset inspection report only
    train       full training pipeline (same as ``python train.py``)
    reference   rebuild models/reference_stats.json from the training split
    evaluate    re-measure the *production* artifact on the held-out test split
    predict     predict one or more image files
    demo        predict the bundled samples/ images and print a table
    serve       run the FastAPI app (loads the saved model, never retrains)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text()) or {}


# --------------------------------------------------------------------------- inspect
def cmd_inspect(args) -> int:
    from ams.dataset import ensure_extracted
    from ams.reporting import write_dataset_report
    from ams.trainer import build_data

    cfg = _load_yaml_config(args.config)
    root = Path(args.root or cfg.get("root", "data"))
    archive = Path(cfg.get("archive")) if cfg.get("archive") else None
    if not root.is_absolute():
        root = ROOT / root
    if archive and not archive.is_absolute():
        archive = ROOT / archive
    ensure_extracted(root, archive)
    db = build_data(
        root, ROOT / "artifacts" / "cache", archive=archive,
        cache_size=int(cfg.get("cache_size", 64)), max_images_per_class=args.limit, verbose=not args.quiet,
    )
    out = write_dataset_report(db, ROOT / "reports" / "eda.md")
    print(json.dumps({k: v for k, v in db.inventory.to_dict().items() if k != "rejected"}, indent=2)[:2000])
    print(f"\nwrote {out}")
    return 0


# --------------------------------------------------------------------------- train
def cmd_train(args) -> int:
    from train import main as train_main

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    for item in args.set or []:
        argv += ["--set", item]
    if args.smoke:
        argv.append("--smoke")
    if args.inspect_only:
        argv.append("--inspect-only")
    return train_main(argv)


# ----------------------------------------------------------------------- reference
def cmd_reference(args) -> int:
    from ams.explain import build_reference
    from ams.trainer import build_data

    cfg = _load_yaml_config(args.config)
    out = Path(args.out or ROOT / "models" / "reference_stats.json")
    db = build_data(
        ROOT / str(cfg.get("root", "data")), ROOT / "artifacts" / "cache",
        archive=Path(str(cfg.get("archive"))) if cfg.get("archive") else None,
        cache_size=int(cfg.get("cache_size", 64)), verbose=not args.quiet,
    )
    payload = build_reference(db, out, progress=not args.quiet)
    print(f"wrote {out} with {len(payload['per_class'])} classes x {len(payload['signals'])} signals")
    return 0


# ------------------------------------------------------------------------ evaluate
def cmd_evaluate(args) -> int:
    import numpy as np

    from ams.metrics import binary_metrics, reliability_table
    from ams.predictor import Detector, PredictorConfig
    from ams.reporting import plot_confusion, plot_reliability
    from ams.trainer import build_data

    cfg = _load_yaml_config(args.config)
    db = build_data(
        ROOT / str(cfg.get("root", "data")), ROOT / "artifacts" / "cache",
        archive=Path(str(cfg.get("archive"))) if cfg.get("archive") else None,
        cache_size=int(cfg.get("cache_size", 64)), verbose=not args.quiet,
    )
    det = Detector(PredictorConfig(models_dir=ROOT / "models", enable_face=False, enable_gradcam=False))
    idx = db.split.test_idx
    limit = int(args.limit or 0)
    subsampled = False
    if 0 < limit < len(idx):
        # NEVER a head slice: the split arrays are grouped by class, so slicing would
        # evaluate a single class and produce meaningless metrics.
        subsampled = True
        per_class = max(1, limit // db.n_classes)
        picked = []
        for c in range(db.n_classes):
            cls_idx = idx[db.labels[idx] == c]
            step = max(1, len(cls_idx) // per_class)
            picked.append(cls_idx[::step][:per_class])
        idx = np.sort(np.concatenate(picked))
    if subsampled:
        print(f"[warn] evaluating a class-balanced subsample of {len(idx)} of {len(db.split.test_idx)} "
              f"test images; metrics will differ from the full-split numbers in models/metrics.json",
              flush=True)
    t0 = time.time()
    scores = np.array([det.score(db.images[int(i)]) for i in idx])
    dt = time.time() - t0
    m = binary_metrics(db.labels[idx], scores, threshold=det.threshold)
    per_image_ms = dt / max(len(idx), 1) * 1000
    reported = det.model_info.get("metrics") or {}
    # The training-time metrics come from the stored float32 score array, this recomputation
    # re-scores the images through the shipped artifact, so tiny reordering differences are
    # expected - 1e-4 is the tolerance, and it is written into the report rather than assumed.
    tol = 1e-4 if not subsampled else 5e-3
    agree, deltas = {}, {}
    for k in ("accuracy", "f1", "roc_auc", "recall", "precision"):
        a, b = reported.get(k), m.get(k)
        try:
            d = abs(float(a) - float(b))
        except (TypeError, ValueError):
            d = float("nan")
        deltas[k] = float(d)
        agree[k] = bool(d == d and d <= tol)
    rep = {
        "production_model": det.model_info.get("model_name"),
        "full_test_split": not subsampled,
        "agreement_tolerance": tol,
        "architecture": det.arch,
        "threshold": det.threshold,
        "n_evaluated": int(len(idx)),
        "seconds": round(dt, 1),
        "ms_per_image": round(per_image_ms, 2),
        "metrics": m,
        "agrees_with_reported_metrics": {
            k: {"reported": reported.get(k), "recomputed": m.get(k), "abs_delta": deltas[k],
                "agree": agree[k]}
            for k in ("accuracy", "f1", "roc_auc", "recall", "precision")
        },
    }
    out_json = ROOT / "reports" / "production_evaluation.json"
    out_json.write_text(json.dumps(rep, indent=2, default=str))

    lines = ["# Production model re-evaluation (test split)\n",
             f"- model: `{rep['production_model']}` ({rep['architecture']}), threshold {rep['threshold']:.3f}",
             f"- images: {rep['n_evaluated']} | {rep['seconds']}s | {rep['ms_per_image']} ms/image on this machine",
             f"- accuracy {m['accuracy']:.4f} | F1 {m['f1']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
             f"recall {m['recall']:.4f} | precision {m['precision']:.4f} | EER {m['eer']:.4f}",
             f"- FPR (false lock rate on authentic) {m['false_positive_rate']:.4f} | "
             f"FNR (missed synthetic) {m['false_negative_rate']:.4f}\n",
             ("- recomputed metrics reproduce the reported `models/metrics.json` within the "
              f"stated tolerance {tol:.0e} (max abs delta "
              f"{max([v for v in deltas.values() if v == v], default=0.0):.3g}) - the deployed "
              "artifact is the same model the dashboard serves"
              if all(agree.values()) else
              "- recomputed metrics do NOT match `models/metrics.json` "
              + ("(expected: a class-balanced subsample was evaluated)" if subsampled
                 else "(unexpected - the shipped artifact may not be the model that was measured)")
              + f"; deltas {json.dumps({k: round(v, 6) for k, v in deltas.items()})}\n"),
             "",
             "This recomputation loads only the shipped artifact; it does not retrain anything.\n"]
    plot_confusion(np.array(m["confusion_matrix"]), db.classes, ROOT / "reports" / "figures" / "confusion_production.png",
                   title=f"{rep['architecture']} (test)")
    plot_reliability(reliability_table(db.labels[idx], scores),
                     ROOT / "reports" / "figures" / "reliability_production.png", title=rep["architecture"] or "")
    (ROOT / "reports" / "production_evaluation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out_json}")
    return 0


# ------------------------------------------------------------------------- predict
def cmd_predict(args) -> int:
    from ams.predictor import Detector, PredictorConfig, ModelNotTrained

    try:
        det = Detector(PredictorConfig(models_dir=Path(args.models_dir), enable_face=not args.no_face, enable_gradcam=not args.no_cam))
    except ModelNotTrained as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for f in args.files:
        res = det.predict_path(Path(f))
        if args.plain:
            gi = res.get("generator_identification") or {}
            gen_txt = (f"  generator={gi.get('generator')}(p={gi.get('confidence'):.3f})"
                       if gi.get("available") and gi.get("confidence") is not None
                       else ("  generator=n/a" if "generator_identification" in res else ""))
            print(f"{f}: {res['verdict']}  P(synthetic)={res['probability_synthetic']:.3f}  "
                  f"(thr {res['threshold']:.3f}, {res['timing_ms']} ms){gen_txt}")
            continue
        slim = dict(res)
        if not args.include_images:
            sal = dict(slim.get("saliency") or {})
            sal.pop("overlay_png_b64", None)
            sal.pop("input_png_b64", None)
            slim["saliency"] = sal
        print(json.dumps(slim, indent=2, default=str))
    return 0


def cmd_demo(args) -> int:
    from ams.predictor import Detector, PredictorConfig

    det = Detector(PredictorConfig(models_dir=Path(args.models_dir)))
    man = Path(args.samples_dir) / "manifest.json"
    items = json.loads(man.read_text()) if man.exists() else [
        {"file": p.name, "class": p.stem.rsplit("_", 1)[0]} for p in sorted(Path(args.samples_dir).glob("*.png"))
    ]
    print(f"{'file':<18}{'true class':<12}{'prediction':<28}{'P(syn)':>8}")
    print("-" * 66)
    ok = 0
    for it in items:
        p = Path(args.samples_dir) / it["file"]
        if not p.exists():
            continue
        r = det.predict_bytes(p.read_bytes())
        hit = r["predicted_class"] == it["class"]
        ok += int(hit)
        print(f"{it['file']:<18}{it['class']:<12}{r['verdict']:<28}{r['probability_synthetic']:>8.3f}  {'' if hit else 'MISS'}")
    print(f"\n{ok}/{len(items)} samples matched their dataset class (illustrative only - the real numbers are in models/metrics.json)")
    return 0


# --------------------------------------------------------------------------- serve
def cmd_serve(args) -> int:
    import uvicorn

    print(f"[ams] serving http://{args.host}:{args.port} (model dir: {args.models_dir}) - no training happens here")
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ams", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="dataset inspection report")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--root")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("train", help="run the training pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("reference", help="rebuild models/reference_stats.json from the training split")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_reference)

    p = sub.add_parser("evaluate", help="re-measure the production artifact on the test split")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of test images (class-balanced); default: the whole test split")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict", help="predict image files")
    p.add_argument("files", nargs="+")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--plain", action="store_true", help="one line per image")
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-cam", action="store_true")
    p.add_argument("--include-images", action="store_true")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("demo", help="predict the bundled samples")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--samples-dir", default=str(ROOT / "samples"))
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("serve", help="run the FastAPI app")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.set_defaults(fn=cmd_serve)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
