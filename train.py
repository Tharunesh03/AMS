#!/usr/bin/env python3
"""Training entry point: dataset -> candidates -> comparison -> production model.

    python train.py                       # full run from configs/default.yaml
    python train.py --config configs/smoke.yaml
    python train.py --only classical              # sklearn candidates only
    python train.py --only deep --no-ablations     # deep candidates, skip the ablation arms
    python train.py --set max_images_per_class=800 --set epochs=2      # quick check
    python train.py --inspect-only                # dataset report, no training

The trained selection lands in ``models/`` (one artifact + metrics + model card);
per-candidate artifacts go to ``artifacts/<run_name>/`` (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ams.pipeline import PipelineConfig, run_training  # noqa: E402


def _coerce(text: str):
    v = yaml.safe_load(text)
    return v


PATH_KEYS = {"root", "archive", "artifacts_dir", "models_dir", "reports_dir", "samples_dir"}


def build_config(args: argparse.Namespace) -> PipelineConfig:
    raw: Dict[str, object] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    for key in list(raw):
        if key in PATH_KEYS and raw[key]:
            raw[key] = str(ROOT / str(raw[key])) if not Path(str(raw[key])).is_absolute() else str(raw[key])
    cfg = PipelineConfig(**raw)
    if args.set:
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"--set expects key=value, got {item!r}")
            k, v = item.split("=", 1)
            k = k.strip()
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown config key {k!r}")
            val = _coerce(v)
            if k in PATH_KEYS and val:
                p = Path(str(val))
                val = p if p.is_absolute() else ROOT / p
            setattr(cfg, k, val)
    if args.only == "classical":
        cfg.deep = False
    elif args.only == "deep":
        cfg.classical = False
    elif args.only == "ablations":
        cfg.generator_task = False
        # ablations are reported, never selected - a minimal deep pool is trained as well so the
        # run still ends with a real selection instead of an empty candidate list
        cfg.classical = False
        cfg.deep = True
        cfg.deep_ids = cfg.deep_ids or ["lightcnn_32"]
        if not list(cfg.ablations):
            raise SystemExit("--only ablations needs ablation presets in the config "
                             "(configs/default.yaml defines two)")
    if args.smoke:
        cfg.max_images_per_class = 600
        cfg.cache_size = 48
        cfg.feature_size = 48
        cfg.export_samples_n = 1
        cfg.run_name = "smoke"
        cfg.ablations = []
        cfg.epochs = 2
        cfg.deep_ids = ["lightcnn_32"]
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config (see configs/)")
    ap.add_argument("--set", action="append", help="override, e.g. --set epochs=4 --set deep_ids=[resnet18_96]")
    ap.add_argument("--only", choices=["classical", "deep", "ablations", "all"], default="all")
    ap.add_argument("--pretrained", action="store_true", help="request ImageNet-pretrained backbones")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="force training from scratch")
    ap.set_defaults(pretrained=None)
    ap.add_argument("--rebuild-cache", action="store_true", help="re-decode every image into artifacts/cache")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the whole pipeline")
    ap.add_argument("--no-ablations", dest="skip_ablations", action="store_true",
                    help="skip the ablation arms (they are reported, never selected)")
    ap.add_argument("--no-generator-task", dest="skip_generator", action="store_true",
                    help="skip the multi-class generator fingerprinting step entirely")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--inspect-only", action="store_true", help="write reports/eda.md and exit")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.skip_ablations:
        cfg.ablations = []
    if args.skip_generator:
        cfg.generator_task = False
    if args.rebuild_cache:
        cfg.rebuild_cache = True
    cfg.verbose = not args.quiet

    if args.inspect_only:
        from ams.dataset import ensure_extracted
        from ams.reporting import write_dataset_report
        from ams.trainer import build_data

        ensure_extracted(Path(cfg.root), Path(cfg.archive) if cfg.archive else None)
        db = build_data(Path(cfg.root), Path(cfg.artifacts_dir) / "cache", cache_size=cfg.cache_size,
                        max_images_per_class=cfg.max_images_per_class, verbose=cfg.verbose)
        Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
        out = write_dataset_report(db, Path(cfg.reports_dir) / "eda.md")
        print(f"wrote {out}")
        return 0

    print(f"[ams] run={cfg.run_name} device={cfg.device} threads={cfg.threads} "
          f"classical={cfg.classical} deep={cfg.deep} pretrained={cfg.pretrained}", flush=True)
    result = run_training(cfg)
    sel = result.selected
    print("\n================ SELECTED ================")
    print(f"task=ai_detector  model={sel.get('selected')}")
    print(f"reason: {sel.get('reason')}")
    print(f"artifacts: {', '.join(Path(v).name for v in sel.get('artifacts', {}).values())}")
    print(f"reports: {Path(cfg.reports_dir)}/model_comparison.md, "
          f"{Path(cfg.reports_dir)}/training_report.md, {Path(cfg.reports_dir)}/eda.md")
    gen = result.generator or {}
    gtask = gen.get("task")
    if gtask and gtask.get("trained"):
        gm = gtask.get("metrics") or {}
        print(f"task=generator    model={gtask.get('selected_model')} classes={gtask.get('classes')} "
              f"macroF1={gm.get('macro_f1')} top3={gm.get('top3_accuracy', 'n/a')} "
              f"gate(min_prob={gtask['unknown_generator_gate']['min_prob']}, "
              f"margin={gtask['unknown_generator_gate']['min_margin']})")
    else:
        print(f"task=generator    NOT TRAINED - {str(gen.get('reason'))[:110]}")
    fm = result.metrics.get("face_model") or {}
    print(f"face analysis     {fm.get('analytical_cv_module', 'ams.forensics')} "
          f"(trained classifier: {fm.get('trained')}, identity model: {fm.get('identity_model_available')})")
    print(f"models : {Path(cfg.models_dir)}/metrics.json, {Path(cfg.models_dir)}/model_info.json, "
          f"{Path(cfg.models_dir)}/MODEL_CARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
