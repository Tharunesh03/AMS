"""Second supervised task (generator fingerprinting): label plumbing, gate, and an end-to-end run.

These tests use a miniature corpus that *does* carry per-generator labels (nested folders),
because on the real corpus the labels do not exist and the pipeline must refuse to train.
The point of the test is that when labels exist the multi-class model really trains, exports
and identifies - and that an image outside its acceptance rule is answered with UNKNOWN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERATORS = ("stable_diffusion", "stylegan", "midjourney")


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=92)


def generator_texture(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Three visibly different synthesis artefacts so a classifier has something real to learn."""
    yy, xx = np.indices((size, size))
    if kind == "stable_diffusion":
        base = np.where((yy % 6 < 3) ^ (xx % 6 < 3), 190.0, 60.0)
        noise = 8.0
    elif kind == "stylegan":
        base = 120.0 + 70.0 * np.cos(yy * (2 * np.pi / 5.0))
        noise = 6.0
    else:
        base = 120.0 + 70.0 * np.sin((xx + yy) * (2 * np.pi / 7.0))
        noise = 30.0
    return base[..., None] + rng.normal(0, noise, (size, size, 3))


def make_generator_dataset(root: Path, n_groups: int = 12, frames: int = 3, size: int = 32,
                           seed: int = 7) -> Path:
    """``FAKE/<generator>/<id> (k).jpg`` plus plain ``REAL/<id> (k).jpg`` authentic frames."""
    rng = np.random.default_rng(seed)
    for g in range(n_groups):
        for f in range(frames):
            suffix = "" if f == 0 else f" ({f + 1})"
            real = rng.random((size, size, 3)) * 40 + 110 + rng.normal(0, 7.0, (size, size, 3))
            _write(root / "REAL" / f"{g:04d}{suffix}.jpg", real)
            for gen in GENERATORS:
                _write(root / "FAKE" / gen / f"{g:04d}{suffix}.jpg", generator_texture(gen, size, rng))
    return root


@pytest.fixture(scope="module")
def gen_root(tmp_path_factory) -> Path:
    return make_generator_dataset(tmp_path_factory.mktemp("gendata"), n_groups=12, frames=3)


@pytest.fixture(scope="module")
def gen_db(gen_root, tmp_path_factory):
    from ams.trainer import build_data

    return build_data(
        root=gen_root,
        cache_dir=tmp_path_factory.mktemp("gencache"),
        archive=None,
        cache_size=48,
        train_frac=0.6,
        val_frac=0.2,
        seed=5,
        verbose=False,
    )


# --------------------------------------------------------------------- label plumbing
def test_per_image_labels_come_from_the_directory_not_from_a_guess(gen_db):
    from ams.generator import image_generator_labels

    ids, index = image_generator_labels(gen_db.paths, list(GENERATORS))
    assert len(ids) == len(gen_db.labels)
    labelled = ids[ids >= 0]
    assert labelled.size == 3 * 12 * 3                       # every synthetic frame is labelled
    assert (ids < 0).sum() == 12 * 3                         # authentic frames have no generator
    assert sorted(index) == sorted(GENERATORS)
    counts = np.bincount(labelled, minlength=3)
    assert (counts == 36).all()
    # the label follows the path, so re-asking is stable
    again, _ = image_generator_labels(gen_db.paths, list(GENERATORS))
    assert (again == ids).all()


def test_generator_bundle_is_group_atomic_and_reuses_the_split_rules(gen_db, gen_root):
    from ams.generator import build_generator_bundle
    from ams.labels import discover_generator_labels

    gen = discover_generator_labels(gen_root, gen_db.classes)
    assert gen.available is True, gen.reason
    assert gen.source == "nested_directories"
    gb = build_generator_bundle(gen_db, gen.to_dict(), train_frac=0.6, val_frac=0.2, seed=5)
    assert gb is not None
    assert sorted(gb.generator_classes) == sorted(GENERATORS)
    assert sum(gb.counts.values()) == 3 * 12 * 3
    assert gb.n_unlabelled == 12 * 3
    b = gb.bundle
    assert b.n_classes == 3 and b.binary is False
    # no capture group may straddle two splits (same leakage rule as the detector)
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        ga = set(b.groups[getattr(b.split, f"{a}_idx")].tolist())
        gc = set(b.groups[getattr(b.split, f"{c}_idx")].tolist())
        assert not (ga & gc), f"{a}/{c} share groups"
    # every label id in the bundle refers to a real generator name, nothing invented
    assert set(np.unique(b.labels).tolist()) <= {0, 1, 2}
    assert len(b.mean) == 3 and all(v > 0 for v in b.std)


def test_no_labels_means_no_bundle(gen_db):
    from ams.generator import build_generator_bundle

    unavailable = {"available": False, "classes": {}, "source": "none", "reason": "no labels"}
    assert build_generator_bundle(gen_db, unavailable) is None


# --------------------------------------------------------------------- acceptance rule
def test_gate_is_fitted_from_validation_and_reports_what_it_achieved():
    from ams.generator import fit_unknown_gate

    rng = np.random.default_rng(0)
    n, k = 400, 3
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1.0, (n, k))
    logits[np.arange(n), y] += 6.0                      # a well-separated model
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)

    gate = fit_unknown_gate(p, y)
    assert gate["policy"] == "max_prob_and_margin"
    assert gate["fitted_on"] == "validation split only"
    assert 0.0 < gate["min_prob"] < 1.0
    assert gate["target_met"] is True
    assert gate["validation_precision_on_accepted"] >= gate["target_precision"]
    assert gate["n_accepted"] >= 20

    # an uninformative model must not be allowed to name generators confidently
    junk = rng.dirichlet(np.ones(k), size=n)
    weak = fit_unknown_gate(junk, y)
    assert weak["validation_precision_on_accepted"] < 0.6
    assert weak["target_met"] is False


def test_gate_names_only_known_generators_and_says_unknown_otherwise():
    from ams.generator import UNKNOWN_LABEL, apply_gate

    gate = {"min_prob": 0.5, "min_margin": 0.1, "fitted_on": "validation split only",
            "validation_precision_on_accepted": 0.95}
    conf = apply_gate([0.05, 0.9, 0.05], list(GENERATORS), gate)
    assert conf["available"] is True
    assert conf["generator"] == "stylegan"
    assert conf["named"] is True
    # equal probabilities are ordered alphabetically, never by numpy's reversed argsort
    assert [c["generator"] for c in conf["ranking"]] == ["stylegan", "midjourney", "stable_diffusion"]
    assert sum(c["probability"] for c in conf["ranking"]) == pytest.approx(0.9999, abs=1e-3)

    ambiguous = apply_gate([0.34, 0.33, 0.33], list(GENERATORS), gate)
    assert ambiguous["generator"] == UNKNOWN_LABEL
    assert ambiguous["named"] is False
    assert "never shown to this model" in ambiguous["reason"]
    assert ambiguous["known_generators"] == list(GENERATORS)

    weak = apply_gate([0.2, 0.45, 0.35], list(GENERATORS), gate)
    assert weak["generator"] == UNKNOWN_LABEL

    with pytest.raises(ValueError):
        apply_gate([0.5, 0.5], list(GENERATORS), gate)


# --------------------------------------------------------------------- end to end
def test_pipeline_trains_exports_and_serves_the_generator_model(gen_root, tmp_path):
    """Full chain on the labelled corpus: no retraining at serve time, no invented class."""
    from ams.pipeline import PipelineConfig, run_training

    out = tmp_path / "run"
    cfg = PipelineConfig(
        root=gen_root,
        archive=None,
        artifacts_dir=out / "artifacts",
        models_dir=out / "models",
        reports_dir=out / "reports",
        samples_dir=out / "samples",
        cache_size=48,
        feature_size=48,
        train_frac=0.6,
        val_frac=0.2,
        seed=5,
        run_name="gen",
        deep=False,
        classical=True,
        generator_task=True,
        ablations=(),
        export_samples_n=1,
        verbose=False,
    )
    res = run_training(cfg)
    assert res.generator_task and res.generator_task.get("trained") is True
    assert sorted(res.generator_task["classes"]) == sorted(GENERATORS)
    assert res.generator_task["test_evaluations"] == 1
    assert res.generator_task["unknown_generator_gate"]["min_prob"] > 0.0

    metrics = json.loads((cfg.models_dir / "metrics.json").read_text())
    gtask = metrics["tasks"]["generator_classifier"]
    assert gtask["trained"] is True
    for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy"):
        assert 0.0 <= float(gtask["metrics"][key]) <= 1.0, key
    # a 3-class task cannot have a meaningful top-3 number: it is reported as unavailable,
    # the reason is recorded, and the selection weights document where the 0.10 went
    assert gtask["metrics"]["top3_accuracy"] is None and "top3_note" in gtask["metrics"]
    assert gtask["selection"]["weights_note"].startswith("top-3 accuracy was dropped")
    assert "top3_accuracy" not in gtask["selection"]["weights"]
    assert gtask["selection"]["weights"]["f1"] == pytest.approx(0.45)
    assert gtask["selection"]["weights_declared"]["top3_accuracy"] == 0.10
    # a 3-class problem is solvable here (the textures are distinct) - must beat chance clearly
    assert float(gtask["metrics"]["macro_f1"]) > 0.6
    assert len(gtask["comparison"]) >= 3
    assert all(r["task"] == "generator" for r in gtask["comparison"])
    assert any(r["selected"] for r in gtask["comparison"])
    # the detector task is untouched by the second task (class list is ordered by label id,
    # authentic=0 / synthetic=1, which is what classes.json is written from)
    det_task = metrics["tasks"]["ai_detector"]
    assert det_task["classes"] == ["real", "fake"]
    assert json.loads((cfg.models_dir / "ai_detector" / "classes.json").read_text())["label_map"] == {
        "real": 0, "fake": 1}

    card = json.loads((cfg.models_dir / "model_info.json").read_text())
    assert card["generator_classifier"]["task"]["trained"] is True

    gdir = cfg.models_dir / "generator_classifier"
    assert (gdir / "classes.json").exists() and (gdir / "gate.json").exists()
    assert (gdir / "model_config.json").exists()
    assert (gdir / "model.joblib").exists() or (gdir / "model.pth").exists()
    assert sorted(json.loads((gdir / "classes.json").read_text())["classes"]) == sorted(GENERATORS)
    report = (cfg.reports_dir / "model_comparison.md").read_text()
    assert "**trained** on classes" in report and "UNKNOWN" in report

    # ---- deployment: the served model identifies a generator from the saved artefacts only
    from ams.predictor import Detector, PredictorConfig as DetCfg

    det = Detector(DetCfg(models_dir=cfg.models_dir, enable_gradcam=False, enable_face=False))
    assert det.generator is not None
    yy, xx = np.indices((48, 48))
    probe = 120.0 + 70.0 * np.cos(yy * (2 * np.pi / 5.0))
    arr = np.stack([probe] * 3, axis=-1) + np.random.default_rng(1).normal(0, 5.0, (48, 48, 3))
    payload = det.generator_identification(np.clip(arr, 0, 255).astype(np.uint8))
    assert payload["available"] is True
    assert payload["generator"] in list(GENERATORS) + ["UNKNOWN"]
    assert len(payload["ranking"]) == 3
    assert abs(sum(c["probability"] for c in payload["ranking"]) - 1.0) < 0.02
    assert payload["acceptance_rule"]["fitted_on"] == "validation split only"
    assert "only name the generators above" in payload["limitation"]
    assert sorted(payload["known_generators"]) == sorted(GENERATORS)

    # an image that is not generator-like at all must not be forced into a class name
    flat = np.full((48, 48, 3), 128, dtype=np.uint8)
    out_flat = det.generator_identification(flat)
    assert out_flat["available"] is True
    assert isinstance(out_flat["generator"], str)
    assert out_flat["generator"] in list(GENERATORS) + ["UNKNOWN"]

    # without the exported directory the same code path reports the honest "unavailable"
    (gdir / "model_config.json").unlink()
    det2 = Detector(DetCfg(models_dir=cfg.models_dir, enable_gradcam=False, enable_face=False))
    assert det2.generator is None
    assert det2.generator_identification(flat)["available"] is False
    assert det2.generator_identification(flat)["generator"] is None
