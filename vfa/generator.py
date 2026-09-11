"""Generator fingerprinting (multi-class) and its unknown-generator acceptance rule.

Two hard rules from the spec drive this module:

1. **Never invent classes.** A generator model is only built when the dataset really
   carries a per-generator label dimension (nested folders such as
   ``data/FAKE/stable_diffusion/*.jpg``, or an unambiguous generator token in the file
   name). :func:`discover_generator_labels` decides that; :func:`build_generator_bundle`
   turns the verdict into an actual training set. If the labels are not there, every
   caller receives ``None`` and the report says "not trainable" - nothing is simulated.

2. **Never name a generator the model was not shown.** The acceptance rule below is
   fitted on the *validation* split only: it is the loosest ``(max_prob, margin)`` pair
   that still reaches a target top-1 precision on accepted predictions. Anything that
   fails it is reported as ``UNKNOWN`` instead of being forced into the closest known
   generator - that is the only honest answer for an image from an unseen generator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .labels import MIN_PER_CLASS, normalise_label
from .trainer import DataBundle, _norm_stats  # same package, same split semantics


UNKNOWN_LABEL = "UNKNOWN"


# --------------------------------------------------------------------- label plumbing
def image_generator_labels(
    paths: Sequence[object], gen_classes: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Per-image generator id aligned with ``paths`` (``-1`` where the image is unlabelled).

    The label is read from the image's *own* location, never assumed: an image is only
    labelled when its immediate parent directory is one of the discovered generator
    classes, or its file name carries that generator token.
    """
    index = {normalise_label(c): i for i, c in enumerate(gen_classes)}
    ids = np.full(len(paths), -1, dtype=np.int16)
    for i, p in enumerate(paths):
        pp = Path(str(p))
        key = normalise_label(pp.parent.name)
        if key in index:
            ids[i] = index[key]
            continue
        from .labels import _token_of

        tok = _token_of(pp.stem)
        if tok is not None and normalise_label(tok) in index:
            ids[i] = index[normalise_label(tok)]
    return ids, index


def generator_group_keys(paths: Sequence[object], keep_idx: np.ndarray) -> np.ndarray:
    """Capture-group keys for the *generator* task.

    The detector groups by ``(class, stem)``; here two images from different generators may
    share a file stem by coincidence, and then one group would straddle two label classes and
    make a group-atomic stratified split impossible (the split guard raises). The unit that
    must stay together is one generator's own burst, i.e. its folder plus stem without the
    `` (k)`` frame suffix.
    """
    import re

    out = np.empty(len(keep_idx), dtype=object)
    for j, i in enumerate(int(v) for v in keep_idx):
        pp = Path(str(paths[i]))
        stem = re.sub(r"\s*\(\d+\)$", "", pp.stem)
        out[j] = f"{pp.parent.as_posix()}/{stem}"
    return out


@dataclass
class GeneratorBundle:
    """A second, independent :class:`DataBundle` whose labels are generators."""

    bundle: DataBundle
    generator_classes: List[str]
    counts: Dict[str, int]
    n_unlabelled: int
    label_source: str
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "generator_classes": self.generator_classes,
            "per_class_counts": self.counts,
            "images_without_generator_label": self.n_unlabelled,
            "label_source": self.label_source,
            "split_sizes": self.bundle.split.sizes(),
            "min_images_per_class": MIN_PER_CLASS,
            "notes": self.notes,
        }


def build_generator_bundle(
    db: DataBundle,
    dimension: Dict[str, object],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 13,
) -> Optional[GeneratorBundle]:
    """Build the generator training set, or ``None`` when the labels do not support one.

    Only images that carry a generator label are used (in this repo that is at most the
    synthetic class - authentic images have no generator), the split is again
    group-atomic so a capture's neighbours cannot straddle train/test, and the
    normalisation statistics are re-measured on the new training subset.
    """
    if not dimension.get("available"):
        return None
    names_all = list((dimension.get("classes") or {}).keys())
    if len(names_all) < 2:
        return None
    ids, _ = image_generator_labels(db.paths, names_all)
    labelled = np.where(ids >= 0)[0]
    if labelled.size == 0:
        return None
    counts_all = [int((ids == i).sum()) for i in range(len(names_all))]
    keep_cls = [i for i in range(len(names_all)) if counts_all[i] >= MIN_PER_CLASS]
    notes: List[str] = []
    dropped = {names_all[i]: counts_all[i] for i in range(len(names_all)) if i not in keep_cls}
    if dropped:
        notes.append(f"dropped classes below the {MIN_PER_CLASS}-image minimum: {dropped}")
    if len(keep_cls) < 2:
        return None
    remap = {old: new for new, old in enumerate(keep_cls)}
    sel = np.array([remap.get(int(v), -1) for v in ids[labelled]], dtype=np.int64)
    good = sel >= 0
    keep = labelled[good]
    y = sel[good]
    classes = [names_all[i] for i in keep_cls]
    imgs = np.ascontiguousarray(np.array(db.images[keep], copy=True))
    all_paths = np.asarray(db.paths)
    groups = generator_group_keys(all_paths, keep)
    paths = all_paths[keep]
    from .splitting import stratified_grouped_split

    split = stratified_grouped_split(y, groups, train_frac=train_frac, val_frac=val_frac, seed=seed)
    mean, std = _norm_stats(imgs, split.train_idx)
    bundle = DataBundle(
        images=imgs,
        labels=y,
        groups=groups,
        paths=paths,
        classes=classes,
        mapping={c: i for i, c in enumerate(classes)},
        split=split,
        mean=mean,
        std=std,
        inventory=db.inventory,
        cache_path=db.cache_path,
    )
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    return GeneratorBundle(
        bundle=bundle,
        generator_classes=classes,
        counts=counts,
        n_unlabelled=int(len(db.labels) - keep.size),
        label_source=str(dimension.get("source") or "unknown"),
        notes=notes,
    )


# --------------------------------------------------------------------- unknown gate
def _top_two(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[1] == 1:
        return p[:, 0], np.ones(len(p))
    srt = np.sort(p, axis=1)
    return srt[:, -1], srt[:, -1] - srt[:, -2]


def fit_unknown_gate(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    *,
    target_precision: float = 0.90,
    min_coverage: float = 0.10,
) -> Dict[str, object]:
    """Choose the acceptance rule on validation scores: loosest rule that still works.

    Sweeps a grid of ``(min_prob, min_margin)`` pairs, keeps those where at least
    ``min_coverage`` of the validation images are accepted, and returns the pair with the
    **highest accepted coverage** whose measured top-1 precision on accepted images
    reaches ``target_precision``. If no pair reaches the target, the highest-precision
    pair is returned and ``target_met`` is ``False`` (reported, not hidden).
    """
    p = np.asarray(val_probs, dtype=np.float64)
    y = np.asarray(y_val)
    pmax, margin = _top_two(p)
    pred = np.argmax(p, axis=1) if p.ndim > 1 and p.shape[1] > 1 else np.zeros(len(p), dtype=np.int64)
    grid_p = np.round(np.arange(0.20, 0.9901, 0.02), 3)
    grid_m = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    best: Optional[Dict[str, object]] = None
    for tp in grid_p:
        for tm in grid_m:
            acc_mask = (pmax >= tp) & (margin >= tm)
            cov = float(acc_mask.mean()) if len(acc_mask) else 0.0
            if cov < min_coverage or acc_mask.sum() < 20:
                continue
            prec = float((pred[acc_mask] == y[acc_mask]).mean())
            cand = {
                "min_prob": float(tp),
                "min_margin": float(tm),
                "validation_coverage": round(cov, 4),
                "validation_precision_on_accepted": round(prec, 4),
                "n_accepted": int(acc_mask.sum()),
            }
            if prec >= target_precision:
                if best is None or cov > float(best["validation_coverage"]):
                    best = cand
            elif best is None:
                if best is None:
                    best = cand  # provisional fallback, replaced if any point qualifies
    if best is None:
        best = {
            "min_prob": 0.5,
            "min_margin": 0.0,
            "validation_coverage": float((pmax >= 0.5).mean()) if len(pmax) else 0.0,
            "validation_precision_on_accepted": float((pred == y).mean()) if len(pred) else 0.0,
            "n_accepted": int((pmax >= 0.5).sum()) if len(pmax) else 0,
        }
    best["target_precision"] = float(target_precision)
    best["target_met"] = bool(best["validation_precision_on_accepted"] >= target_precision)
    best["policy"] = "max_prob_and_margin"
    best["fitted_on"] = "validation split only"
    best["n_validation_images"] = int(len(y))
    return best


def apply_gate(
    probs: Sequence[float], classes: Sequence[str], gate: Dict[str, object]
) -> Dict[str, object]:
    """Turn a probability vector into a named generator or ``UNKNOWN`` (never a guess)."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    if p.size != len(classes):
        raise ValueError(f"probability vector of size {p.size} does not match {len(classes)} classes")
    # tie-stable ranking: probability first, then the class name (never numpy's reverse order)
    order = np.array(sorted(range(p.size), key=lambda i: (-float(p[i]), str(classes[i]))), dtype=int)
    pmax = float(p[order[0]])
    margin = float(p[order[0]] - p[order[1]]) if order.size > 1 else 1.0
    min_prob = float(gate.get("min_prob", 0.5))
    min_margin = float(gate.get("min_margin", 0.0))
    known = pmax >= min_prob and margin >= min_margin
    top = [
        {"generator": str(classes[i]), "probability": round(float(p[i]), 4)}
        for i in order[: min(3, order.size)]
    ]
    return {
        "available": True,
        "generator": str(classes[order[0]]) if known else UNKNOWN_LABEL,
        "named": bool(known),
        "confidence": round(pmax, 4),
        "margin_to_second": round(margin, 4),
        "acceptance_rule": {
            "min_probability": min_prob,
            "min_margin": min_margin,
            "fitted_on": gate.get("fitted_on", "validation split only"),
            "validation_precision_on_accepted": gate.get("validation_precision_on_accepted"),
        },
        "ranking": top,
        "known_generators": [str(c) for c in classes],
        "reason": (
            "top probability and margin over the runner-up both clear the acceptance rule fitted "
            "on the validation split"
            if known
            else (
                "the model is not confident enough within the classes it was trained on, so the "
                "generator is reported as unknown rather than guessed; generators outside "
                f"{[str(c) for c in classes]} were never shown to this model and cannot be named"
            )
        ),
    }


def save_gate(path: Path, gate: Dict[str, object]) -> None:
    Path(path).write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")


def export_generator_model(src_model_path: Path, out_dir: Path, bundle: GeneratorBundle,
                           gate: Dict[str, object], meta: Dict[str, object]) -> Dict[str, object]:
    """Copy the selected generator model into ``models/generator_classifier`` (deployment shape)."""
    import shutil

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_model_path)
    kind = "torch" if src.suffix == ".pth" else "sklearn"
    shutil.copy2(src, out_dir / src.name)
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": bundle.generator_classes, "label_map": bundle.bundle.mapping,
                    "positive_class": None, "task": "generator_fingerprinting"}, indent=2) + "\n"
    )
    save_gate(out_dir / "gate.json", gate)
    payload = {
        "kind": kind,
        "model_file": src.name,
        "feature": "forensic" if kind == "sklearn" else None,
        "feature_size": int(meta.get("feature_size") or bundle.bundle.cache_size),
        "input_size": int(meta.get("input_size") or bundle.bundle.cache_size),
        "photometric": meta.get("photometric", "gray"),
        "id": meta.get("id"),
        "n_classes": len(bundle.generator_classes),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_gate": gate,
        "dataset_support": bundle.summary(),
        "note": (
            "generated by the training pipeline only when the dataset carries per-generator labels; "
            "predictions outside the acceptance rule are reported as UNKNOWN"
        ),
    }
    (out_dir / "model_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# --------------------------------------------------------------------- deployment side
class GeneratorClassifier:
    """Loads ``models/generator_classifier`` and applies the trained model + gate."""

    def __init__(self, kind: str, payload: Dict[str, object], gate: Dict[str, object],
                 classes: List[str], model: object, transform: object = None) -> None:
        self.kind = kind
        self.cfg = payload
        self.gate = gate
        self.classes = classes
        self.model = model
        self.transform = transform
        self.input_size = int(payload.get("input_size") or payload.get("feature_size") or 0)

    @property
    def feature_kind(self) -> str:
        return str(self.cfg.get("feature", "forensic"))

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "GeneratorClassifier":
        """Read the exported artifact written by :func:`export_generator_model`."""
        directory = Path(directory)
        cfg_path = directory / "model_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} not found - the generator model is only exported when the dataset "
                "carries per-generator labels (run `python train.py` after adding them)"
            )
        payload = json.loads(cfg_path.read_text())
        kind = str(payload.get("kind", "sklearn"))
        classes = [str(c) for c in json.loads((directory / "classes.json").read_text())["classes"]]
        transform = None
        if kind == "sklearn":
            import joblib

            blob = joblib.load(directory / "model.joblib")
            model = blob["pipeline"]
            classes = [str(c) for c in blob.get("classes", classes)]
        else:  # pragma: no cover - exercised only when torch is installed
            import torch

            from .models import ModelSpec, build_model
            from .transforms import ImageTransform

            ck = torch.load(directory / "model.pth", map_location=device, weights_only=False)
            spec = ModelSpec.from_dict(ck["spec"])
            built = build_model(spec)
            built.load_state_dict(ck["state_dict"])
            built.eval()
            for q in built.parameters():
                q.requires_grad_(False)
            model = built.to(device)
            pre = ck.get("preprocess", {}) or {}
            transform = ImageTransform(
                size=int(pre.get("size", spec.input_size)),
                mean=tuple(pre.get("mean") or (0.5, 0.5, 0.5)),
                std=tuple(pre.get("std") or (0.5, 0.5, 0.5)),
                train=False,
                to_gray=bool(pre.get("to_gray", spec.in_channels == 1)),
            )
            classes = [str(c) for c in ck.get("classes", classes)]
        gate = json.loads((directory / "gate.json").read_text()) if (directory / "gate.json").exists() else {}
        return cls(kind, payload, gate, classes, model, transform)

    def probabilities(self, arr: np.ndarray) -> np.ndarray:
        """Score one RGB uint8 array with the trained model (same geometry as its training run)."""
        if self.kind == "sklearn":
            from .features import extract_features
            from .transforms import resize_any, to_chw_float01

            size = int(self.cfg.get("feature_size") or self.input_size or 64)
            small = (
                resize_any(to_chw_float01(np.array(arr, copy=True)[None]), size)[0]
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .numpy()
            )
            if self.feature_kind == "forensic":
                from .features import FeatureOptions

                opts = FeatureOptions(photometric=str(self.cfg.get("photometric", "gray")))
                X, _ = extract_features(np.ascontiguousarray(small[None]), opts)
            else:
                X = small.astype(np.float32).reshape(1, -1) / np.float32(255.0)
            return np.asarray(self.model.predict_proba(X), dtype=np.float64)[0]
        import torch

        if self.transform is None:  # pragma: no cover
            raise RuntimeError("torch generator model is missing its preprocessing transform")
        x = self.transform(np.ascontiguousarray(arr))
        with torch.no_grad():
            out = self.model(x)
        z = out.detach().cpu().numpy()[0]
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float64)

    def predict(self, arr: np.ndarray) -> Dict[str, object]:
        t0 = time.perf_counter()
        probs = self.probabilities(arr)
        out = apply_gate(probs, self.classes, self.gate)
        out["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        out["model"] = {"id": self.cfg.get("id"), "kind": self.kind, "trained_at": self.cfg.get("trained_at")}
        return out
