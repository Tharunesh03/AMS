"""Explanation engine: human-readable evidence built from *measured* image signals.

Two parts:

1. :func:`build_reference` - per-signal distributions estimated on the **training split**
   only, saved to ``models/reference_stats.json`` so the deployed app does not need the
   dataset (no retraining, no leakage from val/test).
2. :class:`EvidenceEngine` - measures the same signals on an uploaded image, converts
   departures from the authentic-capture reference into LOW/MEDIUM/HIGH evidence, and
   writes both a simple and a technical explanation.

Nothing here can claim "this is AI" on its own: the verdict always comes from the trained
supervised model, and these signals are reported as *why the model may have decided so*.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import (
    FeatureOptions,
    _blockiness,
    _box_blur,
    _gray,
    _hog,
    _lbp_hist,
    _moments,
    _radial_spectrum,
)
from .gradcam import cam_bbox

SIGNALS: List[Dict[str, str]] = [
    {"key": "hp_std", "label": "micro-texture energy", "unit": "relative",
     "meaning": "sensor noise and skin/pore detail vs synthesised smoothness"},
    {"key": "noise_floor", "label": "residual noise floor", "unit": "relative",
     "meaning": "second-order residual energy in flat regions"},
    {"key": "block_v", "label": "vertical 8px grid energy", "unit": "ratio",
     "meaning": "JPEG / VAE decoding block structure"},
    {"key": "block_h", "label": "horizontal 8px grid energy", "unit": "ratio",
     "meaning": "JPEG / VAE decoding block structure"},
    {"key": "spec_peak", "label": "frequency-spectrum anomaly", "unit": "deviation",
     "meaning": "peaks/voids in the radial spectrum typical of upsampled generations"},
    {"key": "lbp_entropy", "label": "local micro-pattern diversity", "unit": "nats",
     "meaning": "diversity of local binary patterns (texture complexity)"},
    {"key": "edge_asym", "label": "edge-direction balance", "unit": "ratio",
     "meaning": "anisotropy between horizontal and vertical edge energy"},
    {"key": "contrast", "label": "tonal contrast", "unit": "relative",
     "meaning": "global contrast after per-image normalisation (reported for context only)"},
    {"key": "hog_diag", "label": "diagonal-gradient concentration", "unit": "share",
     "meaning": "share of gradient energy in diagonal orientation bins"},
    {"key": "luma_kurtosis", "label": "tonal-distribution kurtosis", "unit": "ratio",
     "meaning": "tailedness of the brightness histogram; very peaked or very flat histograms are"
                " common in synthesised faces that lack sensor tonal spread"},
]


def measure_signals(img: np.ndarray, opts: Optional[FeatureOptions] = None) -> Dict[str, float]:
    """Compute the explanation signals for one RGB uint8 image (``[H,W,3]``)."""
    opts = opts or FeatureOptions()
    x = img.astype(np.float32)[None] / 255.0
    if opts.photometric == "gray":
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        s = x.std(axis=(1, 2, 3), keepdims=True) + 1e-5
        x = (x - m) / s
    g = _gray(x)
    hp = g - _box_blur(g, 1)
    hp2 = hp - _box_blur(hp, 1)
    blocks = np.log(_blockiness(g, 8) + 1e-4)
    spec = _radial_spectrum(g, opts.spec_bins)
    lbp = _lbp_hist(g)
    p = np.clip(lbp, 1e-9, 1)
    lbp_ent = float(-(p * np.log(p)).sum(1)[0])
    e_h = float(np.abs(np.diff(g, axis=2)).mean())
    e_v = float(np.abs(np.diff(g, axis=1)).mean())
    hog = _hog(g, opts.grid, opts.bins)[0].reshape(opts.bins, opts.grid, opts.grid).mean(axis=(1, 2))
    diag_bins = [i for i in range(opts.bins) if i in (2, 3, 6, 7)]
    hog_diag = float(hog[diag_bins].sum() / (hog.sum() + 1e-9))
    skew, kurt = _moments(g.reshape(1, -1))
    return {
        "hp_std": float(hp.std()),
        "noise_floor": float(hp2.std()),
        "block_v": float(blocks[0, 4]),
        "block_h": float(blocks[0, 5]),
        "spec_peak": float(np.abs(spec).max()),
        "lbp_entropy": lbp_ent,
        "edge_asym": float(np.log(e_h + 1e-4) - np.log(e_v + 1e-4)),
        "contrast": float(g.std()),
        "hog_diag": hog_diag,
        "luma_kurtosis": float(kurt[0]),
    }


def build_reference(db, out_path: Path, progress: bool = True) -> Dict[str, object]:
    """Distribution of each signal on the training split, per class (mean/std/quantiles)."""
    idx = db.split.train_idx           # indices only - the array itself stays memory-mapped
    size = db.cache_size
    from .transforms import resize_any, to_chw_float01  # local import: keeps CLI light

    per_class: Dict[str, Dict[str, Dict[str, float]]] = {}
    store: Dict[str, Dict[str, List[float]]] = {c: {s["key"]: [] for s in SIGNALS} for c in db.classes}
    for i, gi in enumerate(idx):
        if progress and i and i % 2000 == 0:
            print(f"  reference stats {i}/{len(idx)}", flush=True)
        img = db.images[int(gi)]
        x = resize_any(to_chw_float01(img[None]), size)[0].permute(1, 2, 0).mul(255).byte().numpy()
        sig = measure_signals(x)
        cls = db.classes[int(db.labels[int(gi)])]
        for k, v in sig.items():
            if k in store[cls]:
                store[cls][k].append(float(v))
    for cls, sigs in store.items():
        per_class[cls] = {}
        for k, vals in sigs.items():
            a = np.asarray(vals, dtype=np.float64)
            if a.size == 0:
                continue
            per_class[cls][k] = {
                "n": int(a.size),
                "mean": float(a.mean()),
                "std": float(a.std() + 1e-9),
                "median": float(np.median(a)),
                "q01": float(np.quantile(a, 0.01)),
                "q99": float(np.quantile(a, 0.99)),
            }
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "estimated_on": "training split only (no validation/test images used)",
        "authentic_class": db.negative_class,
        "synthetic_class": db.positive_class,
        "grid": int(size),
        "photometric": "gray",
        "signals": SIGNALS,
        "per_class": per_class,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return payload


@dataclass
class EvidenceItem:
    signal: str
    label: str
    strength: str          # LOW | MEDIUM | HIGH
    direction: str
    value: float
    real_median: float
    z: float
    meaning: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "signal": self.signal, "label": self.label, "strength": self.strength,
            "direction": self.direction, "value": round(self.value, 4),
            "authentic_median": round(self.real_median, 4), "z": round(self.z, 2),
            "meaning": self.meaning,
        }


class EvidenceEngine:
    """Turns measured signals + the model's own saliency into plain language."""

    def __init__(self, reference: Dict[str, object]) -> None:
        self.ref = reference
        keys = list(reference.get("per_class", {}).keys()) or ["real", "fake"]
        # which key means "authentic" comes from the training run, not from list order
        self.neg_class = str(reference.get("authentic_class") or keys[0])
        self.pos_class = str(reference.get("synthetic_class") or (keys[1] if len(keys) > 1 else keys[0]))
        if self.neg_class not in keys:
            self.neg_class = keys[0]
        if self.pos_class not in keys:
            self.pos_class = keys[1] if len(keys) > 1 else keys[0]
        self.signals = {s["key"]: s for s in reference.get("signals", SIGNALS)}

    @staticmethod
    def load(path: Path) -> Optional["EvidenceEngine"]:
        p = Path(path)
        if not p.exists():
            return None
        return EvidenceEngine(json.loads(p.read_text()))

    # ------------------------------------------------------------------ internals
    def _stats(self, cls: str, key: str) -> Optional[Dict[str, float]]:
        return (self.ref.get("per_class", {}) or {}).get(cls, {}).get(key)

    def evaluate(self, img: np.ndarray, model_out: Dict[str, object], cam: Optional[GradCamLike] = None,  # type: ignore[valid-type]
                 face: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        sig = measure_signals(img, FeatureOptions(photometric=self.ref.get("photometric", "gray"),
                                                   spec_bins=16, grid=8, bins=9))
        items: List[EvidenceItem] = []
        for s in SIGNALS:
            k = s["key"]
            neg = self._stats(self.neg_class, k)
            pos = self._stats(self.pos_class, k)
            if not neg:
                continue
            v = float(sig.get(k, 0.0))
            z = (v - neg["mean"]) / neg["std"]
            sep = abs((pos or neg)["mean"] - neg["mean"]) / neg["std"] if pos else 0.0
            # strength combines how far the image is from authentic captures AND how
            # discriminative the signal is on this corpus
            mag = abs(z)
            if mag >= max(1.5, 0.6 * sep):
                strength = "HIGH"
            elif mag >= max(0.7, 0.3 * sep):
                strength = "MEDIUM"
            else:
                strength = "LOW"
            direction = "above" if z > 0 else "below"
            items.append(
                EvidenceItem(
                    signal=k, label=s["label"], strength=strength, direction=direction,
                    value=v, real_median=neg["median"], z=z,
                    meaning=f"{s['meaning']}; this image is {abs(z):.1f} SD {direction} the authentic-capture median "
                            f"({neg['median']:.3f}) and {sep:.1f} SD separates the two classes in the training data",
                )
            )
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        items.sort(key=lambda it: (order[it.strength], -abs(it.z)))
        top = [it for it in items if it.strength != "LOW"][:3] or items[:2]

        verdict = str(model_out.get("verdict", "unknown"))
        conf = float(model_out.get("confidence", float("nan")))
        simple, technical = self._narrate(verdict, conf, top, model_out, cam, face, items)
        return {
            "signals": {k: float(v) for k, v in sig.items()},
            "evidence": [it.to_dict() for it in items],
            "headline_evidence": [it.to_dict() for it in top],
            "simple_explanation": simple,
            "technical_explanation": technical,
            "reference": {
                "grid": self.ref.get("grid"), "estimated_on": self.ref.get("estimated_on"),
                "classes": {"authentic": self.neg_class, "synthetic": self.pos_class},
            },
        }

    # -------------------------------------------------------------------- wording
    def _narrate(self, verdict, conf, top, model_out, cam, face, items) -> Tuple[str, str]:
        ev_txt = ", ".join(f"{t.label.lower()} ({t.strength.lower()})" for t in top) or "no strong signal departure"
        prob = float(model_out.get("probability", float("nan")))
        cls_pos = self.pos_class
        simple = (
            f"The trained model calls this image {verdict.lower()} with {conf*100:.0f}% confidence "
            f"(P({cls_pos}) = {prob:.3f} against a decision threshold of "
            f"{float(model_out.get('threshold', 0.5)):.3f}). Supporting measurements: {ev_txt}. "
            "This is a probabilistic estimate about image provenance, not proof."
        )
        tech_bits: List[str] = []
        for t in top:
            tech_bits.append(
                f"{t.label}: {t.value:.4f} vs authentic median {t.real_median:.4f} "
                f"(z={t.z:+.2f}, {t.strength})"
            )
        if cam is not None and getattr(cam, "available", False):
            bb = cam_bbox(cam.heatmap) if cam.heatmap is not None else None
            if bb:
                tech_bits.append(
                    f"Grad-CAM saliency from layer '{cam.target_layer}' concentrates "
                    f"top-quartile mass in box {np.round(bb,2).tolist()} (x0,y0,x1,y1 in unit coords)"
                )
        if face:
            n = int(face.get("n_faces", 0) or 0)
            tech_bits.append(
                f"analytical face module: {n} face(s) detected; consistency summary -> "
                + (str(face.get("summary", {})).replace("'", "")[:220] if face.get("summary") else "not available")
            )
        technical = (
            f"Model: {model_out.get('model','?')} ({model_out.get('family','?')}, input "
            f"{model_out.get('input_size','?')}px). Signals are measured on the model input grid "
            f"({self.ref.get('grid')}px, per-image z-scored) and compared with training-split reference "
            f"distributions for authentic captures. " + "; ".join(tech_bits) + "."
        )
        return simple, technical


try:  # optional typing aid only
    from .gradcam import GradCamResult as GradCamLike  # type: ignore
except Exception:  # noqa: E402, BLE001
    class GradCamLike:  # type: ignore
        pass
