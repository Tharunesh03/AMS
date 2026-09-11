"""Production inference: loads the trained artifact and predicts. Never retrains.

    models/ai_detector/model.pth | model.joblib  +  classes.json + preprocessing.json
    models/metrics.json  (reported metrics, thresholds, model comparison)
    models/reference_stats.json (optional: enables the evidence engine)

The predictor is what ``app/`` imports; the same object backs the CLI, so a prediction in
the terminal and in the browser are byte-identical.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 40_000_000
FACE_WORK_SIZE = 512

VERDICT_LABEL = {1: "AI-GENERATED / SYNTHETIC", 0: "AUTHENTIC CAPTURE"}


@dataclass
class PredictorConfig:
    models_dir: Path = Path("models")
    device: str = "cpu"
    enable_gradcam: bool = True
    enable_face: bool = True
    enable_generator: bool = True
    max_input_px: int = 2400


class ModelNotTrained(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Detector:
    """Loads the selected supervised model once and reuses it for every request."""

    def __init__(self, cfg: Optional[PredictorConfig] = None) -> None:
        self.cfg = cfg or PredictorConfig()
        root = Path(self.cfg.models_dir)
        self.root = root
        det = root / "ai_detector"
        self.classes_path = det / "classes.json"
        if not self.classes_path.exists():
            raise ModelNotTrained(
                f"{self.classes_path} not found. Train once with `python train.py` and commit the "
                "models/ directory; the app never retrains at serve time."
            )
        self.classes_meta = json.loads(self.classes_path.read_text())
        self.classes: List[str] = list(self.classes_meta["classes"])
        self.label_map: Dict[str, int] = {k: int(v) for k, v in self.classes_meta["label_map"].items()}
        self.positive_class = self.classes_meta.get("positive_class") or next(
            (c for c, i in self.label_map.items() if i == int(self.classes_meta.get("positive_index", 1))),
            self.classes[-1],
        )
        self.pos_index = int(self.label_map.get(self.positive_class, self.classes_meta.get("positive_index", 1)))
        self.neg_index = 0 if self.pos_index == 1 else 1
        self.meaning_by_id = {int(k): str(v) for k, v in (self.classes_meta.get("labels_meaning") or {}).items()}
        self.preprocessing = json.loads((root / "preprocessing.json").read_text()) if (root / "preprocessing.json").exists() else {}
        self.model_info = json.loads((root / "model_info.json").read_text()) if (root / "model_info.json").exists() else {}
        self.metrics_doc = json.loads((root / "metrics.json").read_text()) if (root / "metrics.json").exists() else {}
        task = (self.metrics_doc.get("tasks", {}) or {}).get("ai_detector", {})
        self.threshold = float(task.get("threshold", 0.5))

        self.kind: str = ""
        self.model = None
        self.transform = None
        self.sklearn_payload = None
        torch_ckpt = det / "model.pth"
        joblib_ckpt = det / "model.joblib"
        if torch_ckpt.exists():
            self.kind = "torch"
            self._load_torch(torch_ckpt)
        elif joblib_ckpt.exists():
            self.kind = "classical"
            self._load_classical(joblib_ckpt)
        else:
            raise ModelNotTrained(f"no model.pth or model.joblib under {det}")

        from .explain import EvidenceEngine

        self.engine = EvidenceEngine.load(root / "reference_stats.json")
        self.faces = None
        if self.cfg.enable_face:
            from .forensics import FaceAnalyzer

            self.faces = FaceAnalyzer()
        self.cam = None
        if self.kind == "torch" and self.cfg.enable_gradcam:
            from .gradcam import GradCAM

            self.cam = GradCAM(self.model, device=self.cfg.device)

        #: second supervised task - only present when the dataset had generator labels
        self.generator = None
        self.generator_error = None
        gen_dir = root / "generator_classifier"
        if self.cfg.enable_generator and (gen_dir / "model_config.json").exists():
            try:
                from .generator import GeneratorClassifier

                self.generator = GeneratorClassifier.load(gen_dir, device=self.cfg.device)
            except Exception as exc:  # noqa: BLE001 - the detector must still serve
                self.generator_error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ semantics
    def verdict_for(self, class_id: int) -> str:
        """Verdict text from the class *name* (never from a hardcoded index)."""
        name = str(self.classes[class_id]) if 0 <= class_id < len(self.classes) else ""
        blob = f"{name} {self.meaning_by_id.get(int(class_id), '')}".lower()
        if any(k in blob for k in ("ai_generated", "synthetic", "generated", "fake", "deepfake", "spoof", "manipul")):
            return "AI-GENERATED / SYNTHETIC"
        if any(k in blob for k in ("authentic", "camera-captured", "camera captured", "real", "genuine", "bona")):
            return "AUTHENTIC CAPTURE"
        return VERDICT_LABEL.get(int(class_id), (name or "UNKNOWN").upper())

    # ------------------------------------------------------------------- loaders
    def _load_torch(self, path: Path) -> None:
        import torch

        from .models import ModelSpec, build_model
        from .transforms import ImageTransform

        self.weights_sha256 = _sha256(path)
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.ckpt_meta = {k: v for k, v in ck.items() if k != "state_dict"}
        spec = ModelSpec.from_dict(ck["spec"])
        model = build_model(spec)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.cfg.device)
        self.spec = spec
        pre = ck.get("preprocess", {}) or {}
        self.transform = ImageTransform(
            size=int(pre.get("size", spec.input_size)),
            mean=tuple(pre.get("mean", self.preprocessing.get("mean", (0.5, 0.5, 0.5)))),
            std=tuple(pre.get("std", self.preprocessing.get("std", (0.5, 0.5, 0.5)))),
            train=False,
            to_gray=bool(pre.get("to_gray", spec.in_channels == 1)),
        )
        self.threshold = float(ck.get("threshold", self.threshold))
        self.input_size = int(pre.get("size", spec.input_size))
        self.arch = spec.name

    def _load_classical(self, path: Path) -> None:
        import joblib

        from .features import FeatureOptions

        payload = joblib.load(path)
        self.sklearn_payload = payload
        self.pipeline = payload["pipeline"]
        self.feature_kind = payload.get("feature", "forensic")
        self.threshold = float(payload.get("threshold", self.threshold))
        self.feature_size = int((payload.get("preprocess") or {}).get("resize_to", 64))
        # robustness: derive the grid from the model itself if the metadata is older/inconsistent
        try:
            nf = int(self.pipeline.named_steps["scale"].n_features_in_)
        except Exception:  # noqa: BLE001
            try:
                nf = int(self.pipeline.n_features_in_)
            except Exception:  # noqa: BLE001
                nf = 0
        if nf and payload.get("feature") == "pixels":
            side = int(round(math.sqrt(nf / 3)))
            if side and side * side * 3 != nf:
                side = 16
            self.feature_size = side
            if side != int((payload.get("preprocess") or {}).get("resize_to", side)):
                print(f"[predictor] classical pixel model expects {nf} features -> using {side}px grid")
        self.feature_photometric = (payload.get("preprocess") or {}).get("photometric", "gray")
        self.feature_opts = FeatureOptions(photometric=self.feature_photometric)
        self.arch = f"{payload.get('feature','forensic')}-descriptor"
        self.input_size = self.feature_size

    # ---------------------------------------------------------------- preprocessing
    def decode(self, data: bytes) -> Tuple[np.ndarray, Dict[str, object]]:
        img = Image.open(io.BytesIO(data))
        img.load()
        fmt, mode, size0 = img.format, img.mode, img.size
        img = ImageOps.exif_transpose(img)
        warnings: List[str] = []
        if img.width * img.height > MAX_PIXELS:
            raise ValueError(f"image too large ({img.width}x{img.height}); limit {MAX_PIXELS} pixels")
        if max(img.size) > self.cfg.max_input_px:
            scale = self.cfg.max_input_px / max(img.size)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
            warnings.append(f"downscaled to {img.width}x{img.height} for analysis")
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
            warnings.append(f"converted from {mode} to RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
            warnings.append("grayscale input: replicated to 3 channels")
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr, {
            "original_size": [int(size0[0]), int(size0[1])],
            "working_size": [int(arr.shape[1]), int(arr.shape[0])],
            "source_format": fmt or "unknown",
            "source_mode": mode,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ scoring
    def score(self, arr: np.ndarray) -> float:
        """Positive-class probability from the trained model (the only source of the verdict)."""
        if self.kind == "torch":
            import torch

            x = self.transform(arr)
            with torch.no_grad():
                out = self.model(x.to(self.cfg.device))
            z = out.detach().cpu().numpy()[0]
            z = float(z[0]) if np.ndim(z) else float(z)
            return float(1.0 / (1.0 + np.exp(-z)))
        from .transforms import resize_any, to_chw_float01
        from .trainer import _predict

        # identical geometry to training: uint8 grid for descriptors, float [0,1] for pixels
        small = resize_any(to_chw_float01(np.array(arr, copy=True)[None]), self.feature_size)[0].permute(1, 2, 0).mul(255).byte().numpy()
        if self.feature_kind == "forensic":
            from .features import extract_features

            X, _ = extract_features(np.ascontiguousarray(small[None]), self.feature_opts)
        else:
            X = small.astype(np.float32).reshape(1, -1) / np.float32(255.0)
        p = _predict(self.pipeline, X)[0]
        pos = self.label_map.get(self.positive_class, 1)
        return float(p[pos]) if len(p) > pos else float(p[-1])

    # ------------------------------------------------------------------ predict
    def predict_bytes(self, data: bytes) -> Dict[str, object]:
        arr, meta = self.decode(data)
        return self.predict_array(arr, meta)

    def predict_path(self, path: Path) -> Dict[str, object]:
        return self.predict_bytes(Path(path).read_bytes())

    def predict_array(self, arr: np.ndarray, meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        cap = int(self.cfg.max_input_px)
        orig = [int(arr.shape[1]), int(arr.shape[0])]
        if max(orig) > cap:                     # same cap as decode(): no path feeds an uncapped image
            scale = cap / max(orig)
            img = Image.fromarray(np.ascontiguousarray(arr)).resize(
                (max(1, int(orig[0] * scale)), max(1, int(orig[1] * scale))), Image.LANCZOS
            )
            arr = np.asarray(img, dtype=np.uint8)
        meta = dict(meta or {"original_size": orig, "working_size": [int(arr.shape[1]), int(arr.shape[0])],
                             "source_format": "array", "source_mode": "RGB", "warnings": []})
        t0 = time.perf_counter()
        prob = self.score(arr)
        pred = self.pos_index if prob >= self.threshold else self.neg_index
        out: Dict[str, object] = {
            "probability_synthetic": round(prob, 6),
            "predicted_class_id": pred,
            "predicted_class": self.classes[pred],
            "verdict": self.verdict_for(pred),
            # confidence is the probability of the *predicted* class. max(p, 1-p) would be
            # wrong here: the operating threshold is fitted (not 0.5), so the side of 0.5 that
            # wins need not be the side of the threshold that wins.
            "confidence": round(prob if pred == self.pos_index else 1.0 - prob, 6),
            "margin_over_threshold": round(abs(prob - self.threshold), 6),
            "threshold": round(self.threshold, 4),
            "model": {
                "name": self.model_info.get("model_name", f"ams-ai-detector-{self.arch}"),
                "architecture": self.arch,
                "family": self.kind,
                "input_size": self.input_size,
                "model_size_mb": self.model_info.get("model_size_mb"),
                "trained_at": self.model_info.get("training_date"),
                "pretrained_backbone": self.model_info.get("pretrained_backbone"),
                "test_f1": (self.model_info.get("metrics") or {}).get("f1"),
                "test_roc_auc": (self.model_info.get("metrics") or {}).get("roc_auc"),
                "classes": self.classes,
            },
            "timing_ms": round((time.perf_counter() - t0) * 1000, 1),
            **meta,
        }
        # reliability caveats that are measurable from the input itself
        warns = list(out.get("warnings") or [])
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if max(out.get("original_size") or [0, 0]) > cap:
            warns.append(f"input {out['original_size'][0]}x{out['original_size'][1]} exceeded the {cap}px "
                         f"serving cap and was downscaled to {w}x{h} before inference")
        if min(h, w) < 24:
            warns.append(f"input is only {w}x{h}px - below the {self.input_size}px the model was trained on; "
                         f"detail the network relies on may not survive the resize")
        elif min(h, w) < self.input_size:
            warns.append(f"input was upsampled from {w}x{h} to the {self.input_size}px model grid; "
                         f"interpolated detail is not real detail")
        out["warnings"] = warns
        # low-margin flag: the model is not sure, say so instead of hiding it
        if abs(prob - self.threshold) < 0.08:
            out.setdefault("warnings", []).append(
                f"close call: probability {prob:.3f} is within 0.08 of the decision threshold {self.threshold:.3f}"
            )

        # evidence engine
        if self.engine is not None:
            try:
                from .transforms import resize_any, to_chw_float01

                g = resize_any(to_chw_float01(np.array(arr, copy=True)[None]), self.input_size)[0].permute(1, 2, 0).mul(255).byte().numpy()
                ev = self.engine.evaluate(np.ascontiguousarray(g), {"probability": prob, "verdict": out["verdict"],
                                                                    "confidence": out["confidence"], "threshold": self.threshold,
                                                                    "model": out["model"]["name"], "family": self.kind,
                                                                    "input_size": self.input_size})
                out["explanation"] = ev
            except Exception as exc:  # noqa: BLE001
                out["explanation"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            out["explanation"] = {
                "unavailable": "reference statistics missing - run `python -m ams.cli reference` once, then restart the app"
            }

        # saliency (only when the loaded model is convolutional)
        if self.cam is not None:
            try:

                x = self.transform(arr)
                cam = self.cam(x)
                payload = cam.to_dict()
                payload.update({"model_input_size": [int(x.shape[-1]), int(x.shape[-2])],
                                "weights_sha256": getattr(self, "weights_sha256", None),
                                "computed_from": "gradients of the loaded checkpoint (no retraining)"})
                if cam.available:
                    payload["overlay_png_b64"] = _png_b64(cam.overlay)
                    payload["input_png_b64"] = _png_b64(cam.rgb)
                    payload["upscaled"] = _upscale_note(arr.shape, x.shape)
                out["saliency"] = payload
            except Exception as exc:  # noqa: BLE001
                out["saliency"] = {"available": False, "reason": f"grad-cam failed: {type(exc).__name__}: {exc}"}
        else:
            out["saliency"] = {
                "available": False,
                "reason": "the selected production model is a classical descriptor classifier: there is no "
                          "convolutional activation to attribute, so no heatmap is shown",
            }

        # analytical face module
        if self.faces is not None:
            try:
                work = _resize_max(arr, FACE_WORK_SIZE)
                out["face_analysis"] = self.faces.analyse(work)
            except Exception as exc:  # noqa: BLE001
                out["face_analysis"] = {"error": f"{type(exc).__name__}: {exc}", "n_faces": 0}
        out["honesty"] = self.model_info.get(
            "honesty",
            [
                "Prediction is probabilistic and is not proof of image origin.",
                "No detector is 100% accurate and this one makes no such claim.",
                "Performance may decrease on unseen generators.",
                "Image editing, compression, screenshots and resizing can affect predictions.",
            ],
        )
        out["generator_identification"] = self.generator_identification(arr)
        return out

    def generator_identification(self, arr: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Name the generator only when the second model exists *and* clears its acceptance rule."""
        record = (self.metrics_doc.get("generator_classifier", {}) or {})
        if self.generator is None:
            return {
                "available": False,
                "generator": None,
                "reason": self.generator_error
                or record.get("reason")
                or "no generator fingerprinting model was trained (the dataset carries no per-generator labels)",
                "dataset_labels": record.get("classes"),
                "known_generators": [],
            }
        classes = list(self.generator.classes)
        base: Dict[str, object] = {
            "available": True,
            "known_generators": classes,
            "model": self.generator.cfg.get("id"),
            "trained_on_images": (record.get("task") or {}).get("per_class_counts"),
            "measured_metrics": (record.get("task") or {}).get("metrics"),
            "limitation": (
                "the model can only name the generators above; an image from any other generator is "
                "reported as UNKNOWN instead of being forced into the closest known class"
            ),
        }
        if arr is None:
            base["generator"] = None
            base["reason"] = "no image supplied"
            return base
        try:
            out = dict(self.generator.predict(np.ascontiguousarray(arr)))
        except Exception as exc:  # noqa: BLE001
            base["generator"] = None
            base["reason"] = f"generator model failed: {type(exc).__name__}: {exc}"
            return base
        out.update(base)
        return out


def _png_b64(arr: Optional[np.ndarray]) -> Optional[str]:
    if arr is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _resize_max(arr: np.ndarray, cap: int) -> np.ndarray:
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= cap:
        return arr
    s = cap / m
    return np.asarray(Image.fromarray(arr).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR), dtype=np.uint8)


def _upscale_note(orig_shape, model_shape) -> str:
    return (
        f"the heatmap is computed on the {model_shape[-1]}x{model_shape[-2]} model input grid and shown "
        f"there; the detector sees a {orig_shape[1]}x{orig_shape[0]} image resized to that grid"
    )


def load_detector(models_dir: Path = Path("models")) -> Detector:
    return Detector(PredictorConfig(models_dir=Path(models_dir)))
