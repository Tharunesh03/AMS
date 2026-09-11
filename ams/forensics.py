"""Analytical face/consistency module - explicitly NOT a trained deepfake classifier.

The provided corpus has only two labels (authentic / synthetic face crops) and no
manipulated-face dimension, so a second "deepfake face" model would be a rename of the
detector rather than evidence. Instead this module measures classical CV signals:

    face detection (Haar cascades)  ->  landmark geometry (eye/mouth line plausibility)
                                     ->  visual-consistency analysis (symmetry, local
                                         sharpness distribution, boundary continuity,
                                         illumination direction, blockiness)

Every number it returns is computed from the uploaded pixels; nothing is inferred from a
model, nothing about identity is estimated, and the UI must present it as heuristics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    HAVE_CV2 = True
except Exception:  # noqa: BLE001 - the module degrades gracefully without OpenCV
    cv2 = None  # type: ignore
    HAVE_CV2 = False


@dataclass
class FaceMeasurement:
    box: Tuple[int, int, int, int]           # x, y, w, h in the analysed (working) image
    detector_score: float
    detected: bool = True                    # False => the region was assumed, not located
    region: str = "haar_detection"
    geometry: Dict[str, float] = field(default_factory=dict)
    landmarks: Dict[str, object] = field(default_factory=dict)
    consistency: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["box"] = [int(v) for v in self.box]
        d["detector_score"] = round(float(self.detector_score), 3)
        for grp in ("geometry", "consistency"):
            d[grp] = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in d[grp].items()}
        return d


class FaceAnalyzer:
    """Haar-cascade detection + geometric/consistency measurements on the raw pixels."""

    def __init__(self, min_face_px: int = 28, assume_face_crop: bool = True,
                 assume_max_px: int = 192) -> None:
        self.min_face_px = int(min_face_px)
        # This corpus is *already* face crops at 32x32, far below what a Haar detector can
        # locate. For small inputs we therefore also analyse an assumed crop region, clearly
        # labelled as such - it measures consistency, it does not claim to have found a face.
        self.assume_face_crop = bool(assume_face_crop)
        self.assume_max_px = int(assume_max_px)
        self._cascades: Dict[str, object] = {}
        if HAVE_CV2:
            base = cv2.data.haarcascades
            for key, fn in (("face", "haarcascade_frontalface_default.xml"), ("eye", "haarcascade_eye.xml")):
                try:
                    c = cv2.CascadeClassifier(base + fn)
                    if not c.empty():
                        self._cascades[key] = c
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------ detection
    def detect(self, gray: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if not self._cascades:
            return []
        cas = self._cascades["face"]
        scale = max(1.0, math.ceil(gray.shape[1] / 640.0))
        work = gray if scale == 1 else cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale), interpolation=cv2.INTER_AREA)
        faces = cas.detectMultiScale(work, scaleFactor=1.12, minNeighbors=4, minSize=(self.min_face_px, self.min_face_px))
        out = []
        for (x, y, w, h) in faces:
            out.append(((int(x) * scale, int(y) * scale, int(w) * scale, int(h) * scale), 0.0))
        return out

    # ------------------------------------------------------------------- analysis
    def analyse(self, rgb: np.ndarray) -> Dict[str, object]:
        """rgb: uint8 ``[H,W,3]`` image at working resolution."""
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        detections = self.detect(gray)
        faces: List[FaceMeasurement] = []
        for box, score in detections[:8]:
            faces.append(self._measure_one(rgb, gray, box, score))
        if not faces and self.assume_face_crop and min(gray.shape[0], gray.shape[1]) <= self.assume_max_px:
            h, w = gray.shape
            box = (int(0.04 * w), int(0.04 * h), max(8, int(0.92 * w)), max(8, int(0.92 * h)))
            fm = self._measure_one(rgb, gray, box, 0.0)
            fm.detected = False
            fm.region = "assumed_face_crop"
            fm.notes.append("region assumed because the input is already a tight face crop - no face was located")
            faces.append(fm)
        summary = self._summarize(rgb, faces)
        return {
            "kind": "analytical_cv_module",
            "is_trained_classifier": False,
            "disclaimer": "Heuristic image analysis only. Not a deepfake detector, not biometric, no identity is estimated.",
            "engine": "opencv-haar" if HAVE_CV2 and self._cascades else "numpy-fallback",
            "face_detector_available": bool(self._cascades),
            "n_faces": len(faces),
            "faces": [f.to_dict() for f in faces],
            "summary": summary,
        }

    def _measure_one(self, rgb: np.ndarray, gray: np.ndarray, box, score: float) -> FaceMeasurement:
        x, y, w, h = box
        face_g = gray[y : y + h, x : x + w].astype(np.float32)
        fm = FaceMeasurement(box=box, detector_score=score)

        # --- geometry (landmark-lite): eye line inside the upper 2/3, aspect plausibility
        if "eye" in self._cascades and face_g.size:
            eyes = self._cascades["eye"].detectMultiScale(
                cv2.resize(face_g, (max(24, w), max(24, h))).astype(np.uint8),
                scaleFactor=1.1, minNeighbors=6, minSize=(int(max(24, w) * 0.10),) * 2,
            )
            eyes = [e for e in eyes if e[1] < 0.75 * max(24, h)]
            fm.landmarks["eyes_detected"] = int(len(eyes))
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: -e[2])[:2]
                cx = [e[0] + e[2] / 2 for e in eyes]
                cy = [e[1] + e[3] / 2 for e in eyes]
                tilt = math.degrees(math.atan2(cy[1] - cy[0], cx[1] - cx[0]))
                fm.landmarks["eye_distance_ratio"] = round(float(abs(cx[1] - cx[0]) / max(w, 1)), 3)
                fm.landmarks["eye_line_tilt_deg"] = round(float(tilt), 2)
                fm.geometry["eye_spacing_plausible"] = 0.16 <= abs(cx[1] - cx[0]) / max(w, 1) <= 0.46
                fm.geometry["eye_line_levelled"] = abs(tilt) <= 12.0
            else:
                fm.landmarks["eye_line_tilt_deg"] = None
        fm.geometry["aspect_ratio"] = round(float(w / max(h, 1)), 3)
        fm.geometry["aspect_plausible"] = 0.55 <= (w / max(h, 1)) <= 1.85
        fm.geometry["face_area_fraction"] = round(float(w * h) / max(gray.size, 1), 4)
        fm.geometry["inside_frame"] = bool(x >= 0 and y >= 0 and x + w <= gray.shape[1] and y + h <= gray.shape[0])

        # --- visual consistency
        cons: Dict[str, float] = {}
        if face_g.shape[0] >= 8 and face_g.shape[1] >= 8:
            mirror = face_g[:, ::-1]
            cons["mirror_correlation"] = round(float(_corr(face_g, mirror)), 3)
            hp = face_g - _blur2(face_g)
            cons["hp_energy"] = round(float(hp.std() / (face_g.std() + 1e-6)), 3)
            k = 8
            tiles = _tiles(face_g, k)
            lap = np.array([_lap_var(t) for t in tiles])
            cons["sharpness_cv"] = round(float(lap.std() / (lap.mean() + 1e-6)), 3)
            cons["flat_tile_fraction"] = round(float((lap < max(1.0, 0.2 * np.median(lap))).mean()), 3)
            # boundary continuity: gradient energy just inside vs just outside the box
            ring_in = face_g[1:-1, 1:-1]
            pad = 4
            y0, y1 = max(y - pad, 0), min(y + h + pad, gray.shape[0])
            x0, x1 = max(x - pad, 0), min(x + w + pad, gray.shape[1])
            outer = gray[y0:y1, x0:x1].astype(np.float32)
            gx_in = float(np.abs(np.diff(ring_in, axis=1)).mean())
            gx_out = float(np.abs(np.diff(outer, axis=1)).mean())
            cons["boundary_gradient_ratio"] = round(gx_in / (gx_out + 1e-6), 3)
            # illumination direction: brightness slope inside face vs face surroundings
            slope_in = float(face_g[:, -1].mean() - face_g[:, 0].mean())
            surround = _surround_gray(gray, box, pad=6)
            slope_out = float(surround[:, -1].mean() - surround[:, 0].mean()) if surround is not None else 0.0
            cons["illumination_agreement"] = round(float(_sign_agree(slope_in, slope_out)), 3)
            # compression grid on the face crop
            cons["blockiness"] = round(float(_block_ratio(face_g)), 3)
        fm.consistency = cons
        if cons:
            if cons["mirror_correlation"] > 0.97 and cons["hp_energy"] < 0.25:
                fm.notes.append("near-perfect left/right symmetry with very low micro-texture energy")
            if cons["sharpness_cv"] < 0.35:
                fm.notes.append("sharpness is unusually uniform across regions (real captures vary)")
            if cons["boundary_gradient_ratio"] > 2.5:
                fm.notes.append("edge energy jumps at the face boundary (possible compositing)")
            if cons["illumination_agreement"] < 0.0:
                fm.notes.append("face brightness slope opposes the surround lighting direction")
        return fm

    def _summarize(self, rgb: np.ndarray, faces: List[FaceMeasurement]) -> Dict[str, object]:
        out: Dict[str, object] = {"image_size": [int(rgb.shape[1]), int(rgb.shape[0])]}
        # whole-image statistics that do not need a face at all
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        hp = g - _blur2(g)
        out["global_hp_energy"] = round(float(hp.std() / (g.std() + 1e-6)), 3)
        out["global_blockiness"] = round(float(_block_ratio(g)), 3)
        out["global_sharpness_cv"] = round(float(np.std([_lap_var(t) for t in _tiles(g, 8)]) / (np.mean([_lap_var(t) for t in _tiles(g, 8)]) + 1e-6)), 3)
        flags = [f for fm in faces for f in fm.notes]
        out["flag_count"] = len(flags)
        out["flags"] = flags[:6]
        out["geometry_issues"] = int(sum(1 for fm in faces for _, v in fm.geometry.items() if v is False))
        out["face_detector_available"] = bool(self._cascades)
        assumed = any((not fm.detected) for fm in faces)
        out["region_source"] = (
            "haar_detection" if any(fm.detected for fm in faces)
            else "assumed_face_crop" if assumed
            else "none"
        )
        if assumed:
            out["note"] = (
                "the Haar cascade located no face at this resolution, so the consistency numbers "
                "were measured on an assumed face-crop region (this corpus is pre-cropped); "
                "they describe image structure only, not identity or provenance"
            )
        elif not faces:
            out["note"] = (
                "no frontal face detected by the classical cascade at this resolution - "
                "face-level consistency checks were skipped, whole-image checks still ran"
            )
        return out


# ------------------------------------------------------------------- small helpers
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[: n[0], : n[1]], b[: n[0], : n[1]]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _blur2(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="reflect")
    out = np.zeros_like(g)
    for dy in range(3):
        for dx in range(3):
            out += p[dy : dy + g.shape[0], dx : dx + g.shape[1]]
    return out / 9.0


def _tiles(g: np.ndarray, k: int) -> List[np.ndarray]:
    h, w = g.shape
    th, tw = max(1, h // k), max(1, w // k)
    return [g[i * th : (i + 1) * th, j * tw : (j + 1) * tw] for i in range(k) for j in range(k) if g[i * th : (i + 1) * th, j * tw : (j + 1) * tw].size]


def _lap_var(a: np.ndarray) -> float:
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    k = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return float(k.var())


def _surround_gray(g: np.ndarray, box, pad: int = 6) -> Optional[np.ndarray]:
    x, y, w, h = box
    y0, y1 = max(y - pad, 0), min(y + h + pad, g.shape[0])
    x0, x1 = max(x - pad, 0), min(x + w + pad, g.shape[1])
    if y1 - y0 < 6 or x1 - x0 < 6:
        return None
    return g[y0:y1, x0:x1]


def _sign_agree(a: float, b: float) -> float:
    if abs(a) < 1e-3 or abs(b) < 1e-3:
        return 0.0
    return 1.0 if (a > 0) == (b > 0) else -1.0


def _block_ratio(g: np.ndarray, period: int = 8) -> float:
    gx = np.abs(np.diff(g, axis=1))
    if gx.shape[1] < period + 1:
        return 0.0
    on = gx[:, period - 1 :: period].mean() if gx[:, period - 1 :: period].size else 0.0
    off = np.abs(gx).mean()
    return float(on / (off + 1e-6))
