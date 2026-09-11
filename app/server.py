"""FastAPI service for the AMS image-authenticity detector (CPU-only, free-deploy).

Design constraints from the project spec:

* the service **loads** the trained artifact from ``models/`` - it never retrains;
* one model instance is cached at startup and reused for every request;
* uploads stay in memory (no temp files to clean), size/type validated, images capped;
* every number shown in the UI comes from the artifact or from measured image signals.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("AMS_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("AMS_REPORTS_DIR", ROOT / "reports"))
SAMPLES_DIR = Path(os.environ.get("AMS_SAMPLES_DIR", ROOT / "samples"))
MAX_UPLOAD_BYTES = int(os.environ.get("AMS_MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

app = FastAPI(
    title="AMS AI Image Detector",
    version="0.1.0",
    description="Supervised REAL vs AI-GENERATED detection with Grad-CAM saliency and an analytical face module.",
)

_detector = None
_load_error: Optional[str] = None
_startup = time.time()


def get_detector():
    global _detector, _load_error
    if _detector is None and _load_error is None:
        try:
            import sys

            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from ams.predictor import Detector, PredictorConfig

            _detector = Detector(PredictorConfig(models_dir=MODELS_DIR))
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health and the landing page
            _load_error = f"{type(exc).__name__}: {exc}"
    return _detector


@app.on_event("startup")
def _warm() -> None:  # pragma: no cover - startup hook
    det = get_detector()
    if det is not None:
        print(f"[ams] model loaded: {det.arch} ({det.kind}), threshold={det.threshold:.3f}, "
              f"input={det.input_size}px in {time.time()-_startup:.1f}s")
    else:
        print(f"[ams] model NOT loaded: {_load_error}")


# --------------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(page.read_text())


@app.get("/api/health")
def health() -> Dict[str, object]:
    det = get_detector()
    return {
        "status": "ok" if det else "model_unavailable",
        "model_loaded": bool(det),
        "load_error": _load_error,
        "uptime_s": round(time.time() - _startup, 1),
        "device": "cpu",
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "models_dir": str(MODELS_DIR),
        "reports_available": sorted(p.name for p in REPORTS_DIR.glob("*.md")) if REPORTS_DIR.exists() else [],
    }


@app.get("/api/model")
def model_info() -> Dict[str, object]:
    info = _read_json(MODELS_DIR / "model_info.json")
    metrics = _read_json(MODELS_DIR / "metrics.json")
    classes = _read_json(MODELS_DIR / "ai_detector" / "classes.json")
    if not info:
        raise HTTPException(503, detail="no trained model in models/ - run `python train.py` once")
    sel = (metrics or {}).get("selection", {})
    task = ((metrics or {}).get("tasks", {}) or {}).get("ai_detector", {})
    return {
        "model": info,
        "classes": classes,
        "selection": sel,
        "reported_metrics": task.get("metrics"),
        "threshold": task.get("threshold"),
        "split": (metrics or {}).get("split"),
        "dataset": (metrics or {}).get("dataset"),
        "generator_classifier": (metrics or {}).get("generator_classifier"),
        "face_model": (metrics or {}).get("face_model"),
        "comparison": (metrics or {}).get("comparison", []),
        "honesty": info.get("honesty", []),
        "limitations": info.get("known_limitations", []),
    }


@app.get("/api/metrics")
def metrics() -> Dict[str, object]:
    m = _read_json(MODELS_DIR / "metrics.json")
    if not m:
        raise HTTPException(503, detail="models/metrics.json missing - run `python train.py`")
    return m


@app.get("/api/samples")
def samples() -> Dict[str, object]:
    man = SAMPLES_DIR / "manifest.json"
    if not man.exists():
        return {"samples": [], "note": "run `python train.py` (export_samples_n) to create samples/"}
    items = json.loads(man.read_text())
    return {"samples": items, "note": "images come from the held-out test split of the training corpus"}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    safe = Path(name).name
    p = SAMPLES_DIR / safe
    if not p.exists() or p.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(404, "sample not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/report/{name}")
def report(name: str):
    safe = Path(name).name                      # strips any traversal; only the basename is used
    stem = Path(safe)
    if stem.suffix.lower() not in {".md", ".png", ".json"}:
        stem = stem.with_suffix(".md")
    p = REPORTS_DIR / stem.name
    if not p.exists():
        p = REPORTS_DIR / "figures" / stem.name
    if not p.exists():
        raise HTTPException(404, "report not found")
    if p.suffix == ".md":
        return {"name": p.name, "markdown": p.read_text()}
    return FileResponse(p)


# ------------------------------------------------------------------------ predict
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    det = get_detector()
    if det is None:
        raise HTTPException(503, detail=f"model unavailable: {_load_error}")
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(415, detail=f"unsupported content type {file.content_type!r}")
    data = await file.read()
    if not data:
        raise HTTPException(400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
    try:
        result = det.predict_bytes(data)
    except Exception as exc:  # noqa: BLE001 - decode failures are user errors, not server faults
        raise HTTPException(422, detail=f"could not read image: {type(exc).__name__}: {exc}")
    result["filename"] = file.filename
    result["upload_bytes"] = len(data)
    return JSONResponse(json.loads(json.dumps(result, default=str)))


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


# static assets last so /api/* keeps priority
STATIC = Path(__file__).resolve().parent / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
