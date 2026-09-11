"""Visual Forensic AI - FastAPI front end for the image-provenance model.

The service **loads** the artifact stored in ``models/``; it never retrains. Design rules:

* one model instance is built at startup and reused for every request (CPU only, small);
* uploads stay in memory: size and content type validated, no temp files to clean;
* every number a page shows comes either from the artifact or from measurements of the
  submitted pixels - the browser never computes a verdict;
* pages are plain static HTML/CSS/JS in ``app/static``, served from a route whitelist.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MODELS_DIR = Path(os.environ.get("VFA_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("VFA_REPORTS_DIR", ROOT / "reports"))
SAMPLES_DIR = Path(os.environ.get("VFA_SAMPLES_DIR", ROOT / "samples"))
MAX_UPLOAD_BYTES = int(os.environ.get("VFA_MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VERSION = "0.1.0"

APP_TITLE = "Visual Forensic AI"
APP_DESCRIPTION = ("Supervised image-provenance analysis: a REAL vs AI-GENERATED classifier with measured "
                   "evidence, Grad-CAM saliency when a CNN is deployed, an analytical face-forensics module, "
                   "and an optional generator-fingerprint task - all CPU-only, all reported honestly.")

#: url -> static file. Only these names are ever resolved, so a request cannot reach anything else.
PAGES: Dict[str, str] = {
    "/": "index.html",
    "/analyze": "analyze.html",
    "/model": "model.html",
    "/metrics": "metrics.html",
    "/reports": "reports.html",
}

app = FastAPI(title=APP_TITLE, version=VERSION, description=APP_DESCRIPTION)

_detector = None
_load_error: Optional[str] = None
_startup = time.time()


def get_detector():
    """Build the predictor once. ``None`` means "no artifact yet", which is a documented state."""
    global _detector, _load_error
    if _detector is None and _load_error is None:
        try:
            import sys

            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from vfa.predictor import Detector, PredictorConfig

            _detector = Detector(PredictorConfig(models_dir=MODELS_DIR))
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health and every page
            _load_error = f"{type(exc).__name__}: {exc}"
    return _detector


@app.on_event("startup")
def _warm() -> None:  # pragma: no cover - startup hook
    det = get_detector()
    if det is not None:
        print(f"[visual-forensic-ai] model loaded: {det.arch} ({det.kind}), threshold={det.threshold:.3f}, "
              f"input={det.input_size}px in {time.time()-_startup:.1f}s")
    else:
        print(f"[visual-forensic-ai] model NOT loaded: {_load_error} - run `python train.py`")


# ------------------------------------------------------------------------------ pages
def _page(name: str) -> FileResponse:
    p = STATIC / name
    if not p.exists():
        raise HTTPException(404, detail=f"{name} is missing from app/static/")
    return FileResponse(p, media_type="text/html")


@app.get("/", response_class=HTMLResponse)
def index():
    """"Overview" - what the tool does, what it measured, where to go next."""
    return _page(PAGES["/"])


for _route, _file in list(PAGES.items()):
    if _route == "/":
        continue

    def _make(_file: str = _file):
        def _get():
            return _page(_file)
        return _get

    app.add_api_route(_route, _make(), methods=["GET"], response_class=HTMLResponse,
                      summary=f"Serve the {_route.strip('/').replace('-', ' ')} page",
                      include_in_schema=True)


# ------------------------------------------------------------------------------ api
@app.get("/api/health")
def health() -> Dict[str, object]:
    det = get_detector()
    return {
        "status": "ok" if det else "model_unavailable",
        "model_loaded": bool(det),
        "load_error": _load_error,
        "how_to_fix": None if det else "run `python train.py` once (or `make train`), then reload",
        "uptime_s": round(time.time() - _startup, 1),
        "device": "cpu",
        "title": APP_TITLE,
        "version": VERSION,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "models_dir": str(MODELS_DIR),
        "pages": sorted(PAGES),
        "reports_available": sorted(p.name for p in REPORTS_DIR.glob("*.md")) if REPORTS_DIR.exists() else [],
        "samples_available": _sample_names(),
        "selected_model": getattr(det, "arch", None),
        "threshold": round(float(det.threshold), 6) if det is not None else None,
    }


@app.get("/api/model")
def model_info() -> Dict[str, object]:
    """Everything the UI shows as documentation, read from the artifacts written by ``train.py``."""
    info = _read_json(MODELS_DIR / "model_info.json")
    metrics = _read_json(MODELS_DIR / "metrics.json")
    classes = _read_json(MODELS_DIR / "ai_detector" / "classes.json")
    if not info:
        raise HTTPException(503, detail="no trained model in models/ - run `python train.py` once")
    sel = (metrics or {}).get("selection", {})
    task = ((metrics or {}).get("tasks", {}) or {}).get("ai_detector", {})
    return {
        "app": {"title": APP_TITLE, "version": VERSION, "device": "cpu"},
        "model": info,
        "classes": classes,
        "preprocessing": _read_json(MODELS_DIR / "preprocessing.json"),
        "selection": sel,
        "reported_metrics": task.get("metrics"),
        "threshold": task.get("threshold"),
        "split": (metrics or {}).get("split"),
        "dataset": (metrics or {}).get("dataset"),
        "reproducibility": (metrics or {}).get("reproducibility"),
        "transfer_learning": (metrics or {}).get("transfer_learning"),
        "generator_classifier": (metrics or {}).get("generator_classifier"),
        "face_model": (metrics or {}).get("face_model"),
        "comparison": (metrics or {}).get("comparison", []),
        "honesty": info.get("honesty", []),
        "limitations": info.get("known_limitations", []),
    }


@app.get("/api/metrics")
def metrics() -> Dict[str, object]:
    """The raw metrics document this deployment was selected from - nothing else is added to it."""
    m = _read_json(MODELS_DIR / "metrics.json")
    if not m:
        raise HTTPException(503, detail="models/metrics.json missing - run `python train.py`")
    return m


@app.get("/api/samples")
def samples() -> Dict[str, object]:
    man = SAMPLES_DIR / "manifest.json"
    if not man.exists():
        return {"samples": [], "note": "no samples/ yet - train.py writes it (export_samples_n)"}
    items = json.loads(man.read_text())
    return {"samples": items, "note": "drawn from the held-out test split: the model never saw these while training"}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    safe = Path(name).name
    p = SAMPLES_DIR / safe
    if not p.exists() or p.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(404, "sample not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/report/{name}")
def report(name: str):
    """Serve one generated report (markdown) or figure.

    Basenames only, whitelisted extensions, and a fixed search order - reports/, reports/figures/,
    then the model card next to the artifact. No user-controlled path is ever joined.
    """
    safe = Path(name).name                      # strips any traversal; only the basename is used
    stem = Path(safe)
    if stem.suffix.lower() not in {".md", ".png", ".json"}:
        stem = stem.with_suffix(".md")
    candidates = [REPORTS_DIR / stem.name, REPORTS_DIR / "figures" / stem.name]
    if stem.name == "MODEL_CARD.md":
        candidates.append(MODELS_DIR / stem.name)
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise HTTPException(404, "report not found")
    if p.suffix == ".md":
        return {"name": p.name, "markdown": p.read_text(), "written_at": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
    return FileResponse(p)


# ---------------------------------------------------------------------------- predict
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    det = get_detector()
    if det is None:
        raise HTTPException(503, detail=f"model unavailable: {_load_error} - run `python train.py` once")
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


# ------------------------------------------------------------------------------ util
def _read_json(path: Path) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _sample_names() -> List[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.glob("*.png"))[:12]


# static assets are mounted last so /api/* and the page routes keep priority
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
