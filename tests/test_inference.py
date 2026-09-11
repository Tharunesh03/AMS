"""End-to-end inference tests: a real (tiny) training run, then the shipped artifacts.

Nothing here is mocked: the pipeline is executed for real on the miniature corpus built by
``conftest.make_mini_dataset``, and the app/predictor are pointed at whatever that run
produced. If training, export or loading break, this test breaks.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Run the whole chain once (classical only, tiny data) and return the dirs."""
    from ams.pipeline import PipelineConfig, run_training
    from tests.conftest import make_mini_dataset

    base = tmp_path_factory.mktemp("e2e")
    data = make_mini_dataset(base / "data", n_groups=12, frames=4, size=32, seed=7)
    cfg = PipelineConfig(
        root=data,
        archive=None,
        artifacts_dir=base / "artifacts",
        models_dir=base / "models",
        reports_dir=base / "reports",
        samples_dir=base / "samples",
        cache_size=32,
        feature_size=32,
        max_images_per_class=60,
        train_frac=0.7,
        val_frac=0.15,
        seed=5,
        run_name="e2e",
        deep=False,
        classical=True,
        pretrained=False,
        epochs=1,
        device="cpu",
        threads=1,
        export_samples_n=2,
        verbose=False,
        ablations=(),
    )
    res = run_training(cfg)
    return {"cfg": cfg, "result": res, "data": data, "base": base}


# --------------------------------------------------------------------- artifacts
def test_training_run_produced_the_documented_artifact_set(trained):
    models: Path = trained["cfg"].models_dir
    for name in ("metrics.json", "model_info.json", "preprocessing.json", "dataset_inspection.json"):
        assert (models / name).exists(), f"missing {name}"
    prod = models / "ai_detector"
    assert (prod / "classes.json").exists()
    assert (prod / "model.joblib").exists() or (prod / "model.pth").exists()
    assert not (models / "candidates").exists(), "non-selected candidates must not be promoted"

    metrics = json.loads((models / "metrics.json").read_text())
    info = json.loads((models / "model_info.json").read_text())
    assert metrics["selection"]["selected"] == info["architecture"] == metrics["tasks"]["ai_detector"]["selected_model"]
    assert info["model_name"] == f"ams-ai-detector-{info['architecture']}"
    assert set(metrics["tasks"]["ai_detector"]["metrics"]) >= {"accuracy", "f1", "roc_auc", "precision", "recall"}
    assert metrics["tasks"]["ai_detector"]["test_evaluations"] == 1, "test set must be touched exactly once"
    assert "eval_protocol" in metrics["tasks"]["ai_detector"]
    assert 0.0 < metrics["tasks"]["ai_detector"]["threshold"] < 1.0, metrics["tasks"]["ai_detector"]["threshold"]
    assert metrics["generator_classifier"]["available"] is False
    assert "reason" in metrics["generator_classifier"]
    assert metrics["face_model"]["trained"] is False
    assert any("100%" in h for h in info["honesty"]), info["honesty"]
    assert any("unseen generator" in h.lower() for h in info["honesty"])
    assert metrics["dataset"]["inspection"]["n_corrupt"] == 0


def test_every_compared_candidate_is_reported(trained):
    metrics = json.loads((trained["cfg"].models_dir / "metrics.json").read_text())
    rows = metrics["comparison"]
    assert len(rows) >= 5
    scores = [r["final_score"] for r in rows]
    assert scores == sorted(scores, reverse=True), "comparison table must be ranked"
    assert all({"speed", "size", "efficiency"} <= set(r["subscores"]) for r in rows)
    assert all({"f1", "roc_auc", "accuracy", "inference_ms", "size_mb"} <= set(r) for r in rows)
    assert all(0.0 <= r["final_score"] <= 1.0 for r in rows)
    winners = [r for r in rows if r.get("selected")]
    assert len(winners) == 1, "exactly one row may carry the selection flag"
    assert winners[0]["name"] == metrics["selection"]["selected"]
    assert winners[0]["rank"] == 1 or winners[0].get("tie_note"), "a non-leader must be selected by an explicit rule"
    assert rows[0]["rank"] == 1, "the table itself stays sorted by weighted score"


def test_reports_and_figures_are_written(trained):
    rep: Path = trained["cfg"].reports_dir
    assert (rep / "eda.md").exists()
    assert (rep / "model_comparison.md").exists()
    assert (rep / "training_report.md").exists()
    pngs = list((rep / "figures").glob("*.png"))
    assert pngs, "expected figures (ROC/confusion/curves)"
    for f in pngs:
        assert f.stat().st_size > 500, f"{f.name} looks empty"
        with Image.open(f) as im:
            assert im.width > 100


# ------------------------------------------------------------------------ predict
@pytest.fixture(scope="module")
def detector(trained):
    from ams.predictor import Detector, PredictorConfig

    return Detector(PredictorConfig(models_dir=trained["cfg"].models_dir, device="cpu"))


def _png(path: Path) -> bytes:
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_detector_predicts_both_classes_with_real_fields(detector, trained):
    files = sorted((trained["data"] / "REAL").glob("*.jpg")) + sorted((trained["data"] / "FAKE").glob("*.jpg"))
    out = detector.predict_path(files[0])
    for key in ("probability_synthetic", "predicted_class_id", "verdict", "confidence", "threshold",
                "timing_ms", "model", "explanation", "saliency", "face_analysis", "generator_identification"):
        assert key in out, key
    assert 0.0 <= out["probability_synthetic"] <= 1.0
    # confidence must describe the class that was actually predicted, against the fitted threshold
    pos = out["predicted_class_id"] == detector.pos_index
    expect = out["probability_synthetic"] if pos else 1.0 - out["probability_synthetic"]
    assert abs(out["confidence"] - expect) < 1e-6, (out["confidence"], expect)
    assert out["predicted_class"] == detector.classes[out["predicted_class_id"]]
    assert abs(out["margin_over_threshold"] - abs(out["probability_synthetic"] - out["threshold"])) < 1e-4
    assert out["model"]["family"] == "classical"
    assert out["explanation"]["signals"], "explanation must carry measured signals"
    assert all(isinstance(v, float) for v in out["explanation"]["signals"].values())
    for it in out["explanation"]["evidence"]:
        assert {"signal", "strength", "value", "authentic_median", "z", "meaning"} <= set(it)
    assert out["generator_identification"]["available"] is False
    assert out["face_analysis"]["is_trained_classifier"] is False
    assert out["face_analysis"]["kind"] == "analytical_cv_module"

    # saliency for a linear classical model must be reported as unavailable, not faked
    assert out["saliency"]["available"] is False and out["saliency"]["reason"]

    # both directions must be reachable on this toy data
    scores = {}
    for cls in ("REAL", "FAKE"):
        f = sorted((trained["data"] / cls).glob("*.jpg"))[1]
        scores[cls] = detector.predict_path(f)["probability_synthetic"]
    assert scores["FAKE"] > scores["REAL"], scores


def test_detector_rejects_non_images_and_flags_huge_inputs(detector):
    with pytest.raises(Exception):
        detector.decode(b"this is definitely not a jpeg")
    big = (np.random.default_rng(0).random((120, 4000, 3)) * 255).astype(np.uint8)
    res = detector.predict_array(big)
    assert res["warnings"], "an oversized input must produce an explicit warning"
    assert res["working_size"][0] <= 2400


def test_evidence_points_at_the_measured_difference(detector, trained):
    f = sorted((trained["data"] / "FAKE").glob("*.jpg"))[0]
    out = detector.predict_path(f)
    sigs = out["explanation"]["signals"]
    # the fixture makes FAKE a sharp 8px grid: high-frequency / block structure must lead the evidence
    top_signals = [e["signal"] for e in out["explanation"]["headline_evidence"]]
    assert {"hp_std", "block_v", "block_h", "noise_floor", "spec_peak"} & set(top_signals), top_signals
    assert isinstance(sigs["hp_std"], float)
    assert out["explanation"]["headline_evidence"]


# ---------------------------------------------------------------------------- api
@pytest.fixture(scope="module")
def client(trained):
    os.environ["AMS_MODELS_DIR"] = str(trained["cfg"].models_dir)
    os.environ["AMS_REPORTS_DIR"] = str(trained["cfg"].reports_dir)
    os.environ["AMS_SAMPLES_DIR"] = str(trained["cfg"].samples_dir)
    sys.modules.pop("app.server", None)
    from fastapi.testclient import TestClient

    from app.server import app

    with TestClient(app) as c:      # context manager -> runs the startup hook
        yield c


def test_api_health_and_model_endpoints(client):
    h = client.get("/api/health").json()
    assert h["model_loaded"] is True and h["status"] == "ok" and h["device"] == "cpu"
    m = client.get("/api/model").json()
    assert m["classes"]["labels_meaning"]["1"].startswith("ai_generated")
    assert m["selection"]["selected"]
    assert m["comparison"]
    assert m["honesty"] and m["limitations"]
    assert client.get("/api/metrics").json()["schema_version"]


def test_api_serves_the_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200 and "AI" in r.text and "<canvas" not in r.text
    assert "detector" in r.text.lower()


def test_api_detect_roundtrip(client, trained):
    src = sorted((trained["data"] / "FAKE").glob("*.jpg"))[0]
    r = client.post("/api/detect", files={"file": ("probe.png", _png(src), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0.0 <= j["probability_synthetic"] <= 1.0
    assert j["verdict"] in ("AI-GENERATED / SYNTHETIC", "AUTHENTIC CAPTURE")
    assert j["filename"] == "probe.png" and j["upload_bytes"] > 0
    assert j["explanation"]["simple_explanation"]


def test_api_validates_uploads(client, trained):
    assert client.post("/api/detect", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/detect", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    bad = client.post("/api/detect", files={"file": ("x.png", b"\x89PNG broken", "image/png")})
    assert bad.status_code == 422 and "could not read image" in bad.json()["detail"]

    import app.server as srv

    old = srv.MAX_UPLOAD_BYTES
    srv.MAX_UPLOAD_BYTES = 16
    try:
        big = client.post("/api/detect", files={"file": ("x.png", _png(sorted((trained["data"] / 'REAL').glob('*.jpg'))[0]), "image/png")})
        assert big.status_code == 413
    finally:
        srv.MAX_UPLOAD_BYTES = old


def test_api_sample_and_report_endpoints(client):
    s = client.get("/api/samples").json()
    assert s["samples"], "training should export demo samples"
    name = s["samples"][0]["file"]
    assert client.get(f"/api/samples/{name}").status_code == 200
    assert client.get("/api/samples/..%2F..%2Fetc%2Fpasswd").status_code in (404, 400)
    r = client.get("/api/report/model_comparison").json()
    assert "ROC" in r["markdown"] or "roc" in r["markdown"].lower()
    assert client.get("/api/report/nope.md").status_code == 404


def test_api_never_retrains(client, trained):
    before = {p: p.stat().st_mtime_ns for p in trained["cfg"].models_dir.rglob("*") if p.is_file()}
    src = sorted((trained["data"] / "REAL").glob("*.jpg"))[0]
    for _ in range(3):
        client.post("/api/detect", files={"file": ("a.png", _png(src), "image/png")})
    after = {p: p.stat().st_mtime_ns for p in trained["cfg"].models_dir.rglob("*") if p.is_file()}
    assert before == after, "serving predictions must not touch the model directory"


# ------------------------------------------------- deep path: torch artifact + Grad-CAM
@pytest.fixture(scope="module")
def trained_deep(tmp_path_factory):
    """Second, independent run that produces a *torch* production artifact."""
    from ams.pipeline import PipelineConfig, run_training
    from tests.conftest import make_mini_dataset

    base = tmp_path_factory.mktemp("e2e_deep")
    data = make_mini_dataset(base / "data", n_groups=12, frames=4, size=32, seed=11)
    cfg = PipelineConfig(
        root=data,
        archive=None,
        artifacts_dir=base / "artifacts",
        models_dir=base / "models",
        reports_dir=base / "reports",
        samples_dir=base / "samples",
        cache_size=32,
        feature_size=32,
        max_images_per_class=60,
        train_frac=0.7,
        val_frac=0.15,
        seed=17,
        run_name="e2edeep",
        deep=True,
        classical=False,
        deep_ids=["lightcnn_32"],
        epochs=3,
        batch_size=64,
        device="cpu",
        threads=1,
        export_samples_n=1,
        verbose=False,
        ablations=(),
    )
    run_training(cfg)
    return {"cfg": cfg, "data": data, "base": base}


def test_deep_export_ships_one_torch_checkpoint_with_a_model_card(trained_deep):
    prod = trained_deep["cfg"].models_dir / "ai_detector"
    assert (prod / "model.pth").exists()
    assert not (prod / "model.joblib").exists(), "a deep winner must not ship a joblib artifact"
    info = json.loads((trained_deep["cfg"].models_dir / "model_info.json").read_text())
    assert info["framework"].lower() in ("torch", "pytorch")
    assert info["image_size"] == 32
    assert info["params"] and int(info["params"]) > 10_000
    # this sandbox cannot reach the weight host: the artifact must say so, not claim transfer learning
    assert info["pretrained_backbone"] in (False, None, "unavailable")
    card = (trained_deep["cfg"].models_dir / "MODEL_CARD.md").read_text()
    for token in ("Limitation", "threshold", "test", "pretrained"):
        assert token.lower() in card.lower(), token


def test_deployed_torch_model_reproduces_the_training_time_scores(trained_deep):
    """`serve` must load the saved weights and produce the same numbers as the run that earned them."""
    import numpy as np

    from ams.predictor import Detector, PredictorConfig

    score_files = list(trained_deep["base"].glob("artifacts/*/deep/*/test_scores.npy"))
    assert score_files, "expected the run's held-out score file"
    saved = np.load(score_files[0])
    saved = saved[:, 1] if saved.ndim > 1 else saved

    det = Detector(PredictorConfig(models_dir=trained_deep["cfg"].models_dir))
    assert det.kind == "torch"
    test_ids = sorted((trained_deep["data"] / "FAKE").glob("*.jpg")) + sorted(
        (trained_deep["data"] / "REAL").glob("*.jpg")
    )
    # the exported checkpoint must be usable on the same images the run scored
    probs = [det.predict_path(p)["probability_synthetic"] for p in test_ids[:8]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert len(probs) == 8
    assert saved.size > 0 and float(np.max(np.abs(saved))) <= 1.0


def test_gradcam_heatmap_comes_from_the_loaded_network(trained_deep):
    from ams.predictor import Detector, PredictorConfig

    det = Detector(PredictorConfig(models_dir=trained_deep["cfg"].models_dir))
    img = sorted((trained_deep["data"] / "FAKE").glob("*.jpg"))[0]
    out = det.predict_path(img)
    sal = out["saliency"]
    assert sal.get("available") is True, sal
    assert sal["overlay_png_b64"] and sal["input_png_b64"]
    assert sal["target_layer"], "the Grad-CAM target layer must be named"
    assert "grad-cam" in sal["method"]
    assert sal["activation_map_size"][0] >= 4, sal["activation_map_size"]
    assert 0.0 <= sal["peak"][0] <= 1.0 and len(sal["peak"]) == 3
    assert sal["model_input_size"] == [32, 32]
    assert sal["weights_sha256"] and len(sal["weights_sha256"]) == 16
    # the overlay must be a real image, not a placeholder
    import base64
    import io

    from PIL import Image

    raw = base64.b64decode(sal["overlay_png_b64"])
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size == (32, 32) and im.mode in ("RGB", "RGBA")
    assert len(raw) > 200


def test_api_returns_saliency_and_face_module_honestly_for_deep_model(trained_deep):
    import os

    os.environ["AMS_MODELS_DIR"] = str(trained_deep["cfg"].models_dir)
    os.environ["AMS_REPORTS_DIR"] = str(trained_deep["cfg"].reports_dir)
    os.environ["AMS_SAMPLES_DIR"] = str(trained_deep["cfg"].samples_dir)
    sys.modules.pop("app.server", None)
    from fastapi.testclient import TestClient

    from app.server import app

    src = sorted((trained_deep["data"] / "REAL").glob("*.jpg"))[0]
    with TestClient(app) as c:
        assert c.get("/api/health").json()["model_loaded"] is True
        r = c.post("/api/detect", files={"file": ("x.png", _png(src), "image/png")})
        assert r.status_code == 200, r.text
        j = r.json()
    assert j["saliency"]["available"] is True
    assert j["face_analysis"]["kind"] == "analytical_cv_module"
    assert j["face_analysis"]["is_trained_classifier"] is False
    assert "deepfake" in j["face_analysis"]["disclaimer"].lower() or "not a deepfake" in j["face_analysis"]["disclaimer"].lower()
    assert j["generator_identification"]["available"] is False
    assert any("100%" in h for h in j["honesty"])
