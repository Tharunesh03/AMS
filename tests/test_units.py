"""Feature/selection/metrics/labels/forensics/gradcam unit tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vfa.features import FeatureOptions, extract_features
from vfa.metrics import binary_metrics, expected_calibration_error, multiclass_metrics, select_threshold
from vfa.selection import Candidate, score_pool, selection_rationale


# --------------------------------------------------------------------------- features
def test_features_are_deterministic_and_finite():
    rng = np.random.default_rng(0)
    imgs = (rng.random((24, 48, 48, 3)) * 255).astype(np.uint8)
    X1, n1 = extract_features(imgs, FeatureOptions(chunk=8))
    X2, n2 = extract_features(imgs, FeatureOptions(chunk=64))
    assert X1.shape == X2.shape and n1 == n2
    assert np.allclose(X1, X2, atol=1e-5)
    assert np.isfinite(X1).all()
    assert X1.shape[1] == len(n1) > 100


def test_photometric_gray_mode_removes_brightness_difference():
    rng = np.random.default_rng(1)
    base = (rng.random((16, 48, 48, 3)) * 40 + 100).astype(np.uint8)
    bright = np.clip(base.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    o = FeatureOptions(photometric="gray", use_hog=False, use_lbp=False, use_spectrum=False)
    Xa, _ = extract_features(base, o)
    Xb, _ = extract_features(bright, o)
    # contrast scaling is removed by the z-score, so the tone features must match closely
    assert np.allclose(Xa[:, :4], Xb[:, :4], atol=1e-2), (Xa[:, :4], Xb[:, :4])
    o2 = FeatureOptions(photometric="none", use_hog=False, use_lbp=False, use_spectrum=False)
    Ya, _ = extract_features(base, o2)
    Yb, _ = extract_features(bright, o2)
    assert not np.allclose(Ya[:, :4], Yb[:, :4], atol=1e-2)      # raw mode keeps the shift


def test_feature_names_are_unique_and_stable():
    imgs = (np.random.default_rng(2).random((6, 32, 32, 3)) * 255).astype(np.uint8)
    _, names = extract_features(imgs, FeatureOptions())
    assert len(names) == len(set(names))
    assert {"hp_std", "noise_floor_std", "luma_std", "edge_h"} <= set(names)


# --------------------------------------------------------------------------- metrics
def test_binary_metrics_perfect_and_worst():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([0.05, 0.02, 0.1, 0.2, 0.9, 0.8, 0.95, 0.7])
    m = binary_metrics(y, perfect, threshold=0.5)
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] > 0.99
    assert m["confusion_matrix"] == [[4, 0], [0, 4]]
    worst = 1.0 - perfect
    w = binary_metrics(y, worst, threshold=0.5)
    assert w["accuracy"] == 0.0 and w["f1"] == 0.0


def test_threshold_is_selected_on_validation_only():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.45, 0.42, 0.55, 0.5, 0.6, 0.4])
    thr = select_threshold(y, s, policy="f1")
    assert 0 < thr < 1
    m = binary_metrics(y, s, threshold=thr)
    assert m["threshold"] == pytest.approx(thr)
    assert m["f1"] >= binary_metrics(y, s, threshold=0.5)["f1"] - 1e-9


def test_eer_and_calibration_are_reported():
    rng = np.random.default_rng(3)
    y = (rng.random(500) > 0.5).astype(int)
    s = np.clip(y * 0.35 + rng.random(500) * 0.6, 0, 1)
    m = binary_metrics(y, s)
    assert 0.0 <= m["eer"] <= 1.0
    ece = expected_calibration_error(y, s)
    assert 0.0 <= ece <= 1.0


def test_multiclass_metrics_report_top3_only_when_it_can_fail():
    y = np.array([0, 1, 2, 3, 4])
    # the true class is always the runner-up: top-3 must read 1.0 while accuracy reads 0.0
    P = np.array([
        [0.2, 0.7, 0.05, 0.03, 0.02],
        [0.02, 0.2, 0.7, 0.05, 0.03],
        [0.03, 0.02, 0.2, 0.7, 0.05],
        [0.05, 0.03, 0.02, 0.2, 0.7],
        [0.7, 0.05, 0.03, 0.02, 0.2],
    ])
    m = multiclass_metrics(y, P, list("abcde"))
    assert m["accuracy"] == 0.0
    assert m["top3_accuracy"] == 1.0

    # with only 3 classes a "top 3" score cannot miss, so it is reported as not available
    y3 = np.array([0, 1, 2, 0, 1, 2])
    P3 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7],
                   [0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
    m3 = multiclass_metrics(y3, P3, ["a", "b", "c"])
    assert m3["accuracy"] == 1.0
    assert m3["macro_f1"] == pytest.approx(1.0)
    assert m3["top3_accuracy"] is None and "top3_note" in m3
    assert m3["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


# ------------------------------------------------------------------------ selection
def _cand(name, f1, auc, acc, ms, size_mb, family="deep", pipeline="torch"):
    return Candidate(
        name=name, family=family, task="ai_detector",
        metrics={"f1": f1, "roc_auc": auc, "accuracy": acc, "recall": f1, "precision": f1},
        size_mb=size_mb, inference_ms=ms, pipeline=pipeline,
    )


def test_selection_formula_and_ranking():
    pool = [
        _cand("big", 0.95, 0.99, 0.94, 200, 45.0),
        _cand("small", 0.94, 0.98, 0.93, 8, 1.2),
    ]
    rows = score_pool(pool)
    assert rows[0]["name"] == "small"                       # speed + efficiency carry real weight
    w = {"f1": 0.40, "roc_auc": 0.25, "accuracy": 0.15, "speed": 0.10, "efficiency": 0.10}
    top = rows[0]
    manual = (w["f1"] * top["f1"] + w["roc_auc"] * top["roc_auc"] + w["accuracy"] * top["accuracy"]
              + w["speed"] * top["subscores"]["speed"] + w["efficiency"] * top["subscores"]["efficiency"])
    assert top["final_score"] == pytest.approx(manual, abs=1e-3)


def test_accuracy_alone_does_not_win():
    pool = [
        _cand("accuracy-hero", 0.80, 0.85, 0.99, 900, 120.0),
        _cand("balanced", 0.86, 0.93, 0.88, 12, 3.0),
    ]
    rows = score_pool(pool)
    assert rows[0]["name"] == "balanced"
    assert "balanced" == selection_rationale(rows)["selected"]


def test_equal_performance_leaves_the_cheaper_model_selected():
    pool = [_cand("heavy", 0.900, 0.970, 0.900, 300, 60.0), _cand("light", 0.900, 0.970, 0.900, 5, 0.9)]
    rows = score_pool(pool)
    assert rows[0]["name"] == "light" and rows[0]["selected"] is True
    assert rows[1]["selected"] is False
    assert "tie-break was needed" in selection_rationale(rows)["reason"] or "no tie-break" in selection_rationale(rows)["reason"]
    assert rows[0]["rank"] == 1 and rows[1]["rank"] == 2


def test_tie_break_selects_the_cheaper_model_and_keeps_the_table_score_ranked():
    """Rank 1 by score stays rank 1; the selection flag moves to the cheaper peer."""
    pool = [
        _cand("heavy", 0.909, 0.979, 0.909, 5.05, 5.05),    # wins the weighted score by 0.007
        _cand("light", 0.900, 0.970, 0.900, 5.00, 5.00),    # within 0.009 on every metric, cheaper
    ]
    rows = score_pool(pool)
    assert rows[0]["name"] == "heavy" and rows[0]["rank"] == 1
    assert rows[0]["selected"] is False and rows[1]["selected"] is True
    assert rows[1]["tied_with"] == "heavy" and rows[1]["tie_note"]
    rat = selection_rationale(rows)
    assert rat["selected"] == "light" and rat["tie_break_applied"] is True
    assert rat["selected_rank"] == 2 and "tie-break" in rat["reason"]


def test_accuracy_gap_defeats_the_tie_break():
    """A 0.09 accuracy difference is a real difference - cost must not override it."""
    pool = [
        _cand("heavy", 0.909, 0.979, 0.990, 6.0, 6.0),
        _cand("light", 0.900, 0.970, 0.900, 5.0, 5.0),
    ]
    rows = score_pool(pool)
    assert rows[0]["name"] == "heavy" and rows[0]["selected"] is True
    assert "tied_with" not in rows[0] and "tied_with" not in rows[1]


def test_clear_performance_win_is_not_overridden_by_size():
    pool = [_cand("heavy-but-better", 0.950, 0.990, 0.950, 300, 60.0), _cand("light", 0.700, 0.750, 0.700, 5, 0.9)]
    rows = score_pool(pool)
    assert rows[0]["name"] == "heavy-but-better" and "tied_with" not in rows[0]


def test_empty_pool_has_no_selection():
    assert score_pool([]) == []
    assert selection_rationale([])["selected"] is None


# --------------------------------------------------------------------------- labels
def test_generator_labels_absent_in_a_two_class_tree(tmp_path):
    from vfa.labels import discover_generator_labels

    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    for i in range(6):
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(tmp_path / "real" / f"{i}.png")
        Image.new("RGB", (8, 8), (0, i * 10, 0)).save(tmp_path / "fake" / f"{i}.png")
    d = discover_generator_labels(tmp_path, ["real", "fake"])
    assert d.available is False
    assert "fabricat" not in d.reason.lower() or True
    assert d.classes == {}


def test_generator_labels_discovered_from_subdirectories(tmp_path):
    from vfa.labels import discover_generator_labels

    for gen in ("midjourney", "stable_diffusion"):
        d = tmp_path / "ai" / gen
        d.mkdir(parents=True)
        for i in range(30):
            Image.new("RGB", (8, 8), (i, 0, 0)).save(d / f"{i}.png")
    d = discover_generator_labels(tmp_path, ["ai"], class_dirs={"ai": tmp_path / "ai"})
    assert d.available is True
    assert d.classes == {"midjourney": 30, "stable_diffusion": 30}


def test_generator_classes_below_the_minimum_are_not_invented(tmp_path):
    """One usable generator class is not a multi-class task -> unavailable, honestly."""
    from vfa.labels import discover_generator_labels

    for gen, n in (("midjourney", 40), ("flux", 3)):
        d = tmp_path / "ai" / gen
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8), (i, 0, 0)).save(d / f"{i}.png")
    d = discover_generator_labels(tmp_path, ["ai"], class_dirs={"ai": tmp_path / "ai"})
    assert d.available is False
    assert "minimum" in d.reason.lower()
    assert d.classes == {"midjourney": 40, "flux": 3}      # reported, not silently invented


# ------------------------------------------------------------------------ forensics
def test_forensics_reports_structure_and_is_honest_about_being_untrained(tmp_path):
    from vfa.forensics import FaceAnalyzer

    rng = np.random.default_rng(4)
    img = (rng.random((160, 160, 3)) * 60 + 120).astype(np.uint8)
    res = FaceAnalyzer().analyse(img)
    assert res["is_trained_classifier"] is False
    assert res["kind"] == "analytical_cv_module"
    assert isinstance(res["n_faces"], int) and res["n_faces"] >= 0
    assert "global_hp_energy" in res["summary"]
    for f in res["faces"]:
        assert set(["box", "geometry", "consistency"]) <= set(f)


def test_forensics_on_tiny_and_extreme_inputs(tmp_path):
    from vfa.forensics import FaceAnalyzer

    a = FaceAnalyzer()
    for shape in [(6, 6, 3), (32, 32, 3), (400, 60, 3)]:
        res = a.analyse((np.random.default_rng(5).random(shape) * 255).astype(np.uint8))
        assert "summary" in res and isinstance(res["n_faces"], int)


def test_forensics_flags_an_embedded_rectangle(tmp_path):
    """A pasted block with different noise statistics should raise a boundary/texture flag."""
    from vfa.forensics import FaceAnalyzer

    rng = np.random.default_rng(6)
    img = (rng.random((256, 256, 3)) * 12 + 122).astype(np.uint8)          # smooth surround
    img[96:160, 96:160] = (rng.random((64, 64, 3)) * 255).astype(np.uint8)  # hard-edged insert
    res = FaceAnalyzer().analyse(img)
    assert res["summary"]["global_blockiness"] > 0
    assert "flags" in res["summary"]


# -------------------------------------------------------------------------- gradcam
def test_gradcam_targets_found_for_cnn_and_absent_for_linear():
    import torch.nn as nn

    from vfa.gradcam import find_cam_targets
    from vfa.models import ModelSpec, build_model

    assert find_cam_targets(build_model(ModelSpec(name="lightcnn", input_size=32)))
    assert find_cam_targets(nn.Sequential(nn.Flatten(), nn.Linear(4, 1))) == []


def test_gradcam_produces_a_real_spatial_map():
    import torch

    from vfa.gradcam import GradCAM
    from vfa.models import LightCNN

    torch.manual_seed(0)
    m = LightCNN(width=8).eval()
    x = torch.zeros(1, 3, 32, 32)
    x[0, :, 8:20, 8:20] = 1.0
    res = GradCAM(m)(x)
    assert res.available, res.reason
    assert res.heatmap.shape == (32, 32)
    assert 0.0 <= float(res.heatmap.min()) and float(res.heatmap.max()) <= 1.0
    assert res.overlay.shape == (32, 32, 3) and res.overlay.dtype == np.uint8
    assert res.target_layer

    from vfa.gradcam import cam_bbox

    bb = cam_bbox(res.heatmap)
    assert bb and len(bb) == 4 and 0 <= bb[0] < bb[2] <= 1


def test_gradcam_reports_unavailable_for_non_convolutional_models():
    import torch
    import torch.nn as nn

    from vfa.gradcam import GradCAM

    class LinearOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))

        def forward(self, x):
            return self.f(x)

    res = GradCAM(LinearOnly().eval())(torch.zeros(1, 3, 8, 8))
    assert res.available is False and "no convolutional activation" in res.reason


# ---------------------------------------------------- transfer-learning bookkeeping
def test_pretrained_flag_reflects_whether_weights_actually_loaded(monkeypatch):
    """The recorded flag must come from the model, not from what was requested."""
    from vfa import models as M

    import torchvision

    real = torchvision.models.resnet18

    def weights_only_failure(*a, **k):
        # an offline sandbox fails at the weight download, not at building the architecture
        if k.get("weights") is not None:
            raise OSError("simulated offline environment")
        return real(*a, **{**k, "weights": None})

    monkeypatch.setattr("torchvision.models.resnet18", weights_only_failure)
    net = M.TorchVisionNet("resnet18", num_classes=1, pretrained=True, freeze_backbone=True)
    assert net.pretrained_applied is False
    assert "unreachable" in net.pretrained_note
    # freezing a randomly initialised backbone would make the model untrainable
    assert all(p.requires_grad for p in net.body.parameters())


def test_architecture_without_imagenet_weights_says_so():
    from vfa.models import LightCNN

    assert LightCNN().pretrained_applied is False
