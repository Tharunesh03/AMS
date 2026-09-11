"""Documentation must not drift from the artifacts it describes.

These tests fail if the README quotes numbers that differ from ``models/metrics.json``, if any
shipped text overclaims certainty, or if the dataset's missing label dimension stops being
documented. They skip (not pass) when a run has not produced artifacts yet.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
README = ROOT / "README.md"


def _readme_and_docs() -> list[Path]:
    files = [README, MODELS / "MODEL_CARD.md"]
    files += sorted((ROOT / "reports").glob("*.md"))
    return [f for f in files if f.exists()]


def test_readme_results_section_is_in_sync_with_metrics():
    if not (MODELS / "metrics.json").exists():
        import pytest

        pytest.skip("no models/metrics.json yet - run `python train.py` first")
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sync_readme_results.py"), "--check"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert r.returncode == 0, f"README results section is stale:\n{r.stderr}"


OVERCLAIM_PATTERNS = [
    r"100\s*%\s*(accurate|correct|reliable)",
    r"guaranteed?(ly)?\s+(to|detect|catch)",
    r"foolproof",
    r"perfect\s+(detector|classifier|model)",
    r"always\s+(correct|right|detects)",
    r"cannot\s+be\s+fooled",
    r"zero\s+false\s+positives",
]
_CLAUSE = re.compile(r"(?<=[.!?\n])\s*")
_NEGATED = re.compile(r"\b(no|not|never|rarely|unlikely|cannot|can't|without|claim)\b", re.I)


def _overclaim_hits(text: str) -> list[str]:
    """Flag certainty language - but not when it is explicitly denied in its own sentence.

    "No detector is 100% accurate" is a limitation statement, not an overclaim, so sentences
    containing a negation are skipped. That keeps the guard strict on prose while letting the
    honesty statements quote the number they deny.
    """
    hits: List[str] = []
    for pat in OVERCLAIM_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start : (line_end if line_end > 0 else len(text))]
            clause = next((c for c in _CLAUSE.split(line) if m.group(0).lower() in c.lower()), line)
            if _NEGATED.search(clause):
                continue
            hits.append(f"{m.group(0)!r} in ...{text[max(0, m.start()-70):m.end()+70]}...")
    return hits


def test_no_overclaiming_language_anywhere_in_the_docs():
    hits = []
    for f in _readme_and_docs():
        for h in _overclaim_hits(f.read_text()):
            hits.append(f"{f.name}: {h}")
    assert not hits, "docs claim certainty the measurements do not support:\n" + "\n".join(hits)


def test_the_overclaim_guard_still_catches_a_real_one():
    """A guard that only ever passes is worthless - prove it fires."""
    bad = ("This detector is 100% accurate on every image.\n"
           "It is guaranteed to catch any manipulation.\n"
           "The model is a perfect detector and foolproof.\n")
    good = ("No detector is 100% accurate and this one makes no such claim.\n"
            "Predictions are probabilistic; the model is not proof of origin.\n")
    hits_bad = _overclaim_hits(bad)
    assert len(hits_bad) >= 4, hits_bad
    assert _overclaim_hits(good) == [], _overclaim_hits(good)


def test_readme_documents_what_the_dataset_cannot_support():
    text = README.read_text().lower()
    assert "generator" in text
    assert ("cannot be trained" in text or "not trainable" in text), "the missing generator labels must be stated"
    assert "invented" in text or "fabricat" in text, "say explicitly that nothing was fabricated"
    assert "not a trained" in text or "not a deepfake detector" in text, "the face module must be labelled"
    assert "probabilistic" in text


def test_selection_weights_are_quoted_verbatim_in_the_docs_and_the_ui():
    from ams.selection import DEFAULT_WEIGHTS

    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, DEFAULT_WEIGHTS
    formula = " + ".join(
        f"{DEFAULT_WEIGHTS[k]:.2f}\u00b7{name}"
        for k, name in (("f1", "F1"), ("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"),
                        ("speed", "Speed"), ("efficiency", "Efficiency"))
    )
    for f in (README, ROOT / "app" / "static" / "index.html"):
        if not f.exists():
            continue
        assert formula in f.read_text(), f"{f.name} does not quote the real weights: {formula}"


def test_shipped_artifact_set_is_what_the_docs_promise():
    if not (MODELS / "metrics.json").exists():
        import pytest

        pytest.skip("no trained model yet")
    promised = ["metrics.json", "model_info.json", "MODEL_CARD.md", "preprocessing.json",
                "dataset_inspection.json", "reference_stats.json", "ai_detector/classes.json"]
    missing = [p for p in promised if not (MODELS / p).exists()]
    assert not missing, f"documented artifacts missing: {missing}"
    prod = list((MODELS / "ai_detector").glob("model.*"))
    assert len(prod) == 1, f"exactly one production model must ship, found {[p.name for p in prod]}"
    doc = json.loads((MODELS / "model_info.json").read_text())
    assert doc.get("known_limitations"), "model card must carry limitations"
    assert doc.get("honesty"), "model card must carry the honesty statements"
    # a second task must ship complete or not at all (it is absent on datasets without those labels)
    gdir = MODELS / "generator_classifier"
    metrics = json.loads((MODELS / "metrics.json").read_text())
    trained = ((metrics.get("tasks") or {}).get("generator_classifier") or {}).get("trained")
    if gdir.exists():
        assert trained is True, "an exported generator model must be declared in metrics.json"
        for extra in ("classes.json", "gate.json", "model_config.json"):
            assert (gdir / extra).exists(), f"generator_classifier/{extra} missing"
        assert len(list(gdir.glob("model.*"))) == 1
    else:
        assert not trained, "metrics.json claims a generator model that was never exported"


def test_generator_task_claims_in_the_readme_point_at_real_code():
    """README S4 promises a working second task; every name it cites must exist."""
    text = README.read_text()
    sec = text[text.index("## 4."):]
    sec = sec[: sec.index("## 5.")]
    src = (ROOT / "ams" / "generator.py").read_text()
    pipe = (ROOT / "ams" / "pipeline.py").read_text()
    for name in ("ams/generator.py", "--no-generator-task", "fit_unknown_gate", "apply_gate",
                 "models/generator_classifier"):
        assert name in sec, f"README S4 cites {name}"
    for fn in ("def fit_unknown_gate", "def apply_gate", "def build_generator_bundle",
               "def export_generator_model", "def generator_group_keys"):
        assert fn in src, f"{fn} missing from ams/generator.py"
    assert "def run_generator_task" in pipe and "build_generator_bundle" in pipe
    assert src.count('UNKNOWN_LABEL = "') == 1                     # one constant, not scattered literals
    gate_fn = src[src.index("def apply_gate"): src.index("def save_gate")]
    assert "UNKNOWN_LABEL" in gate_fn and '"generator"' in gate_fn  # the gate itself returns the constant
    for claim in ("validation", "90 %"):
        assert claim in sec, f"README S4 must state how the gate was fitted ({claim})"
    assert (ROOT / "tests" / "test_generator_task.py").exists()
    # the honest case must still be the one this dataset produces
    lbl = (ROOT / "ams" / "labels.py").read_text()
    assert "would require inventing labels" in lbl, "the no-labels verdict must stay explicit"


def test_transfer_learning_record_is_consistent_with_the_frozen_backbone_claim():
    """If no weights loaded, nothing can claim to be a frozen-backbone transfer arm."""
    text = README.read_text()
    assert "frozen" in text and "transfer_learning" in text
    cfg_text = (ROOT / "configs" / "default.yaml").read_text()
    assert "freeze_backbone" in cfg_text, "the frozen arm must be documented in the config"
    metrics = MODELS / "metrics.json"
    if not metrics.exists():
        import pytest

        pytest.skip("no models/metrics.json yet - run `python train.py` first")
    tl = json.loads(metrics.read_text()).get("transfer_learning") or {}
    assert {"requested", "weights_applied_for", "frozen_backbone_applied_for"} <= set(tl)
    if not tl["weights_applied_for"]:
        assert tl["frozen_backbone_applied_for"] == [], "no weights, so nothing was frozen"
        assert "unreachable" in tl["note"] or "no loadable pretrained weights" in tl["note"]
        assert "not trained" not in tl["note"]
    card = json.loads((MODELS / "model_info.json").read_text())
    assert card.get("transfer_learning", {}).get("requested") == tl["requested"]


def test_dashboard_renders_both_generator_states_from_the_payload():
    """The UI must read the model's answer, never assert availability or fabricate a name."""
    js = (ROOT / "app" / "static" / "app.js").read_text()
    html = (ROOT / "app" / "static" / "index.html").read_text()
    assert "generator_identification" in js, "the panel must read the per-prediction generator field"
    assert "UNKNOWN" in js and "UNKNOWN" in html, "unknown-generator handling must be visible in the UI"
    assert "acceptance_rule" in js, "the gate that produced the answer must be shown"
    assert "genStatus" in html and "genStatus" in js
    assert "state.model.generator_classifier" in js, "the model card drives the not-trained explanation"
    blob = html + js
    for phrase in ("never", "nothing is invented", "UNKNOWN"):
        assert phrase in blob, f"UI copy must keep the claim {phrase!r}"


def test_report_pane_escapes_before_it_renders_markdown():
    """Reports are rendered as HTML in the dashboard, so escaping must come first."""
    js = (ROOT / "app" / "static" / "app.js").read_text()
    html = (ROOT / "app" / "static" / "index.html").read_text()
    assert "function mdLite" in js and "mdLite(j.markdown)" in js
    body = js[js.index("function mdLite"): js.index("/* ---------------------------------------------------------------- samples */")]
    assert body.index("const inline = (t) => esc(t)") < body.index("<code>") , "escape before inserting tags"
    assert "String(src || '')" in body, "the renderer must take the report text, not assume a shape"
    assert "/api/report/" in body, "figures are fetched through the whitelisted report route"
    assert 'class="reportview md"' in html


def test_makefile_targets_reference_existing_files():
    mk = (ROOT / "Makefile").read_text()
    targets = dict(re.findall(r"^([a-z_-]+):.*?## (.+)$", mk, flags=re.M))
    assert {"train", "serve", "test", "smoke", "evaluate", "demo"} <= set(targets), targets.keys()
    for rel in ["configs/default.yaml", "configs/smoke.yaml", "train.py", "Dockerfile", "render.yaml",
                "tools/sync_readme_results.py"]:
        assert (ROOT / rel).exists(), rel
    assert "configs/smoke.yaml" in mk and "app.server:app" in mk


def test_requirements_pins_the_opencv_major_that_ships_cascades():
    req = (ROOT / "requirements.txt").read_text()
    assert re.search(r"opencv-python-headless>=4[^\n]*,<5", req), (
        "opencv 5.x wheels have no haarcascades; the face module needs the 4.x pin"
    )
