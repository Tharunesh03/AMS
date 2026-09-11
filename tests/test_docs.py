"""Documentation must not drift from the artifacts it describes.

These tests fail if the README quotes numbers that differ from ``models/metrics.json``, if any
shipped text overclaims certainty, or if the dataset's missing label dimension stops being
documented. They skip (not pass) when a run has not produced artifacts yet.
"""

from __future__ import annotations

import json
import os
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
    from vfa.selection import DEFAULT_WEIGHTS

    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, DEFAULT_WEIGHTS
    formula = " + ".join(
        f"{DEFAULT_WEIGHTS[k]:.2f}\u00b7{name}"
        for k, name in (("f1", "F1"), ("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"),
                        ("speed", "Speed"), ("efficiency", "Efficiency"))
    )
    for f in (README, ROOT / "app" / "static" / "metrics.html", ROOT / "app" / "static" / "model.html"):
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
    src = (ROOT / "vfa" / "generator.py").read_text()
    pipe = (ROOT / "vfa" / "pipeline.py").read_text()
    for name in ("vfa/generator.py", "--no-generator-task", "fit_unknown_gate", "apply_gate",
                 "models/generator_classifier"):
        assert name in sec, f"README S4 cites {name}"
    for fn in ("def fit_unknown_gate", "def apply_gate", "def build_generator_bundle",
               "def export_generator_model", "def generator_group_keys"):
        assert fn in src, f"{fn} missing from vfa/generator.py"
    assert "def run_generator_task" in pipe and "build_generator_bundle" in pipe
    assert src.count('UNKNOWN_LABEL = "') == 1                     # one constant, not scattered literals
    gate_fn = src[src.index("def apply_gate"): src.index("def save_gate")]
    assert "UNKNOWN_LABEL" in gate_fn and '"generator"' in gate_fn  # the gate itself returns the constant
    for claim in ("validation", "90 %"):
        assert claim in sec, f"README S4 must state how the gate was fitted ({claim})"
    assert (ROOT / "tests" / "test_generator_task.py").exists()
    # the honest case must still be the one this dataset produces
    lbl = (ROOT / "vfa" / "labels.py").read_text()
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


def test_pages_render_both_generator_states_from_the_payload():
    """The UI must read the model's answer, never assert availability or fabricate a name."""
    js = (ROOT / "app" / "static" / "analyze.js").read_text()
    html = (ROOT / "app" / "static" / "analyze.html").read_text()
    assert "generator_identification" in js, "the panel must read the per-prediction generator field"
    assert "acceptance_rule" in js and "measured_metrics" in js, "the gate and its metrics must be shown"
    assert "genStatus" in html and "genStatus" in js
    assert "UNKNOWN" in js and "UNKNOWN" in html, "abstention must be visible in the markup, not only in data"
    assert "S.model || {}).generator_classifier" in js, "the artifact record drives the not-trained explanation"
    assert "dataset_labels" in js, "the reason must quote what the dataset actually carries"
    blob = html + js
    for phrase in ("insufficient", "invent", "not proof", "no per-generator labels"):
        assert phrase in blob, f"UI copy must keep the claim {phrase!r}"


def test_report_pane_escapes_before_it_renders_markdown():
    """Reports are rendered as HTML, so escaping must come first and survive the split."""
    js = (ROOT / "app" / "static" / "common.js").read_text()
    page = (ROOT / "app" / "static" / "reports.js").read_text()
    html = (ROOT / "app" / "static" / "reports.html").read_text()
    assert "function mdLite" in js, "the shared renderer must exist exactly once"
    assert js.count("function mdLite") == 1
    assert "VFA.mdLite(" in page, "the reports page must render through it"
    body = js[js.index("function mdLite"): js.index("/* ---------------------------------------------------------------- shared state */")]
    assert body.index("const inline = (t) => esc(t)") < body.index("<code>"), "escape before inserting tags"
    assert "String(src || '')" in body, "the renderer must take the report text, not assume a shape"
    assert "/api/report/" in body, "figures are fetched through the whitelisted report route"
    assert 'class="reportview md"' in html
    assert "mdLite(j.markdown)" in page.replace("VFA.mdLite(", "mdLite(")


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


# ------------------------------------------------------------------- brand and page chrome
UI = ROOT / "app" / "static"
PAGES = ["index.html", "analyze.html", "model.html", "metrics.html", "reports.html"]
PAGE_SCRIPTS = {"index.html": "home.js", "analyze.html": "analyze.js",
                "model.html": "model.js", "metrics.html": "metrics.js", "reports.html": "reports.js"}
OLD_NAME = re.compile(r"\bAMS\b")


def _brand_hits(text: str) -> list[str]:
    """Find the old abbreviation, ignoring filesystem paths that happen to contain it.

    Recorded artifacts quote the directory the training run executed in (``/home/user/AMS/...``).
    Those are measurements of the environment, not branding, so a leading or trailing ``/`` exempts
    the match; everything else has to carry the product name.
    """
    hits = []
    for m in OLD_NAME.finditer(text):
        before = text[m.start() - 1] if m.start() else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if before == "/" or after in {"/", '"'}:
            continue
        hits.append(text[max(0, m.start() - 60):m.end() + 60].replace("\n", " "))
    return hits


def _source_text_files() -> list[Path]:
    out: list[Path] = []
    for rel in ["README.md", "pyproject.toml", "Makefile", "Dockerfile", "render.yaml",
                "requirements.txt", "runtime.txt", "train.py"]:
        f = ROOT / rel
        if f.exists():
            out.append(f)
    for d in ("app", "vfa", "configs", "tools"):
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file() or p.suffix not in {".py", ".html", ".css", ".js", ".yaml", ".yml", ".toml", ".md"}:
                continue
            # the one-shot migration script has to name the old package to rewrite it
            if p.name.startswith("rebrand_"):
                continue
            out.append(p)
    return out


def test_the_old_project_abbreviation_is_gone_from_shipped_text():
    bad = []
    for f in _source_text_files():
        for h in _brand_hits(f.read_text(errors="replace")):
            bad.append(f"{f.relative_to(ROOT)}: ...{h}...")
    assert not bad, "shipped text still carries the old abbreviation:\n" + "\n".join(bad)

    # and the generated documents must use the new product name
    card = MODELS / "MODEL_CARD.md"
    if card.exists():
        assert not _brand_hits(card.read_text()), "MODEL_CARD.md must carry the product name"


def test_the_project_is_named_visual_forensic_ai_in_every_entry_point():
    first = README.read_text().splitlines()[0]
    assert "Visual Forensic AI" in first, f"the README title must carry the product name, got: {first!r}"
    py = (ROOT / "pyproject.toml").read_text()
    assert 'name = "visual-forensic-ai"' in py, py[:200]
    srv = (ROOT / "app" / "server.py").read_text()
    assert 'APP_TITLE = "Visual Forensic AI"' in srv
    for page in PAGES:
        text = (UI / page).read_text()
        assert "Visual Forensic AI" in text, page
        assert "<title>" in text and "· Visual Forensic AI</title>" in text, f"{page} title must name the product"


def test_every_page_offers_the_same_navigation_and_theme_control():
    common = (UI / "common.js").read_text()
    css = (UI / "styles.css").read_text()
    for page in PAGES:
        html = (UI / page).read_text()
        for target in PAGES:                                   # full navigation on every page
            href = "/" + ("" if target == "index.html" else target.replace(".html", ""))
            assert f'href="{href}"' in html, f"{page} cannot reach {href}"
        assert 'data-theme-btn' in html, f"{page} has no theme control"
        assert "vfa-theme" in html, f"{page} must apply the stored theme before paint (no flash)"
        assert 'aria-label="Toggle dark theme"' in html, f"{page} theme button needs a label"
        assert "/static/common.js" in html and "/static/styles.css" in html, page
        assert 'class="nav"' in html and 'aria-label="Main"' in html, f"{page} nav must be a labelled landmark"
        assert 'data-status-pills' in html, f"{page} must show whether a model is loaded"
        assert 'id="main"' in html and 'Skip to content' in html, f"{page} needs a skip link"
    for needed in ("localStorage", "prefers-color-scheme", "data-theme", "addEventListener"):
        assert needed in common, f"theme handling must survive in common.js: {needed}"
    assert 'location.pathname' in common, "the current nav item must be marked from the URL"
    assert 'aria-current' in common
    assert '[data-theme="dark"]' in css and ":root" in css, "both themes must be defined as variables"
    assert "prefers-reduced-motion" in css


def test_page_scripts_only_reference_ids_their_own_page_defines():
    """A typo in a selector is invisible until the panel stays empty, so check the wiring."""
    for page, script in PAGE_SCRIPTS.items():
        html = (UI / page).read_text()
        defined = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
        used = set(re.findall(r"\$\('#([A-Za-z0-9_]+)'", (UI / script).read_text()))
        missing = sorted(used - defined)
        assert not missing, f"{script} writes to ids {page} does not define: {missing}"
    shared = (UI / "common.js").read_text()
    for hook in ("data-status-pills", "data-model-name", "data-theme-btn"):
        assert hook in shared, hook


def test_no_shipped_module_still_imports_the_old_package_name():
    for d in ("app", "vfa", "tools"):
        for f in (ROOT / d).rglob("*.py"):
            src = f.read_text()
            for bad in ("import ams", "from ams", "AMS_MODELS_DIR", "AMS_REPORTS_DIR", "AMS_SAMPLES_DIR"):
                assert bad not in src or f.name.startswith("rebrand_"), f"{f.relative_to(ROOT)} mentions {bad!r}"


def test_the_ui_ships_no_measurements_of_its_own():
    """Every 4-decimal number on screen comes from the API, so none may be typed into a page."""
    for f in sorted(UI.glob("*")):
        text = f.read_text()
        assert not re.search(r"\b0\.\d{4}\b", text), f"{f.name} hard-codes a metric value"
        assert "/home/user" not in text, f"{f.name} quotes a machine path"
        assert not re.search(r"\bdata/(REAL|FAKE)\b", text), f"{f.name} reads the raw dataset"


def test_readme_documents_every_route_the_server_exposes():
    srv = (ROOT / "app" / "server.py").read_text()
    routes = set(re.findall(r'"(/[a-z0-9/_{}-]*)"', srv))
    # /static is an asset mount and the parameterised routes are covered by their own tests
    served = {r for r in routes
              if r not in {"/static"} and not r.startswith(("/api/report/{", "/api/samples/{"))}
    doc = README.read_text()
    sec = doc[doc.index("## 8."):]
    sec = sec[: sec.index("## 9.")] if "## 9." in sec else sec
    for r in sorted(served):
        assert r in doc, f"server exposes {r} but the README never mentions it"
    for r in re.findall(r"^\|\s*`?(GET|POST)\s+(/[a-z0-9/_{}.-]+)`?", sec, flags=re.M):
        assert r[1] in served or r[1].startswith("/api/"), f"README documents {r[1]}, which the server does not define"
    static_pages = {p.name for p in UI.glob("*.html")}
    declared = set(re.findall(r'"[a-z_]+\.html"', srv))
    assert {f'"{p}"' for p in static_pages} <= declared, f"pages in app/static not routed: {static_pages}"


def test_every_page_still_works_when_no_model_is_trained(tmp_path):
    """An untrained checkout must serve the real pages and say what to run - never a placeholder result."""
    code = (
        "import sys, json; sys.path.insert(0, {root!r})\n"
        "from fastapi.testclient import TestClient\n"
        "from app.server import app\n"
        "out = {{}}\n"
        "with TestClient(app) as c:\n"
        "    for route in ('/', '/analyze', '/model', '/metrics', '/reports'):\n"
        "        r = c.get(route)\n"
        "        out[route] = (r.status_code, 'python train.py' in r.text, 'class=\"nav\"' in r.text,\n"
        "                      'data-theme-btn' in r.text)\n"
        "    out['/api/health'] = (200, c.get('/api/health').json())\n"
        "    out['/api/model'] = (c.get('/api/model').status_code, c.get('/api/model').json().get('detail'))\n"
        "    out['/api/metrics'] = (c.get('/api/metrics').status_code, None)\n"
        "print(json.dumps(out))\n"
    ).format(root=str(ROOT))
    env = dict(os.environ, VFA_MODELS_DIR=str(tmp_path / "models"),
               VFA_REPORTS_DIR=str(tmp_path / "reports"), VFA_SAMPLES_DIR=str(tmp_path / "samples"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for route in ("/", "/analyze", "/model", "/metrics", "/reports"):
        status, names_command, has_nav, has_theme = out[route]
        assert status == 200, route
        assert names_command, f"{route} must tell the reader to run `python train.py` when nothing is trained"
        assert has_nav and has_theme, f"{route} must keep the navigation and the theme control"
    assert out["/api/health"][1]["status"] == "model_unavailable"
    assert out["/api/health"][1]["model_loaded"] is False
    assert "train.py" in (out["/api/model"][1] or "")
    assert out["/api/model"][0] == 503 and out["/api/metrics"][0] == 503
