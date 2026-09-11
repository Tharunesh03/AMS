#!/usr/bin/env python3
"""One-off migration: rewrite project-name references inside already-generated artifacts.

The project was renamed from its old two-letter package abbreviation to **Visual Forensic AI**
(package ``vfa``). Training artifacts (``models/*.json``, ``models/MODEL_CARD.md``,
``reports/*.md``) were produced before the rename, so they still point at module paths
that no longer exist (``ams.forensics``, ``python -m ams predict``, ``ams/generator.py``)
and carry the old model-name prefix.

What this tool may touch, and nothing else:

* exact identifier/path strings from :data:`REPLACEMENTS`,
* and it refuses to write if any numeric value or any JSON key changes - metrics are
  measurements, not branding, so they must survive byte-identical.

Verify afterwards with ``python -m vfa evaluate``, which re-measures the shipped model on
the whole test split and compares against ``models/metrics.json``.

    python tools/rebrand_artifacts.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent

REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("ams-ai-detector-", "vfa-ai-detector-"),
    ("python -m ams", "python -m vfa"),
    ("from ams.", "from vfa."),
    ("ams.forensics", "vfa.forensics"),
    ("ams.predictor", "vfa.predictor"),
    ("ams.pipeline", "vfa.pipeline"),
    ("ams/generator.py", "vfa/generator.py"),
    ("ams/forensics.py", "vfa/forensics.py"),
    ("ams/metrics.py", "vfa/metrics.py"),
    ("`ams/", "`vfa/"),
)

TARGETS: Tuple[str, ...] = (
    "models/metrics.json",
    "models/model_info.json",
    "models/MODEL_CARD.md",
    "models/ai_detector/classes.json",
    "models/preprocessing.json",
    "reports/eda.md",
    "reports/model_comparison.md",
    "reports/training_report.md",
    "reports/production_evaluation.md",
    "reports/production_evaluation.json",
)

_NUMBERS = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+")


def _numbers(text: str) -> List[str]:
    return _NUMBERS.findall(text)


def _keys(path: Path, text: str):
    if path.suffix == ".json":
        return sorted(_walk_keys(json.loads(text)))
    return None


def _walk_keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_keys(v)


def rebrand(check: bool = False) -> int:
    changed: List[str] = []
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            continue
        before = path.read_text()
        after = before
        for old, new in REPLACEMENTS:
            after = after.replace(old, new)
        if after == before:
            continue
        # hard guarantees: no number moved, no key moved, no leftover package token
        if _numbers(before) != _numbers(after):
            print(f"REFUSING to write {rel}: a numeric value changed", file=sys.stderr)
            return 2
        if path.suffix == ".json":
            try:
                json.loads(after)
            except json.JSONDecodeError as exc:
                print(f"REFUSING to write {rel}: invalid JSON after rewrite ({exc})", file=sys.stderr)
                return 2
            if _keys(path, before) != _keys(path, after):
                print(f"REFUSING to write {rel}: a JSON key changed", file=sys.stderr)
                return 2
        if re.search(r"(?<![\w.-])ams(?=\.|/|\b)", after):
            leftover = re.findall(r".{40}(?<![\w.-])ams(?=\.|/|\b).{20}", after)
            print(f"REFUSING to write {rel}: unhandled 'ams' reference remains: {leftover[:2]}",
                  file=sys.stderr)
            return 2
        changed.append(rel)
        if not check:
            path.write_text(after)
    for rel in changed:
        print(("would rewrite " if check else "rewrote ") + rel)
    if not changed:
        print("nothing to do - artifacts already carry the current project name")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report what would change without writing")
    args = ap.parse_args(argv)
    return rebrand(check=bool(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
