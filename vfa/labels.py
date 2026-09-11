"""Label discovery for the *second* supervised task (generator fingerprinting).

Spec rule: only train classes that actually exist. This module looks for a usable
second label dimension - per-generator sub-directories, or generator tokens in file
names - and reports honestly when there is none, instead of inventing classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Known diffusion / GAN generators. Used only to *detect* labels in the provided data.
GENERATOR_TOKENS: Dict[str, List[str]] = {
    "midjourney": ["midjourney", "mj"],
    "dalle": ["dalle", "dall-e", "dall_e", "openai"],
    "stable_diffusion": ["stable_diffusion", "stablediffusion", "sd1", "sd2", "sd3", "sdxl", "stable-diffusion"],
    "adobe_firefly": ["firefly", "adobe_firefly"],
    "leonardo_ai": ["leonardo"],
    "ideogram": ["ideogram"],
    "flux": ["flux"],
    "imagen": ["imagen"],
    "dalle3": ["dalle3", "dall-e-3"],
    "gans": ["stylegan", "progan", "biggan", "pggan"],
    "deepfake": ["deepfake", "faceswap", "simswap", "inswapper", "roop"],
}
MIN_PER_CLASS = 25


@dataclass
class LabelDimension:
    name: str
    available: bool
    classes: Dict[str, int] = field(default_factory=dict)
    reason: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "classes": self.classes,
            "n_classes": len(self.classes),
            "reason": self.reason,
            "source": self.source,
        }


def normalise_label(text: str) -> str:
    """Canonical form for a directory/file token: ``Stable-Diffusion XL`` -> ``stable_diffusion_xl``."""
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _token_of(text: str) -> Optional[str]:
    t = text.lower().replace("-", "_").replace(" ", "_")
    for canon, aliases in GENERATOR_TOKENS.items():
        for a in aliases:
            if re.search(rf"(^|[^a-z0-9]){re.escape(a)}([^a-z0-9]|$)", t):
                return canon
    return None


def discover_generator_labels(
    data_root: Path,
    classes: Sequence[str],
    class_dirs: Optional[Dict[str, Path]] = None,
) -> LabelDimension:
    """Look for per-generator sub-labels; never fabricate them.

    Two sources are inspected, in order:
      1. nested directories, e.g. ``data/ai/midjourney/*.png``
      2. generator tokens in file names, e.g. ``img_sdxl_001.jpg``
    A dimension is only "available" when at least two classes clear :data:`MIN_PER_CLASS`;
    a single usable class is not a multi-class task, so it is reported as unavailable.
    """
    root = Path(data_root)
    # class *labels* are normalised ("fake") while the directories may not be ("FAKE"),
    # so resolve them case-insensitively - otherwise nested generator folders are missed.
    listing = {normalise_label(q.name): q for q in root.iterdir() if q.is_dir()} if root.exists() else {}
    dirs: Dict[str, Path] = {}
    for c in classes:
        if class_dirs and c in class_dirs:
            dirs[c] = Path(class_dirs[c])
            continue
        exact = root / c
        dirs[c] = exact if exact.is_dir() else listing.get(normalise_label(c), exact)

    nested: Dict[str, int] = {}
    for cls, d in dirs.items():
        if not Path(d).exists():
            continue
        for sub in sorted(Path(d).iterdir()):
            if sub.is_dir():
                n = sum(1 for p in sub.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
                if n:
                    key = sub.name.lower().replace("-", "_").replace(" ", "_")
                    nested[key] = nested.get(key, 0) + n
    if nested:
        return _verdict(nested, "nested_directories")

    tokens: Dict[str, int] = {}
    for cls, d in dirs.items():
        if not Path(d).exists():
            continue
        for p in list(Path(d).rglob("*"))[:4000]:
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                t = _token_of(p.stem)
                if t:
                    tokens[t] = tokens.get(t, 0) + 1
    if tokens:
        return _verdict(tokens, "filename_tokens")

    return LabelDimension(
        "generator", False, {},
        reason=(
            "the provided dataset carries exactly one label dimension (the real/ai class folders); "
            "no per-generator sub-directories and no generator tokens in file names, so generator "
            "fingerprinting cannot be trained - it would require inventing labels"
        ),
        source="none",
    )


def _verdict(found: Dict[str, int], source: str) -> LabelDimension:
    usable = {k: v for k, v in found.items() if v >= MIN_PER_CLASS}
    if len(usable) >= 2:
        dropped = {k: v for k, v in found.items() if k not in usable}
        note = f"; ignored below-minimum classes: {dropped}" if dropped else ""
        return LabelDimension("generator", True, usable, reason=f"discovered from {source}{note}", source=source)
    return LabelDimension(
        "generator", False, found,
        reason=(
            f"only {len(usable)} generator class(es) reach the {MIN_PER_CLASS}-image minimum "
            f"(counts: {found}); one class is not a multi-class task, so nothing is trained"
        ),
        source=source,
    )
