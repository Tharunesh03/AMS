"""Shared fixtures: a miniature dataset that mimics the real corpus structure.

The real corpus is 32x32 face crops in ``data/{real,fake}`` with ``<id>.jpg`` and
``<id> (k).jpg`` frames. The fixture reproduces exactly that layout (and the same kind of
statistical difference between classes) so tests exercise the real code paths in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=92)


def make_mini_dataset(root: Path, n_groups: int = 10, frames: int = 4, size: int = 32, seed: int = 0) -> Path:
    """real = smooth + low high-frequency energy; fake = sharp grid texture (like the corpus)."""
    rng = np.random.default_rng(seed)
    for g in range(n_groups):
        base = rng.random((size, size, 3)) * 40 + 110
        for f in range(frames):
            frame = np.roll(np.roll(base, f, axis=0), f, axis=1)
            real = frame + rng.normal(0, 6.0, frame.shape) + np.kron(
                np.sin(np.linspace(0, 3, size)), np.ones((size, 1))
            )[..., None] * 18.0
            yy, xx = np.indices((size, size))
            grid = np.where((yy % 8 == 0) | (xx % 8 == 0), 60.0, 0.0)
            fake = frame * 0.85 + grid[..., None] + rng.normal(0, 22.0, frame.shape)
            stem = f"{g:04d}"
            suffix = "" if f == 0 else f" ({f + 1})"
            _write(root / "REAL" / f"{stem}{suffix}.jpg", np.stack([real[..., i] for i in range(3)], axis=-1))
            _write(root / "FAKE" / f"{g}{suffix}.jpg", np.stack([fake[..., i] for i in range(3)], axis=-1))
    return root


@pytest.fixture(scope="session")
def mini_root(tmp_path_factory) -> Path:
    return make_mini_dataset(tmp_path_factory.mktemp("minidata"), n_groups=12, frames=4)


@pytest.fixture(scope="session")
def mini_db(mini_root, tmp_path_factory):
    from ams.trainer import build_data

    return build_data(
        root=mini_root,
        cache_dir=tmp_path_factory.mktemp("minicache"),
        archive=None,
        cache_size=48,
        train_frac=0.6,
        val_frac=0.2,
        seed=3,
        verbose=False,
    )
