"""Dataset inspection: class discovery, corruption handling, duplicate detection, balance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ams.dataset import ensure_extracted, scan_dataset
from tests.conftest import make_mini_dataset, _write


def test_discovers_classes_counts_and_geometry(tmp_path: Path):
    root = make_mini_dataset(tmp_path / "d", n_groups=4, frames=3, size=32)
    records, inv = scan_dataset(root)
    assert sorted(inv.classes_found) == ["fake", "real"]
    assert inv.classes_found["real"] == 12 and inv.classes_found["fake"] == 12
    assert inv.n_valid == 24 and inv.n_corrupt == 0
    assert inv.sizes == {"32x32": 24}
    assert "JPEG" in inv.formats
    assert abs(inv.minority_ratio - 1.0) < 1e-9          # perfectly balanced
    assert {r.width for r in records} == {32}


def test_source_grouping_parses_frames(tmp_path: Path):
    root = make_mini_dataset(tmp_path / "d", n_groups=3, frames=4)
    records, inv = scan_dataset(root)
    by_src = {}
    for r in records:
        by_src.setdefault((r.label, r.source_id), []).append(r.frame)
    assert all(sorted(v) == [1, 2, 3, 4] for v in by_src.values()), by_src
    assert inv.n_groups == 6                              # 3 sources x 2 classes


def test_corrupt_and_truncated_files_are_rejected(tmp_path: Path):
    root = make_mini_dataset(tmp_path / "d", n_groups=3, frames=2)
    bad = root / "REAL" / "broken.jpg"
    bad.write_bytes(b"\xff\xd8not really a jpeg")
    tiny = root / "FAKE" / "tiny.jpg"
    tiny.write_bytes(b"\x00\x01")
    records, inv = scan_dataset(root)
    assert inv.n_corrupt == 2
    assert inv.n_valid == 12
    assert all("decode error" in r["error"] or "too small" in r["error"] for r in inv.rejected)


def test_exact_and_near_duplicates_detected(tmp_path: Path):
    root = make_mini_dataset(tmp_path / "d", n_groups=3, frames=2)
    src = root / "REAL" / "0000.jpg"
    dup = root / "REAL" / "0002.jpg"          # same source id as an existing frame set
    dup.write_bytes(src.read_bytes())
    records, inv = scan_dataset(root)
    assert inv.n_duplicate_exact >= 1, inv.n_duplicate_exact


def test_degenerate_tiny_images_are_filtered(tmp_path: Path):
    root = tmp_path / "d"
    _write(root / "REAL" / "0000.jpg", np.full((8, 8, 3), 128, np.uint8))
    _write(root / "REAL" / "0001.jpg", np.full((4, 4, 3), 128, np.uint8))  # too small -> rejected
    _write(root / "FAKE" / "0.jpg", np.zeros((8, 8, 3), np.uint8) + 40)
    records, inv = scan_dataset(root)
    assert inv.n_valid == 2 and inv.n_corrupt == 1


def test_per_class_stats_are_reported(tmp_path: Path):
    root = make_mini_dataset(tmp_path / "d", n_groups=3, frames=2)
    _, inv = scan_dataset(root)
    st = inv.per_class_stats
    assert set(st) == {"real", "fake"}
    # the fixture makes the 'fake' class noisier/griddier than 'real'
    assert st["fake"]["laplacian_var_median"] > st["real"]["laplacian_var_median"]


def test_ensure_extracted_from_zip(tmp_path: Path):
    import zipfile

    src = make_mini_dataset(tmp_path / "src", n_groups=2, frames=2)
    zipped = tmp_path / "archive.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for p in src.rglob("*.jpg"):
            zf.write(p, f"data/{p.relative_to(src)}")
    out = tmp_path / "out"
    ensure_extracted(out, zipped)
    assert (out / "data" / "REAL" / "0000.jpg").exists()
    _, inv = scan_dataset(out / "data")
    assert inv.n_valid == 8
    assert not (out / "data" / "evil").exists()


def test_no_path_escape_from_archive(tmp_path: Path):
    import zipfile

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.jpg", b"\xff\xd8garbage")
    try:
        ensure_extracted(tmp_path / "out", evil)
    except RuntimeError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("path traversal member should have been rejected")
