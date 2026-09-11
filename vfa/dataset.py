"""Dataset inspection, validation and caching for the image-provenance corpus.

The corpus committed to this repository is ``archive (4).zip`` with the layout::

    data/
      FAKE/   0.jpg, 0 (2).jpg ... 999 (10).jpg     (synthetic / generated face crops)
      REAL/   0000.jpg, 0000 (2).jpg ... 0999 (10).jpg  (genuine captures)

Everything here is *dataset-first*: class names, class counts, image geometry and
label availability are discovered from disk. Nothing about the dataset is hard coded,
and the inspection results feed the model card and the report.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 256_000_000  # we only ever see small crops; keep DoS guard on for big files

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# "0042 (3).jpg" -> source id "0042", frame index 3 ; "0042.jpg" -> source "0042", frame 1
NAME_RE = re.compile(r"^(?P<stem>.+?)(?: \((?P<frame>\d+)\))?\.(?P<ext>[\w]+)$", re.IGNORECASE)

# Label semantics used throughout the project (attack/synthetic == positive class).
POSITIVE_LABELS = {"fake", "ai", "ai_generated", "generated", "synthetic", "deepfake", "spoof", "fake_face"}
NEGATIVE_LABELS = {"real", "bona_fide", "bonafide", "genuine", "authentic", "original", "true"}


@dataclass
class ImageRecord:
    """One validated image on disk."""

    path: str
    rel_path: str
    label: str
    class_id: int
    source_id: str          # grouping key: images sharing a stem are near-duplicates
    frame: int
    width: int
    height: int
    mode: str
    format: str
    bytes: int
    md5: str
    phash: int              # 64-bit perceptual hash (mean-DCT-free dhash, see _dhash)
    gray_mean: float
    gray_std: float
    lap_var: float          # sharpness proxy
    is_gray: bool


@dataclass
class DatasetInventory:
    """Result of the mandatory *inspect the dataset first* step."""

    root: str
    classes_found: Dict[str, int] = field(default_factory=dict)
    labels_available: List[str] = field(default_factory=list)
    generator_labels_available: List[str] = field(default_factory=list)
    n_images: int = 0
    n_valid: int = 0
    n_corrupt: int = 0
    n_groups: int = 0
    n_duplicate_exact: int = 0
    n_duplicate_near: int = 0
    n_duplicate_perceptual_cross_source: int = 0
    formats: Dict[str, int] = field(default_factory=dict)
    sizes: Dict[str, int] = field(default_factory=dict)
    channels: Dict[str, int] = field(default_factory=dict)
    class_balance: Dict[str, float] = field(default_factory=dict)
    minority_ratio: float = 1.0
    per_class_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def binary(self) -> bool:
        return len(self.classes_found) == 2

    @property
    def multi_source_groups(self) -> bool:
        """True when at least one source id carries more than one image (batch structure)."""
        return self.n_images > self.n_groups

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# --------------------------------------------------------------------------- archive


def ensure_extracted(data_root: Path, archive: Optional[Path], rebuild: bool = False) -> Path:
    """Materialise ``data_root`` from the committed zip archive when needed."""
    has_images = data_root.exists() and any(
        p.suffix.lower() in IMAGE_EXTS for p in data_root.rglob("*") if p.is_file()
    )
    if has_images and not rebuild:
        return data_root
    if archive is None or not Path(archive).exists():
        raise FileNotFoundError(
            f"No images under {data_root} and no archive to extract. "
            "Place the dataset (data/REAL, data/FAKE) or the zip at the configured path."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        all_members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in IMAGE_EXTS]
        for m in all_members:
            parts = Path(m.filename).parts
            if ".." in parts or Path(m.filename).is_absolute():
                raise RuntimeError(f"unsafe zip member {m.filename!r}: archive must contain only relative paths")
        members = [m for m in all_members if m.filename.startswith(("data/", "REAL/", "FAKE/", "real/", "fake/")) or "/" in m.filename]
        if not members:
            members = all_members
        for m in members:
            target = (data_root / m.filename).resolve()
            if data_root.resolve() not in target.parents and target != data_root.resolve():
                raise RuntimeError(f"unsafe zip member {m.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or rebuild:
                target.write_bytes(zf.read(m))
    return data_root


# --------------------------------------------------------------------------- hashing


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash on a resized luminance image (robust to JPEG re-compression)."""
    small = _resize_np(gray, size + 1, size)
    left, right = small[:, :-1], small[:, 1:]
    bits = (left > right).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _resize_np(a: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


# --------------------------------------------------------------------------- scanning


def _label_class_id(name: str, classes: Sequence[str]) -> Optional[int]:
    key = name.lower().replace("-", "_").replace(" ", "_")
    for i, c in enumerate(classes):
        if c == key:
            return i
    return None


def scan_dataset(
    data_root: Path,
    max_images_per_class: Optional[int] = None,
    progress_every: int = 5000,
) -> Tuple[List[ImageRecord], DatasetInventory]:
    """Walk ``data_root``, validate every image and build the inventory.

    Sub-directories define the classes. A directory whose name is not clearly a
    real/ai (or real/fake) indicator is still kept as a class, but is *not* assigned a
    positive/negative meaning - the binary detector then refuses to guess and the run
    is reported as multi-class instead.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    dirs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if not dirs:
        raise RuntimeError(f"no class sub-directories found under {data_root}")

    norm = {d: d.lower().replace("-", "_").replace(" ", "_") for d in dirs}
    classes = sorted(set(norm.values()))
    notes: List[str] = []
    if len(classes) < 2:
        raise RuntimeError(f"need >=2 classes for supervised classification, found {classes}")

    records: List[ImageRecord] = []
    rejected: List[Dict[str, str]] = []
    formats: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    channels: Dict[str, int] = {}
    counts: Dict[str, int] = {c: 0 for c in classes}
    stats: Dict[str, Dict[str, List[float]]] = {c: {"mean": [], "std": [], "lap": [], "bytes": []} for c in classes}

    seen_dir_files: Dict[str, List[Path]] = {}
    for d in dirs:
        cls = norm[d]
        files = sorted(p for p in (data_root / d).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if max_images_per_class:
            files = files[:max_images_per_class]
        seen_dir_files[cls] = files

    total = sum(len(v) for v in seen_dir_files.values())
    done = 0
    for cls in classes:
        files = sorted(seen_dir_files[cls])
        if max_images_per_class:
            files = files[:max_images_per_class]
        for p in files:
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  inspected {done}/{total} images", flush=True)
            rec, err = _inspect_one(p, data_root, cls)
            if rec is None:
                rejected.append({"path": str(p.relative_to(data_root)), "error": err or "unreadable"})
                continue
            records.append(rec)
            formats[rec.format] = formats.get(rec.format, 0) + 1
            key = f"{rec.width}x{rec.height}"
            sizes[key] = sizes.get(key, 0) + 1
            ch = "gray" if rec.is_gray else "rgb"
            channels[ch] = channels.get(ch, 0) + 1
            counts[cls] += 1
            stats[cls]["mean"].append(rec.gray_mean)
            stats[cls]["std"].append(rec.gray_std)
            stats[cls]["lap"].append(rec.lap_var)
            stats[cls]["bytes"].append(rec.bytes)

    # duplicate analysis over the valid records
    exact = 0
    by_md5: Dict[str, List[str]] = {}
    for r in records:
        by_md5.setdefault(r.md5, []).append(r.rel_path)
    for paths in by_md5.values():
        exact += len(paths) - 1
    near, dup_perc = _count_near_duplicates(records)

    labels_available = sorted({c for c in classes if c in POSITIVE_LABELS | NEGATIVE_LABELS})
    if not all(c in POSITIVE_LABELS | NEGATIVE_LABELS for c in classes):
        notes.append(
            "class names are not all recognisable real/ai labels; binary detector will use "
            "the class directory order with the 'fake/ai/generated' class as positive when possible"
        )

    valid_counts = {c: counts[c] for c in classes}
    tot = sum(valid_counts.values()) or 1
    balance = {c: round(v / tot, 4) for c, v in valid_counts.items()}
    minority_ratio = round(min(valid_counts.values()) / max(valid_counts.values()), 4)

    per_class_stats = {}
    for c in classes:
        s = stats[c]
        if not s["mean"]:
            continue
        per_class_stats[c] = {
            "n": len(s["mean"]),
            "gray_mean": round(float(np.mean(s["mean"])), 2),
            "gray_std": round(float(np.mean(s["std"])), 2),
            "laplacian_var_median": round(float(np.median(s["lap"])), 1),
            "bytes_median": round(float(np.median(s["bytes"])), 1),
        }

    n_groups = len({(r.label, r.source_id) for r in records})
    inv = DatasetInventory(
        root=str(data_root),
        classes_found=valid_counts,
        labels_available=labels_available,
        n_images=int(total),
        n_valid=len(records),
        n_corrupt=len(rejected),
        n_groups=n_groups,
        n_duplicate_exact=exact,
        n_duplicate_near=near,
        n_duplicate_perceptual_cross_source=dup_perc,
        formats=formats,
        sizes=sizes,
        channels=channels,
        class_balance=balance,
        minority_ratio=minority_ratio,
        per_class_stats=per_class_stats,
        rejected=rejected[:25],
        notes=notes,
    )
    return records, inv


def _inspect_one(p: Path, root: Path, cls: str) -> Tuple[Optional[ImageRecord], Optional[str]]:
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return None, f"read error: {exc}"
    if len(raw) < 64:
        return None, "file too small / truncated"
    try:
        with Image.open(BytesIO_(raw)) as im:
            im.load()
            fmt = (im.format or p.suffix.lstrip(".")).upper()
            mode, (w, h) = im.mode, im.size
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - any PIL/decoding problem = corrupt image
        return None, f"decode error: {type(exc).__name__}: {exc}"
    if w < 8 or h < 8:
        return None, f"degenerate size {w}x{h}"
    gray = arr.mean(axis=2)
    m = NAME_RE.match(p.name)
    stem = m.group("stem") if m else p.stem
    frame = int(m.group("frame")) if (m and m.group("frame")) else 1
    lap = _laplacian_var(gray)
    is_gray = bool(np.allclose(arr[:, :, 0], arr[:, :, 1], atol=1e-3) and np.allclose(arr[:, :, 1], arr[:, :, 2], atol=1e-3))
    rec = ImageRecord(
        path=str(p),
        rel_path=str(p.relative_to(root)),
        label=cls,
        class_id=0,
        source_id=stem,
        frame=frame,
        width=int(w),
        height=int(h),
        mode=mode,
        format=fmt,
        bytes=len(raw),
        md5=hashlib.md5(raw).hexdigest(),
        phash=_dhash(gray),
        gray_mean=float(gray.mean()),
        gray_std=float(gray.std()),
        lap_var=float(lap),
        is_gray=is_gray,
    )
    return rec, None


def BytesIO_(b: bytes):
    import io

    return io.BytesIO(b)


def _laplacian_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    k = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(k.var())


def _count_near_duplicates(records: List[ImageRecord], hamming_max: int = 2) -> Tuple[int, int]:
    """Near-duplicate frames detected with a 64-bit dhash.

    A full O(N^2) scan over 20k images buys nothing here: images that share a source id
    are the only plausible near-duplicates (frames of one capture), so comparison stays
    inside those buckets. ``hamming_max`` tolerates JPEG-level bit flips. Exact hash
    collisions are also counted across sources, which catches wholesale re-use.
    """
    buckets: Dict[Tuple[str, str], List[ImageRecord]] = {}
    for r in records:
        buckets.setdefault((r.label, r.source_id), []).append(r)

    near = 0
    for grp in buckets.values():
        if len(grp) < 2:
            continue
        hs = sorted(r.phash for r in grp)
        for a, b in zip(hs, hs[1:]):
            if bin(a ^ b).count("1") <= hamming_max:
                near += 1

    # perceptual collisions *across* capture sources: the same picture re-labelled or reused,
    # which the grouped split cannot repair - reported so the report can state it explicitly
    seen: Dict[Tuple[str, int], int] = {}
    for r in records:
        seen[(r.label, r.phash)] = seen.get((r.label, r.phash), 0) + 1
    cross = sum(c - 1 for c in seen.values() if c > 1)
    return int(near), int(cross)


def assign_class_ids(records: List[ImageRecord], classes: Sequence[str]) -> Dict[str, int]:
    """Map class names to integer ids, with the synthetic/'fake' class as positive (1)."""
    positive = [c for c in classes if c in POSITIVE_LABELS]
    negative = [c for c in classes if c in NEGATIVE_LABELS]
    if len(classes) == 2 and len(positive) == 1 and len(negative) == 1:
        mapping = {negative[0]: 0, positive[0]: 1}
    else:
        mapping = {c: i for i, c in enumerate(sorted(classes))}
    for r in records:
        r.class_id = int(mapping[r.label])
    return mapping


def group_key(label: str, source_id: str) -> str:
    """Split-grouping key: frames of one capture must never straddle two splits."""
    return f"{label}/{source_id}"


# --------------------------------------------------------------------------- cache


def build_cache(records: List[ImageRecord], cache_dir: Path, size: int = 64, rebuild: bool = False) -> Path:
    """Decode every image once into a fixed-size grid, stored memmap-friendly.

    ``images.npy`` is a raw uint8 ``[N, size, size, 3]`` array opened with
    ``mmap_mode="r"``: with 20k images that keeps the process at a few hundred MB
    instead of holding two full copies (this box has 3.9 GB and will OOM-kill
    otherwise - which is exactly what the first version of this pipeline did).
    """
    cache_dir = Path(cache_dir)
    img_path = cache_dir / f"images_{size}px.npy"
    meta_path = cache_dir / f"meta_{size}px.npz"
    if img_path.exists() and meta_path.exists() and not rebuild:
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    # open_memmap writes a correct flat header (`|u1`, not a structured descr) and lets the
    # 20k-image array stay on disk: holding two copies in RAM got this process OOM-killed.
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    try:
        for i, r in enumerate(records):
            try:
                with Image.open(r.path) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
                images[i] = _resize_np(arr, size, size).clip(0, 255).astype(np.uint8)
            except Exception:  # noqa: BLE001 - unreadable at cache time = zero image (already flagged)
                images[i] = 0
            if i and i % 5000 == 0:
                print(f"  cached {i}/{n} images", flush=True)
    finally:
        images.flush()
        del images
    np.savez_compressed(
        meta_path,
        labels=np.array([r.label for r in records]),
        source_ids=np.array([r.source_id for r in records]),
        frames=np.array([r.frame for r in records], dtype=np.int32),
        paths=np.array([r.rel_path for r in records]),
    )
    return cache_dir


def load_cache(cache_dir: Path, size_hint: int = 64) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    cand = sorted(cache_dir.glob("images_*px.npy"))
    if not cand:
        raise FileNotFoundError(f"no cache in {cache_dir}; run the prepare step")
    img_path = cand[0]
    size = int(img_path.name.replace("images_", "").replace("px.npy", ""))
    images = np.load(img_path, mmap_mode="r")
    meta = dict(np.load(cache_dir / f"meta_{size}px.npz", allow_pickle=True))
    return images, meta


def summarize(records: Iterable[ImageRecord]) -> Dict[str, float]:
    recs = list(records)
    if not recs:
        return {}
    return {
        "mean_gray": float(np.mean([r.gray_mean for r in recs])),
        "mean_std": float(np.mean([r.gray_std for r in recs])),
        "median_lap_var": float(np.median([r.lap_var for r in recs])),
    }
