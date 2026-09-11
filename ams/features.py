"""Hand-designed forensic descriptors used by the classical (sklearn) candidates.

Everything measured here is image physics, not a learned representation:

* ``luma_*`` / ``col_*``   tone, channel-coupling and saturation statistics;
* ``hp_*``               high-pass residual - generators leave a different micro-texture
                          than camera sensor noise;
* ``noise_floor_std``    second-order residual energy (flat-region noise estimate);
* ``block_*``            8x8 grid boundary energy - JPEG / VAE decoding signature;
* ``spec_*``             radial log-power spectrum - upscaling leaves spectral peaks;
* ``lbp_*``              uniform local-binary-pattern histogram - local micro-structure;
* ``hog_*``              histogram of oriented gradients - edge/contour statistics.

``photometric="gray"`` z-scores every image, which deliberately destroys the global
brightness/contrast cue (this corpus has a strong one - see ``reports/eda.md``), so we
can tell a real texture detector from a mean-pixel cheat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["FeatureOptions", "extract_features"]


@dataclass
class FeatureOptions:
    photometric: str = "gray"   # none | gray (per-image z-score)
    use_hog: bool = True
    use_lbp: bool = True
    use_spectrum: bool = True
    use_color: bool = True
    use_block: bool = True
    grid: int = 8               # HOG cells per side
    bins: int = 9               # HOG orientation bins
    spec_bins: int = 16
    chunk: int = 512

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.__dict__.items()))


# --------------------------------------------------------------------- primitives


def _resize_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / np.float32(255.0)


def _gray(x: np.ndarray) -> np.ndarray:
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def _box_blur(g: np.ndarray, r: int = 1) -> np.ndarray:
    """Mean filter with reflect padding (exact, 9 slice adds for r=1)."""
    n, h, w = g.shape
    p = np.pad(g, ((0, 0), (r, r), (r, r)), mode="reflect")
    out = np.zeros_like(g)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[:, dy : dy + h, dx : dx + w]
    return out / float((2 * r + 1) ** 2)


def _moments(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = a - a.mean(axis=1, keepdims=True)
    v = (a**2).mean(axis=1) + 1e-9
    return (a**3).mean(axis=1) / v**1.5, (a**4).mean(axis=1) / v**2 - 3.0


def _entropy(hist: np.ndarray) -> np.ndarray:
    p = hist / (hist.sum(axis=1, keepdims=True) + 1e-9)
    return -(p * np.log(p + 1e-9)).sum(axis=1)


def _histogram(a: np.ndarray, bins: int) -> np.ndarray:
    """Per-image tone histogram (robust 0.1-99.9 percentile range).

    Quantiles are computed per image, never per batch - a batch-level range would make a
    feature value depend on which other images happened to be in the chunk.
    """
    flat = a.reshape(a.shape[0], -1)
    lo = np.quantile(flat, 0.001, axis=1, keepdims=True)
    hi = np.quantile(flat, 0.999, axis=1, keepdims=True)
    idx = np.clip(((flat - lo) / (hi - lo + 1e-6) * bins).astype(np.int64), 0, bins - 1)
    rows = np.repeat(np.arange(flat.shape[0]), flat.shape[1])
    out = np.zeros((flat.shape[0], bins), dtype=np.float32)
    np.add.at(out.reshape(-1), rows * bins + idx.reshape(-1), 1.0)
    return out / (out.sum(axis=1, keepdims=True) + 1e-6)


_LBP_TABLE: Optional[np.ndarray] = None


def _lbp_table() -> np.ndarray:
    global _LBP_TABLE
    if _LBP_TABLE is None:
        bits = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.int8)
        trans = (np.diff(np.concatenate([bits, bits[:, :1]], axis=1), axis=1) != 0).sum(axis=1)
        table = np.full(256, 59, dtype=np.int64)  # non-uniform bucket
        uniform = np.flatnonzero(trans <= 2)
        table[uniform] = np.arange(len(uniform), dtype=np.int64)
        _LBP_TABLE = table
    return _LBP_TABLE


def _lbp_hist(g: np.ndarray) -> np.ndarray:
    """Uniform-LBP histogram (59 uniform patterns + 1 non-uniform bucket), row-normalised."""
    center = g[:, 1:-1, 1:-1]
    nb = np.stack(
        [
            g[:, :-2, :-2], g[:, :-2, 1:-1], g[:, :-2, 2:],
            g[:, 1:-1, 2:], g[:, 2:, 2:], g[:, 2:, 1:-1],
            g[:, 2:, :-2], g[:, 1:-1, :-2],
        ],
        axis=-1,
    )
    code = ((nb >= center[..., None]).astype(np.int32) << np.arange(8, dtype=np.int32)).sum(axis=-1)
    uni = _lbp_table()[code.reshape(-1)]
    n = g.shape[0]
    out = np.zeros((n, 60), dtype=np.float32)
    per_img = code.size // n
    rows = np.repeat(np.arange(n, dtype=np.int64), per_img)
    np.add.at(out.reshape(-1), rows * 60 + uni, 1.0)
    return out / (out.sum(axis=1, keepdims=True) + 1e-6)


def _hog(g: np.ndarray, grid: int, bins: int) -> np.ndarray:
    """Global HOG: ``grid x grid`` cells x ``bins`` unsigned orientations, L2-normalised."""
    n, h, w = g.shape
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.arctan2(gy, gx) + np.pi                     # [0, 2pi)
    ori = (ori % np.pi) / (np.pi / bins)                  # unsigned, [0, bins)
    bo = np.floor(ori).astype(np.int32).clip(0, bins - 1)
    py, px = np.indices((h, w))
    cell = (((py * grid) // h) * grid + ((px * grid) // w)).astype(np.int32)  # cell id per pixel
    flat = (bo * (grid * grid) + cell[None]).reshape(n, -1)
    out = np.zeros((n, bins * grid * grid), dtype=np.float32)
    rows = np.repeat(np.arange(n, dtype=np.int64), h * w)
    np.add.at(out.reshape(-1), rows * out.shape[1] + flat.reshape(-1), mag.reshape(-1))
    out = np.sqrt(out)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)


def _radial_spectrum(g: np.ndarray, bins: int) -> np.ndarray:
    n, h, w = g.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.rfft2(g * win[None], axes=(1, 2))
    p = np.log1p(np.abs(f))
    yy, xx = np.meshgrid(np.arange(h) - h // 2, np.arange(w // 2 + 1) - w // 2, indexing="ij")
    r = np.sqrt(yy**2 + xx**2)
    edges = np.quantile(r.ravel(), np.linspace(0, 1, bins + 1))
    out = np.zeros((n, bins), dtype=np.float32)
    for i in range(bins):
        m = (r >= edges[i]) & (r < edges[i + 1] + 1e-6)
        if m.any():
            out[:, i] = p[:, m].mean(axis=1)
    return out - out.mean(axis=1, keepdims=True)


def _blockiness(x: np.ndarray, period: int = 8) -> np.ndarray:
    """Boundary-vs-interior gradient energy on the 8-pixel coding grid."""
    gx = np.abs(np.diff(x, axis=2))   # vertical edges:  [N,H,W-1]
    gy = np.abs(np.diff(x, axis=1))   # horizontal edges: [N,H-1,W]
    cols = gx.shape[2]
    rows = gy.shape[1]
    on_c = np.arange(period - 1, cols, period)
    on_r = np.arange(period - 1, rows, period)
    off_c = np.setdiff1d(np.arange(cols), on_c)
    off_r = np.setdiff1d(np.arange(rows), on_r)
    on_v = gx[:, :, on_c].mean((1, 2))
    off_v = gx[:, :, off_c].mean((1, 2))
    on_h = gy[:, on_r, :].mean((1, 2))
    off_h = gy[:, off_r, :].mean((1, 2))
    return np.stack([on_v, off_v, on_h, off_h, on_v / (off_v + 1e-6), on_h / (off_h + 1e-6)], axis=1)


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    if a.ndim > 2:
        return a.reshape(a.shape[0], -1)
    return a


# --------------------------------------------------------------------- public api


def extract_features(
    images: np.ndarray,
    opts: Optional[FeatureOptions] = None,
    progress: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the forensic descriptor matrix for ``images`` (``[N,H,W,3]`` uint8)."""
    opts = opts or FeatureOptions()
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected [N,H,W,3] uint8, got {images.shape}")
    n = images.shape[0]
    blocks: List[np.ndarray] = []
    names: List[str] = []
    for start in range(0, n, opts.chunk):
        end = min(start + opts.chunk, n)
        xb = _resize_float(images[start:end])
        if opts.photometric == "gray":
            m = xb.mean(axis=(1, 2, 3), keepdims=True)
            s = xb.std(axis=(1, 2, 3), keepdims=True) + 1e-5
            xb = (xb - m) / s
        g = _gray(xb)
        feats: List[np.ndarray] = []
        meta: List[str] = []

        gf = g.reshape(g.shape[0], -1)
        skew, kurt = _moments(gf)
        hist = _histogram(g, 12)
        feats += [
            np.stack([gf.mean(1), gf.std(1), skew, kurt], axis=1),
            hist,
            _entropy(hist)[:, None],
        ]
        meta += ["luma_mean", "luma_std", "luma_skew", "luma_kurtosis"]
        meta += [f"luma_hist_{i}" for i in range(hist.shape[1])]
        meta += ["luma_entropy"]

        hp = g - _box_blur(g, 1)
        hpf = hp.reshape(hp.shape[0], -1)
        hp_skew, hp_kurt = _moments(hpf)
        feats.append(np.stack([hpf.std(1), np.abs(hpf).mean(1), hp_kurt, hp_skew], axis=1))
        meta += ["hp_std", "hp_meanabs", "hp_kurtosis", "hp_skew"]

        e_h = np.abs(np.diff(g, axis=2)).mean((1, 2))
        e_v = np.abs(np.diff(g, axis=1)).mean((1, 2))
        d1 = np.abs(g[:, 2:, 2:] - g[:, :-2, :-2]).mean((1, 2))
        d2 = np.abs(g[:, 2:, :-2] - g[:, :-2, 2:]).mean((1, 2))
        feats.append(np.stack([e_h, e_v, d1, d2], axis=1))
        meta += ["edge_h", "edge_v", "edge_d1", "edge_d2"]
        feats.append((np.log(e_h + 1e-4) - np.log(e_v + 1e-4))[:, None])
        meta += ["edge_dir_asym"]

        hp2 = hp - _box_blur(hp, 1)
        feats.append(hp2.reshape(hp2.shape[0], -1).std(1)[:, None])
        meta += ["noise_floor_std"]

        if opts.use_block:
            feats.append(np.log(_blockiness(g, 8) + 1e-4))  # block grid on luminance
            meta += ["block_on_v", "block_off_v", "block_on_h", "block_off_h", "block_ratio_v", "block_ratio_h"]

        if opts.use_spectrum:
            s = _radial_spectrum(g, opts.spec_bins)
            feats.append(s)
            meta += [f"spec_{i}" for i in range(s.shape[1])]

        if opts.use_lbp:
            l = _lbp_hist(g)
            feats.append(l)
            meta += [f"lbp_{i}" for i in range(l.shape[1])]

        if opts.use_hog:
            h = _hog(g, opts.grid, opts.bins)
            feats.append(h)
            meta += [f"hog_{i}" for i in range(h.shape[1])]

        if opts.use_color:
            ch = xb.reshape(xb.shape[0], 3, -1)
            mx = xb.max(axis=3).mean(axis=2)
            mn = xb.min(axis=3).mean(axis=2)
            sat = 1.0 - mn / (mx + 1e-4)
            feats.append(
                np.stack(
                    [
                        ch[:, 0].mean(1), ch[:, 1].mean(1), ch[:, 2].mean(1),
                        ch[:, 0].std(1), ch[:, 1].std(1), ch[:, 2].std(1),
                        (ch[:, 0] - ch[:, 1]).mean(1),
                        (ch[:, 1] - ch[:, 2]).mean(1),
                        np.abs(ch[:, 0] - ch[:, 1]).std(1),
                        np.abs(ch[:, 1] - ch[:, 2]).std(1),
                        sat.mean(1), sat.std(1), _entropy(_histogram(sat[:, :, None], 6)),
                    ],
                    axis=1,
                )
            )
            meta += [
                "col_r_mean", "col_g_mean", "col_b_mean", "col_r_std", "col_g_std", "col_b_std",
                "col_rg_diff", "col_gb_diff", "col_rg_std", "col_gb_std", "sat_mean", "sat_std", "sat_entropy",
            ]

        X = np.concatenate([_as2d(f) for f in feats], axis=1).astype(np.float32)
        blocks.append(X)
        if not names:
            names = meta
        if progress:
            print(f"  features {end}/{n}", flush=True)

    out = np.concatenate(blocks, axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape[1] == len(names), f"feature count {out.shape[1]} != names {len(names)}"
    return out, names
