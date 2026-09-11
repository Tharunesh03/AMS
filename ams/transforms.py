"""Preprocessing + training augmentation, done with batched torch ops.

Pipeline (identical in training and inference apart from the augmentation block)::

    uint8 [B,H,W,3] -> resize -> float [0,1] -> CHW -> augment (train only)
                    -> channel normalisation (mean/std from the *training* split)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    enabled: bool = True
    hflip_p: float = 0.5
    shift_px: int = 2
    rotate_deg: float = 6.0
    scale: float = 0.06
    brightness: float = 0.12
    contrast: float = 0.15
    noise_std: float = 0.02
    jpeg_quant_p: float = 0.35
    jpeg_levels: int = 12
    cutout: int = 6

    @staticmethod
    def none() -> "AugmentConfig":
        return AugmentConfig(enabled=False)


def resize_any(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    antialias = mode != "nearest"
    return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=antialias)


def to_chw_float01(x) -> torch.Tensor:
    """uint8 HWC (numpy or tensor) -> float CHW in [0, 1].

    Memmap-backed arrays are copied first: ``np.asarray`` on a read-only memmap yields a
    view that ``torch.from_numpy`` refuses (it saw a void dtype on this box).

    A float tensor that is already ``[N,C,H,W]`` is passed through unchanged (channels are
    detected by which axis is 1-4 wide), so the function is idempotent.
    """
    if not torch.is_tensor(x):
        x = np.ascontiguousarray(x)
        if not x.flags.writeable or isinstance(x, np.memmap) or x.dtype.kind == "V":
            x = np.array(x, copy=True)
        x = torch.from_numpy(np.ascontiguousarray(x))
    was_uint8 = x.dtype == torch.uint8
    if was_uint8:
        x = x.to(torch.float32) / 255.0
    if x.ndim == 3:  # HWC single image
        x = x.permute(2, 0, 1)[None]
    elif x.ndim == 4 and _is_hwc_batch(x, was_uint8):
        x = x.permute(0, 3, 1, 2)
    return x.contiguous()


def _is_hwc_batch(x: torch.Tensor, was_uint8: bool) -> bool:
    """Decide whether a 4-d batch is ``[N,H,W,C]`` (True) or already ``[N,C,H,W]``.

    Channels are 1, 3 or 4 wide, so the axis width usually disambiguates. When both axes
    look like channels (e.g. a 3x3x3 batch) raw uint8 input is treated as HWC, which is the
    layout every caller in this project produces from the decode cache.
    """
    c_last = int(x.shape[-1]) in (1, 3, 4)
    c_first = int(x.shape[1]) in (1, 3, 4)
    if c_last and not c_first:
        return True
    if c_first and not c_last:
        return False
    return bool(was_uint8)


class ImageTransform:
    """Callable that maps a batch of uint8 HWC images to a normalised CHW float batch."""

    def __init__(
        self,
        size: int,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool = False,
        augment: Optional[AugmentConfig] = None,
        to_gray: bool = False,
        seed: int = 0,
    ) -> None:
        self.size = int(size)
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)
        if self.std.min() <= 0:
            raise ValueError("std must be positive")
        self.train = bool(train)
        self.augment = augment if self.train else AugmentConfig.none()
        self.to_gray = bool(to_gray)
        self._gen = torch.Generator().manual_seed(int(seed))

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self._gen)

    def _normal(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self._gen)

    # --------------------------------------------------------------------- call
    def __call__(self, x) -> torch.Tensor:
        x = to_chw_float01(x)
        x = resize_any(x, self.size)
        if self.to_gray and x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean, std = self.mean, self.std
        if x.shape[1] == 1 and mean.shape[1] == 3:
            w = torch.tensor([0.299, 0.587, 0.114], dtype=mean.dtype).view(1, 3, 1, 1)
            mean = (mean * w).sum(1, keepdim=True)
            std = (std * w).sum(1, keepdim=True).clamp_min(1e-4)
        elif x.shape[1] == 3 and mean.shape[1] == 1:
            mean, std = mean.repeat(1, 3, 1, 1), std.repeat(1, 3, 1, 1)
        if self.augment is not None and self.augment.enabled:
            x = self._augment(x, self.augment)
        return (x - mean) / std

    def _augment(self, x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
        n = x.shape[0]
        if a.hflip_p > 0:
            keep = self._rand(n) < a.hflip_p
            if keep.any():
                x = torch.where(keep.view(-1, 1, 1, 1), x.flip(-1), x)
        if a.rotate_deg > 0 or a.shift_px > 0 or a.scale > 0:
            ang = (self._normal(n) * a.rotate_deg).clamp(-3.0 * a.rotate_deg, 3.0 * a.rotate_deg) if a.rotate_deg else torch.zeros(n)
            rad = ang * math.pi / 180.0
            sc = 1.0 + (self._normal(n) * a.scale).clamp(-0.5, 0.5) if a.scale else torch.ones(n)
            tx = self._normal(n) * (a.shift_px / max(x.shape[-1], 1)) if a.shift_px else torch.zeros(n)
            ty = self._normal(n) * (a.shift_px / max(x.shape[-2], 1)) if a.shift_px else torch.zeros(n)
            theta = torch.stack(
                [torch.cos(rad) / sc, -torch.sin(rad) / sc, tx, torch.sin(rad) / sc, torch.cos(rad) / sc, ty],
                dim=1,
            ).view(n, 2, 3)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        if a.brightness > 0:
            x = x * (1.0 + self._normal(n, 1, 1, 1) * a.brightness)
        if a.contrast > 0:
            c = x.mean(dim=(1, 2, 3), keepdim=True)
            x = c + (x - c) * (1.0 + self._normal(n, 1, 1, 1) * a.contrast)
        if a.noise_std > 0:
            x = x + self._normal(x.shape) * a.noise_std
        if a.jpeg_quant_p > 0:
            mask = (self._rand(n) < a.jpeg_quant_p).view(-1, 1, 1, 1)
            q = (x * a.jpeg_levels).round() / a.jpeg_levels
            x = torch.where(mask, q, x)
        x = x.clamp(0.0, 1.0)
        if a.cutout and a.cutout < x.shape[-1]:
            h, w = x.shape[-2:]
            cy = (self._rand(n) * h).long()
            cx = (self._rand(n) * w).long()
            r = a.cutout // 2
            yy = torch.arange(h).view(1, 1, h, 1)
            xx = torch.arange(w).view(1, 1, 1, w)
            m = ((yy - cy.view(-1, 1, 1, 1)).abs() <= r) & ((xx - cx.view(-1, 1, 1, 1)).abs() <= r)
            x = torch.where(m, torch.zeros_like(x), x)
        return x

    # --------------------------------------------------------------------- meta
    def spec(self) -> dict:
        return {
            "size": self.size,
            "mean": [float(v) for v in self.mean.flatten()],
            "std": [float(v) for v in self.std.flatten()],
            "to_gray": self.to_gray,
            "augment": asdict(self.augment) if self.augment else None,
        }


def dataset_stats(images, chunk: int = 4096, size: int = 64) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Per-channel mean/std of the (training) split, computed on the model input grid."""
    n = images.shape[0]
    tot = torch.zeros(3)
    tot2 = torch.zeros(3)
    cnt = 0
    for s in range(0, n, chunk):
        x = to_chw_float01(np.asarray(images[s : s + chunk]) if not torch.is_tensor(images) else images[s : s + chunk])
        x = resize_any(x, size)
        v = x.mean(dim=(0, 2, 3))
        tot += v * (x.shape[0])
        tot2 += (x * x).mean(dim=(0, 2, 3)) * x.shape[0]
        cnt += x.shape[0]
    mean = tot / cnt
    var = (tot2 / cnt - mean**2).clamp_min(1e-8)
    return tuple(mean.tolist()), tuple(var.sqrt().tolist())

