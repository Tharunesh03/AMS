"""Grad-CAM saliency from the *selected* model only (spec: never a fabricated map).

    input -> trained model -> last conv-stage activation -> d(positive-class logit)/dA
          -> weights = global mean of dA -> ReLU(sum_c w_c * A_c) -> upsample -> overlay

For a classical (non-convolutional) production model there is no activation to
attribute, so :class:`GradCAM` reports ``available=False`` with a reason and the UI
says so plainly instead of drawing a fake heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTS = (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU, nn.Sigmoid, nn.Tanh)


def find_cam_targets(model: nn.Module, keep: int = 4) -> List[Tuple[str, nn.Module]]:
    """Candidate attribution points: the activation right after each Conv2d, deepest last.

    The deepest stage is preferred, but at small inputs the final map can be 2x2, which
    gives a useless heatmap - the caller picks the deepest candidate that still has
    enough spatial resolution (>= ``min_grid``), which is why several are returned.
    """
    items = [(n, m) for n, m in model.named_modules()]
    out: List[Tuple[str, nn.Module]] = []
    for i, (n, m) in enumerate(items):
        if not isinstance(m, nn.Conv2d):
            continue
        target: Optional[Tuple[str, nn.Module]] = None
        for n2, m2 in items[i + 1 :]:
            if isinstance(m2, nn.Conv2d):
                break                                  # stop at the next conv stage
            if isinstance(m2, _ACTS) or (
                isinstance(m2, nn.BatchNorm2d) and m2.num_features == m.out_channels
            ):
                target = (n2, m2)                      # keep looking: prefer the last activation in the block
        if target is None:
            target = (n, m)
        if target[0] and all(target[0] != t[0] for t in out):
            out.append(target)
    return out[-keep:] if out else []


@dataclass
class GradCamResult:
    available: bool
    heatmap: Optional[np.ndarray] = None   # [H, W] float in [0, 1], model input grid
    overlay: Optional[np.ndarray] = None   # uint8 RGB overlay of the same size
    rgb: Optional[np.ndarray] = None       # uint8 viewable input
    target_layer: Optional[str] = None
    reason: str = ""
    variant: str = "grad-cam (ReLU of the gradient-weighted activations)"
    map_size: Tuple[int, int] = (0, 0)
    low_resolution_map: bool = False

    def to_dict(self) -> dict:
        d = {"available": self.available, "target_layer": self.target_layer, "reason": self.reason,
             "method": self.variant, "activation_map_size": list(self.map_size),
             "low_resolution_map": bool(self.low_resolution_map)}
        if self.heatmap is not None:
            h = self.heatmap
            d["peak"] = [float(h.max()), int(np.unravel_index(int(h.argmax()), h.shape)[0]), int(np.unravel_index(int(h.argmax()), h.shape)[1])]
            d["mass_in_top_quartile"] = float((h >= np.quantile(h, 0.75)).mean())
            d["spread"] = float(h.std())
        return d


class GradCAM:
    """Grad-CAM for the loaded production detector (positive class = logit index 0)."""

    def __init__(self, model: nn.Module, device: str = "cpu", min_grid: int = 5) -> None:
        self.model = model.eval()
        self.device = device
        self.cands = find_cam_targets(model)
        self.min_grid = int(min_grid)
        self.target_layer = self.cands[-1][0] if self.cands else None
        self.map_size: Tuple[int, int] = (0, 0)
        self.low_resolution_map = False

    def __call__(self, x: torch.Tensor, class_index: Optional[int] = None) -> GradCamResult:
        if not self.cands:
            return GradCamResult(False, reason="the selected model has no convolutional activation to attribute")
        x = x.to(self.device)
        if not x.requires_grad:
            # a serving model has its parameters frozen for speed; input gradients still work
            # as long as the input itself is a leaf that requires grad
            x = x.detach().requires_grad_(True)
        cap: dict = {}

        def mk(name):
            def hook(_m, _i, out):
                cap[name] = out
            return hook

        handles = [mod.register_forward_hook(mk(name)) for name, mod in self.cands]
        try:
            with torch.enable_grad():
                out = self.model(x)
                if out.ndim == 2 and out.shape[1] == 1:
                    score = out[:, 0].sum()            # positive class = "ai / synthetic"
                elif out.ndim == 2:
                    idx = out.shape[1] - 1 if class_index is None else int(class_index)
                    score = out[:, idx].sum()
                else:
                    score = out.sum()
                usable = []
                for name, _ in self.cands:
                    a = cap.get(name)
                    if a is None or not torch.is_tensor(a) or a.ndim != 4:
                        continue
                    usable.append((name, a))
                if not usable:
                    return GradCamResult(False, reason="activation hooks captured no 4-D feature maps")
                # deepest candidate whose map is still spatially meaningful; if the deepest
                # stages are 2x2 (tiny inputs), attribute at the coarsest map that is not
                self.low_resolution_map = False
                name, act = usable[-1]
                for n2, a2 in reversed(usable):
                    if min(a2.shape[-2:]) >= self.min_grid:
                        name, act = n2, a2
                        break
                else:
                    n2, a2 = max(usable, key=lambda na: min(na[1].shape[-2:]))
                    name, act = n2, a2
                    self.low_resolution_map = True
                self.target_layer = name
                self.map_size = [int(act.shape[-2]), int(act.shape[-1])]
                grad = torch.autograd.grad(score, act, retain_graph=False)[0]
        finally:
            for h in handles:
                h.remove()

        w = grad.mean(dim=(2, 3), keepdim=True)
        weighted = (w * act).sum(dim=1, keepdim=True)
        cam_t = F.relu(weighted)
        variant = "grad-cam (ReLU of the gradient-weighted activations)"
        cam = cam_t[0, 0].detach().double().cpu().numpy()
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(cam.min()), float(cam.max())
        if hi <= lo + 1e-8:
            # all responses negative at this layer: report the signed map instead of nothing,
            # and say so - the map is still this model's own gradient, just not ReLU-ed.
            cam = weighted[0, 0].detach().double().cpu().numpy()
            cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = float(cam.min()), float(cam.max())
            variant = "signed gradient-weighted map (ReLU removed every response at this layer)"
            if hi <= lo + 1e-8:
                return GradCamResult(False, target_layer=self.target_layer,
                                     reason="class score is not spatially dependent at this layer "
                                            "(the gradient-weighted activation map is flat)")
        cam = (cam - lo) / (hi - lo)
        H, W = int(x.shape[-2]), int(x.shape[-1])
        big = F.interpolate(torch.from_numpy(cam).float()[None, None], size=(H, W), mode="bilinear", align_corners=False)
        big = np.clip(big[0, 0].numpy(), 0, 1)
        rgb = _viewable_rgb(x)
        res = GradCamResult(True, heatmap=big, overlay=_overlay(rgb, big), rgb=rgb,
                            target_layer=self.target_layer)
        res.variant = variant                                   # type: ignore[attr-defined]
        res.map_size = tuple(self.map_size)                       # type: ignore[attr-defined]
        res.low_resolution_map = bool(self.low_resolution_map)    # type: ignore[attr-defined]
        return res


def _viewable_rgb(x: torch.Tensor) -> np.ndarray:
    """Render the (normalised) input back to 0-255 for display: per-tensor min/max stretch."""
    t = x.detach().cpu().float()
    t = t - t.amin()
    t = t / t.amax().clamp_min(1e-6)
    t = t[0]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def _overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.clip(img.astype(np.float32) * (1 - alpha) + _jet(heat) * alpha, 0, 255).astype(np.uint8)


def _jet(h: np.ndarray) -> np.ndarray:
    h = np.clip(h, 0, 1)
    r = np.clip(1.6 * h - 0.55, 0, 1)
    g = np.clip(1.6 * (h - 0.25), 0, 1) * np.clip(2.6 - 1.6 * h, 0, 1)
    b = np.clip(1.2 - 1.6 * h, 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def cam_bbox(heat: np.ndarray, frac: float = 0.5) -> Optional[List[float]]:
    """Bounding box covering the top ``frac`` of saliency mass (0..1 coords)."""
    if heat is None or heat.size == 0:
        return None
    thr = np.quantile(heat, 1.0 - frac)
    mask = heat >= max(thr, heat.min() + 1e-6)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = heat.shape
    return [float(xs.min()) / w, float(ys.min()) / h, float(xs.max() + 1) / w, float(ys.max() + 1) / h]
