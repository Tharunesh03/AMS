"""Preprocessing/transform tests, including the memmap boundary that broke training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vfa.transforms import AugmentConfig, ImageTransform, dataset_stats, resize_any, to_chw_float01


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- memmap boundary
def test_to_chw_float01_accepts_read_only_memmap_slices(tmp_path):
    """Regression: a read-only memmap view used to fail inside torch.from_numpy with
    "can't convert np.ndarray of type numpy.void", which killed the training run."""
    path = tmp_path / "arr.npy"
    src = (_rng(1).random((16, 8, 8, 3)) * 255).astype(np.uint8)
    np.save(path, src)
    mm = np.load(path, mmap_mode="r")
    assert not mm.flags.writeable

    t = to_chw_float01(mm[3:7])
    assert t.shape == (4, 3, 8, 8) and t.dtype == torch.float32
    ref = src[3:7].astype(np.float32) / 255.0
    assert np.allclose(t.permute(0, 2, 3, 1).numpy(), ref, atol=1e-6)


def test_to_chw_float01_input_forms():
    a = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    t = to_chw_float01(a)
    assert t.shape == (2, 3, 4, 4) and float(t.max()) <= 1.0
    single = to_chw_float01(a[0])                       # HWC without a batch axis
    assert single.shape == (1, 3, 4, 4)
    assert torch.allclose(single[0], t[0])
    already = to_chw_float01(t)                          # tensors pass through
    assert torch.allclose(already, t)
    grey = to_chw_float01(np.zeros((3, 5, 5, 1), dtype=np.uint8))
    assert grey.shape == (3, 1, 5, 5)


def test_resize_any_is_identity_at_the_right_size():
    x = torch.rand(2, 3, 64, 64)
    assert resize_any(x, 64) is x
    assert resize_any(x, 16).shape == (2, 3, 16, 16)


def test_resize_any_upsamples_and_stays_in_range():
    small = to_chw_float01((_rng(2).random((6, 12, 12, 3)) * 255).astype(np.uint8))
    out = resize_any(small, 24)
    assert out.shape == (6, 3, 24, 24)
    assert -1e-6 <= float(out.min()) and float(out.max()) <= 1.0 + 1e-6


# --------------------------------------------------------------------- transform
def test_normalisation_is_zero_mean_unit_variance_over_the_batch():
    a = (_rng(3).random((64, 24, 24, 3)) * 255).astype(np.uint8)
    mean, std = dataset_stats(a, chunk=16, size=24)
    tf = ImageTransform(24, mean, std)
    x = tf(a)
    assert x.shape == (64, 3, 24, 24)
    assert abs(float(x.mean())) < 0.05 and abs(float(x.std()) - 1.0) < 0.25


def test_transform_rejects_non_positive_std():
    with pytest.raises(ValueError):
        ImageTransform(16, [0.5, 0.5, 0.5], [0.0, 1.0, 1.0])


def test_inference_transform_never_augments_even_when_a_config_is_given():
    a = (_rng(4).random((8, 16, 16, 3)) * 255).astype(np.uint8)
    eval_tf = ImageTransform(16, [0.5] * 3, [0.5] * 3, train=False, augment=AugmentConfig())
    x1, x2 = eval_tf(a), eval_tf(a)
    assert torch.equal(x1, x2), "evaluation must be deterministic"
    assert eval_tf.augment.enabled is False


def test_train_transform_is_deterministic_per_seed_and_changes_the_data():
    a = (_rng(5).random((16, 32, 32, 3)) * 255).astype(np.uint8)
    cfg = AugmentConfig()
    t1 = ImageTransform(32, [0.5] * 3, [0.5] * 3, train=True, augment=cfg, seed=11)
    t2 = ImageTransform(32, [0.5] * 3, [0.5] * 3, train=True, augment=cfg, seed=11)
    t3 = ImageTransform(32, [0.5] * 3, [0.5] * 3, train=True, augment=cfg, seed=12)
    x1, x2, x3 = t1(a), t2(a), t3(a)
    assert x1.shape == (16, 3, 32, 32)
    assert torch.allclose(x1, x2), "same seed must reproduce the same augmentation"
    assert not torch.allclose(x1, x3), "different seed must explore a different augmentation"
    plain = ImageTransform(32, [0.5] * 3, [0.5] * 3)(a)
    assert not torch.allclose(x1, plain)


def test_grayscale_path_folds_the_channel_statistics():
    a = (_rng(6).random((8, 20, 20, 3)) * 255).astype(np.uint8)
    tf = ImageTransform(20, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], to_gray=True)
    x = tf(a)
    assert x.shape == (8, 1, 20, 20)
    # one channel in, three expected by the statistics -> folded luma normalisation, no broadcast error
    assert torch.isfinite(x).all()
    rep = ImageTransform(20, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], to_gray=False)(a)
    assert rep.shape == (8, 3, 20, 20)
    assert not torch.allclose(x.sum(1), rep.mean(1))       # different normalisation, not a re-labelling


def test_augmented_values_stay_bounded():
    a = (_rng(7).random((24, 32, 32, 3)) * 255).astype(np.uint8)
    tf = ImageTransform(32, [0.5] * 3, [0.5] * 3, train=True, augment=AugmentConfig(), seed=3)
    x = tf(a)
    # post-normalisation range is unbounded by definition, but nothing may blow up
    assert torch.isfinite(x).all() and float(x.abs().max()) < 20.0


def test_transform_spec_documents_the_pipeline():
    tf = ImageTransform(64, [0.5] * 3, [0.25] * 3, train=True, augment=AugmentConfig())
    s = tf.spec()
    assert s["size"] == 64 and s["to_gray"] is False
    assert s["augment"]["hflip_p"] == pytest.approx(0.5)
    assert set(["cutout", "jpeg_quant_p", "noise_std"]) <= set(s["augment"])


def test_augment_config_none_disables_everything():
    a = AugmentConfig.none()
    assert a.enabled is False
    b = AugmentConfig()
    assert b.enabled is True and b.cutout > 0


def test_dataset_stats_matches_a_direct_computation():
    imgs = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    imgs[0, :, :, 0] = 255
    (mean, std) = dataset_stats(imgs, chunk=2, size=8)
    ref = imgs.astype(np.float32) / 255.0                      # [N,H,W,C] -> statistics per C
    assert mean == pytest.approx(tuple(ref.mean(axis=(0, 1, 2)).tolist()), abs=1e-5)
    assert std == pytest.approx(tuple(ref.std(axis=(0, 1, 2)).tolist()), abs=1e-4)


def test_dataset_stats_chunks_agree_with_full_batch():
    imgs = (_rng(8).random((40, 16, 16, 3)) * 255).astype(np.uint8)
    m1, s1 = dataset_stats(imgs, chunk=7, size=16)
    m2, s2 = dataset_stats(imgs, chunk=40, size=16)
    assert m1 == pytest.approx(m2, abs=1e-5) and s1 == pytest.approx(s2, abs=1e-5)
