"""Training of every supervised candidate on identical data and identical splits.

Two model families share one protocol so the comparison is fair:

* classical - forensic descriptor -> sklearn pipeline (logistic / linear SVM / RF / GBDT / k-NN)
* deep      - raw pixels -> LightCNN or a torchvision backbone (ImageNet transfer when the
              weights are reachable; recorded honestly as ``pretrained_applied`` when not)

Protocol: threshold is chosen on the **validation** split only, the **test** split is
touched exactly once after selection, and everything in the comparison table
(metrics, ms/image, artifact MB) is measured on this machine, never assumed.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import (
    DatasetInventory,
    assign_class_ids,
    build_cache,
    ensure_extracted,
    group_key,
    load_cache,
    scan_dataset,
)
from .features import FeatureOptions, extract_features
from .metrics import binary_metrics, multiclass_metrics, select_threshold
from .models import ModelSpec, build_model
from .selection import Candidate
from .splitting import Split, stratified_grouped_split


# --------------------------------------------------------------------------- data
@dataclass
class DataBundle:
    images: np.ndarray           # [N, S, S, 3] uint8, cached at S = cache_size
    labels: np.ndarray           # int class ids
    groups: np.ndarray           # capture-group key per image (split unit)
    paths: np.ndarray
    classes: List[str]
    mapping: Dict[str, int]
    split: Split
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    inventory: DatasetInventory
    cache_path: Path

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def binary(self) -> bool:
        return len(self.classes) == 2

    @property
    def cache_size(self) -> int:
        """Edge length of the cached square grid (images are [N, S, S, 3])."""
        return int(self.images.shape[1])

    @property
    def positive_index(self) -> int:
        """Label id of the synthetic/"fake" class (never assumed - read from the names)."""
        from .dataset import POSITIVE_LABELS

        for i, c in enumerate(self.classes):
            if c in POSITIVE_LABELS:
                return i
        names = " ".join(self.classes).lower()
        if any(k in names for k in ("fake", "ai", "gen", "synth", "deepfake", "spoof")):
            for i, c in enumerate(self.classes):
                if any(k in c.lower() for k in ("fake", "ai", "gen", "synth", "deepfake", "spoof")):
                    return i
        return len(self.classes) - 1

    @property
    def positive_class(self) -> str:
        return self.classes[self.positive_index]

    @property
    def negative_class(self) -> Optional[str]:
        for i, c in enumerate(self.classes):
            if i != self.positive_index:
                return c
        return None


def build_data(
    root: Path,
    cache_dir: Path,
    archive: Optional[Path] = None,
    cache_size: int = 64,
    max_images_per_class: Optional[int] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 13,
    rebuild_cache: bool = False,
    verbose: bool = True,
    inventory_path: Optional[Path] = None,
) -> DataBundle:
    """Inspect -> validate -> cache -> split. The decode cache makes retraining cheap."""
    root = Path(root)
    ensure_extracted(root, archive)
    records, inv = scan_dataset(root, max_images_per_class=max_images_per_class)
    classes = sorted(inv.classes_found)
    mapping = assign_class_ids(records, classes)
    # ``classes`` is indexed by label id everywhere (confusion matrices, plot axes,
    # per-class statistics), so it must be ordered by that id - not alphabetically.
    classes = [name for name, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
    if verbose:
        print(
            f"[data] {inv.n_valid} valid / {inv.n_images} scanned | classes={inv.classes_found} "
            f"| groups={inv.n_groups} | corrupt={inv.n_corrupt} | near-dup={inv.n_duplicate_near}",
            flush=True,
        )
    if inventory_path:
        inv.save(Path(inventory_path))

    cache_dir = Path(cache_dir)
    build_cache(records, cache_dir, size=cache_size, rebuild=rebuild_cache)
    images, meta = load_cache(cache_dir)
    cache_path = cache_dir
    import gc

    del records
    gc.collect()
    labels = np.array([mapping[str(l)] for l in meta["labels"]], dtype=np.int64)
    groups = np.array([group_key(str(l), str(s)) for l, s in zip(meta["labels"], meta["source_ids"])], dtype=object)
    split = stratified_grouped_split(labels, groups, train_frac=train_frac, val_frac=val_frac, seed=seed)
    mean, std = _norm_stats(images, indices=split.train_idx)
    if verbose:
        print(f"[data] split sizes {split.sizes()} | norm mean={np.round(mean,3).tolist()} std={np.round(std,3).tolist()}", flush=True)
    db = DataBundle(
        images=images, labels=labels, groups=groups, paths=meta["paths"], classes=classes,
        mapping=mapping, split=split, mean=mean, std=std, inventory=inv, cache_path=cache_path,
    )
    gc.collect()
    return db


def _norm_stats(images: np.ndarray, indices: Optional[np.ndarray] = None, chunk: int = 1024
                ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Per-channel mean/std of the training split (the only place stats are estimated).

    Reads the memmap in small chunks: copying 14k images at once is what OOM-killed the
    first version of this pipeline on a 3.9 GB box.
    """
    tot = np.zeros(3, np.float64)
    tot2 = np.zeros(3, np.float64)
    npx = 0
    n = len(indices) if indices is not None else len(images)
    for st in range(0, n, chunk):
        rows = images[indices[st : st + chunk]] if indices is not None else images[st : st + chunk]
        x = np.asarray(rows, dtype=np.float32) / np.float32(255.0)
        tot += x.sum(axis=(0, 1, 2), dtype=np.float64)
        tot2 += (x * x).sum(axis=(0, 1, 2), dtype=np.float64)
        npx += x.shape[0] * x.shape[1] * x.shape[2]
    mean = tot / npx
    var = np.maximum(tot2 / npx - mean**2, 1e-8)
    return tuple(float(v) for v in mean), tuple(float(v) for v in np.sqrt(var))


# --------------------------------------------------------------------- deep training
def _make_loader(db: DataBundle, idx: np.ndarray, batch_size: int, train: bool, size: int, seed: int, gray: bool = False):
    """Batched transform over an in-RAM uint8 array (no DataLoader workers on 2 cores)."""
    import torch

    from .transforms import AugmentConfig, ImageTransform

    tf = ImageTransform(
        size=size, mean=db.mean, std=db.std, train=train,
        augment=AugmentConfig() if train else AugmentConfig.none(), seed=seed, to_gray=gray,
    )
    order = np.asarray(idx)

    def iterate(shuffle: bool = False, epoch: int = 0):
        order_ = np.random.default_rng(seed * 100003 + epoch).permutation(order) if shuffle else order
        for i in range(0, len(order_), batch_size):
            b = order_[i : i + batch_size]
            x = tf(np.array(db.images[b], copy=True))
            y = torch.from_numpy(db.labels[b].astype(np.float32 if db.binary else np.int64))
            yield x, y

    return tf, iterate


def _scores_from_model(model, db: DataBundle, idx: np.ndarray, size: int, device: str, gray: bool = False) -> np.ndarray:
    import torch

    _, iterate = _make_loader(db, idx, 256, False, size, 0, gray=gray)
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in iterate():
            o = model(x.to(device))
            o = o[:, 0] if (db.binary and o.ndim == 2 and o.shape[1] == 1) else o
            outs.append(o.double().cpu().numpy())
    z = np.concatenate(outs)
    if db.binary:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train_deep(
    db: DataBundle,
    preset: Dict[str, object],
    work_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    pretrained: bool = True,
    device: str = "cpu",
    torch_threads: int = 2,
    patience: int = 5,
    threshold_policy: str = "f1",
    progress: bool = True,
) -> Tuple[Candidate, Dict[str, object]]:
    """Train one deep candidate; returns (comparison row, run summary)."""
    import torch

    torch.set_num_threads(int(torch_threads))
    seed = int(preset.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = ModelSpec.from_dict({**dict(preset["spec"]), "num_classes": 1 if db.binary else db.n_classes})  # type: ignore[arg-type]
    spec.pretrained = bool(pretrained)
    applied_pretrained = True
    pretrained_note = ""
    try:
        model = build_model(spec)
    except Exception as exc:  # noqa: BLE001 - offline sandbox: weight download may fail
        if not pretrained:
            raise
        print(f"[warn] pretrained weights unavailable ({type(exc).__name__}); training from scratch", flush=True)
        applied_pretrained = False
        pretrained_note = f"requested but build_model failed ({type(exc).__name__}); trained from scratch"
    else:
        # the model itself knows whether ImageNet weights actually landed (it falls back internally)
        applied_pretrained = bool(getattr(model, "pretrained_applied", False))
        pretrained_note = str(getattr(model, "pretrained_note", "") or ("ImageNet weights loaded" if applied_pretrained else "not applied"))
        if pretrained and not applied_pretrained:
            print(f"[warn] {preset.get('id')}: pretrained requested but not applied -> {pretrained_note}", flush=True)
        spec.pretrained = False
        model = build_model(spec)
    if pretrained and not getattr(model, "backbone_pretrained", True):
        applied_pretrained = False
    model = model.to(device)

    epochs = int(epochs or preset["epochs"])  # type: ignore[arg-type]
    bs = int(batch_size or preset["batch_size"])  # type: ignore[arg-type]
    lr = float(lr or preset["lr"])  # type: ignore[arg-type]
    size = int(spec.input_size)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    y_tr = db.labels[db.split.train_idx]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) if db.binary else 1.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    gray = int(spec.in_channels) == 1
    tf, iterate = _make_loader(db, db.split.train_idx, bs, True, size, seed, gray=gray)
    tf_eval, _ = _make_loader(db, db.split.val_idx, 256, False, size, 0, gray=gray)
    crit = (
        torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        if db.binary
        else torch.nn.CrossEntropyLoss()
    )
    steps_per_epoch = max(1, math.ceil(len(db.split.train_idx) / bs))
    warmup = max(1, int(0.5 * steps_per_epoch))
    total_steps = steps_per_epoch * epochs

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (0.2 + 0.8 * step / warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return lr * (0.05 + 0.45 * (1 + math.cos(math.pi * min(1.0, p))))

    def evaluate(split_idx: np.ndarray) -> np.ndarray:
        return _scores_from_model(model, db, split_idx, size, device, gray=gray)

    history: List[Dict[str, float]] = []
    best_auc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for x, y in iterate(shuffle=True, epoch=epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            out = model(x.to(device))
            out = out[:, 0] if (db.binary and out.ndim == 2 and out.shape[1] == 1) else out
            loss = crit(out, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            run_loss += float(loss.detach()) * len(y)
            run_n += len(y)
            step += 1
            if progress and step % 40 == 0:
                print(f"  [{preset['id']}] ep{epoch:02d} {step}/{total_steps} loss {run_loss/max(run_n,1):.4f}", flush=True)
        vm = binary_metrics(db.labels[db.split.val_idx], evaluate(db.split.val_idx)) if db.binary else multiclass_metrics(
            db.labels[db.split.val_idx], evaluate(db.split.val_idx), db.classes
        )
        auc = float(vm.get("roc_auc", vm.get("accuracy", 0.0)))
        history.append(
            {"epoch": epoch, "loss": run_loss / max(run_n, 1), "lr": lr_at(step), "val_auc": auc,
             "val_acc": float(vm["accuracy"]), "val_f1": float(vm.get("f1", vm.get("macro_f1", 0.0)))}
        )
        if progress:
            print(f"  [{preset['id']}] epoch {epoch:02d} loss {history[-1]['loss']:.4f} val_auc {auc:.4f} val_acc {history[-1]['val_acc']:.4f}", flush=True)
        if auc > best_auc + 1e-5:
            best_auc, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            if progress:
                print(f"  [{preset['id']}] early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    thr = 0.5
    val_scores = evaluate(db.split.val_idx)
    if db.binary:
        thr = select_threshold(db.labels[db.split.val_idx], val_scores, policy=threshold_policy)
    test_scores = evaluate(db.split.test_idx)
    tm = binary_metrics(db.labels[db.split.test_idx], test_scores, threshold=thr) if db.binary else multiclass_metrics(
        db.labels[db.split.test_idx], test_scores, db.classes
    )

    # measured inference cost on this machine (batch of 1, warm cache)
    one = tf_eval(db.images[db.split.test_idx[:1]])
    with torch.no_grad():
        for _ in range(3):
            model(one)
    t_single = time.perf_counter()
    for _ in range(12):
        model(one)
    ms_per_image = (time.perf_counter() - t_single) / 12 * 1000.0

    ckpt = work_dir / "model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec": spec.to_dict(),
            "classes": db.classes,
            "label_map": db.mapping,
            "preprocess": {**tf_eval.spec(), "cache_size": db.cache_size},
            "threshold": float(thr),
            "arch_id": str(preset["id"]),
            "pretrained_requested": bool(pretrained),
            "pretrained_applied": bool(applied_pretrained),
            "pretrained_note": pretrained_note,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        ckpt,
    )
    np.save(work_dir / "test_scores.npy", test_scores)
    np.save(work_dir / "val_scores.npy", val_scores)
    (work_dir / "history.json").write_text(
        json.dumps({"history": history, "best_epoch": best_epoch, "threshold": thr, "val_auc": best_auc}, indent=2)
    )

    params = int(sum(p.numel() for p in model.parameters()))
    cand = Candidate(
        name=str(preset["id"]),
        family="deep",
        task="ai_detector" if db.binary else "multiclass",
        metrics=_jsonify(tm),
        size_mb=round(ckpt.stat().st_size / 1e6, 4),
        inference_ms=float(ms_per_image),
        pipeline="torch",
        params=params,
        notes=(
            f"{preset.get('note','')} | pretrained={applied_pretrained} | input={size}px "
            f"| best_epoch={best_epoch} | epochs_ran={len(history)} | pos_weight={pos_weight:.2f}"
        ).strip(" |"),
        artifacts={"checkpoint": str(ckpt), "test_scores": str(work_dir / "test_scores.npy"), "history": str(work_dir / "history.json")},
    )
    summary = {
        "id": preset["id"], "spec": spec.to_dict(), "history": history, "best_epoch": best_epoch,
        "threshold": thr, "params": params, "train_seconds": round(time.time() - t0, 1),
        "pretrained_requested": bool(pretrained), "pretrained_applied": bool(applied_pretrained),
        "pretrained_note": pretrained_note,
        "input_size": size, "cache_size": db.cache_size, "pos_weight": pos_weight,
        "trainable_parameters": int(sum(q.numel() for q in trainable)),
        "total_parameters": int(sum(q.numel() for q in model.parameters())),
        "frozen_backbone": bool(getattr(spec, "freeze_backbone", False)) and applied_pretrained,
    }
    return cand, summary


# ---------------------------------------------------------------- classical training
def _predict(pipe, X: np.ndarray) -> np.ndarray:
    """Probabilities. LinearSVC only exposes a margin, squashed through a logistic link."""
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(X), dtype=np.float64)
    d = np.asarray(pipe.decision_function(X), dtype=np.float64)
    if d.ndim == 1:
        p = 1.0 / (1.0 + np.exp(-d))
        return np.stack([1.0 - p, p], axis=1)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forensic_matrix(
    db: DataBundle,
    target_size: int,
    work_dir: Path,
    opts: Optional[FeatureOptions] = None,
    progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Descriptor matrix over the whole corpus, computed once and cached to disk."""

    from .transforms import resize_any, to_chw_float01

    opts = opts or FeatureOptions(chunk=512)
    key = f"features_{db.cache_size}to{target_size}"
    out = Path(work_dir) / f"{key}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return z["X"], [str(n) for n in z["names"]]
    imgs = db.images
    if imgs.shape[1] != target_size or imgs.shape[2] != target_size:
        n = len(imgs)
        out = np.empty((n, target_size, target_size, 3), dtype=np.uint8)
        for st in range(0, n, 2048):
            en = min(st + 2048, n)
            block = np.array(imgs[st:en], copy=True)
            xx = resize_any(to_chw_float01(block), target_size).mul(255.0).clamp(0, 255).byte()
            out[st:en] = xx.permute(0, 2, 3, 1).contiguous().numpy()
        imgs = out
    t0 = time.time()
    X, names = extract_features(imgs, opts, progress=progress)  # memmap slices stay lazy
    if progress:
        print(f"[features] {X.shape} in {time.time()-t0:.1f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, names=np.array(names))
    return X, names


PIXEL_INPUT_SIZE = 16


def pixel_matrix(db: DataBundle, size: int = PIXEL_INPUT_SIZE) -> np.ndarray:
    from .transforms import resize_any, to_chw_float01

    n = len(db.images)
    out = np.empty((n, 3 * size * size), dtype=np.float32)
    for st in range(0, n, 4096):
        en = min(st + 4096, n)
        block = np.array(db.images[st:en], copy=True)          # plain, writable ndarray
        x = resize_any(to_chw_float01(block), size)
        out[st:en] = x.permute(0, 2, 3, 1).reshape(en - st, -1).numpy()
    return out


def train_classical(
    db: DataBundle,
    preset: Dict[str, object],
    work_dir: Path,
    X_forensic: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    threshold_policy: str = "f1",
    progress: bool = True,
) -> Tuple[Candidate, Dict[str, object]]:
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    kind, feat = str(preset["model"]), str(preset["feature"])
    pixel_size = PIXEL_INPUT_SIZE
    X = np.ascontiguousarray(X_forensic) if feat == "forensic" else pixel_matrix(db, pixel_size)
    if X is None:
        raise RuntimeError("forensic features were not provided")
    names = list(feature_names or [])
    tr, va, te = db.split.train_idx, db.split.val_idx, db.split.test_idx
    y = db.labels

    tr = np.asarray(db.split.train_idx)
    steps: List[Tuple[str, object]] = [("scale", StandardScaler())]
    params: Dict[str, object] = {"model": kind, "descriptor": feat}
    if kind == "logreg":
        C = float(preset.get("C", 1.0))
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        params.update({"C": C, "penalty": "l2", "class_weight": "balanced", "max_iter": 3000})
    elif kind == "svm":
        C = float(preset.get("C", 2.0))
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=4000)
        params.update({"C": C, "loss": "squared_hinge", "margin_scores_mapped_to_p": "sigmoid"})
    elif kind == "rf":
        n = int(preset.get("n_estimators", 300))
        clf = RandomForestClassifier(
            n_estimators=n, min_samples_leaf=2, max_features="sqrt",
            n_jobs=2, random_state=0, class_weight="balanced_subsample",
        )
        params.update({"n_estimators": n, "min_samples_leaf": 2, "max_features": "sqrt",
                       "class_weight": "balanced_subsample"})
    elif kind == "gbdt":
        n = int(preset.get("n_estimators", 300))
        clf = HistGradientBoostingClassifier(
            max_iter=n, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.12, random_state=0,
        )
        params.update({"max_iter": n, "learning_rate": 0.06, "max_leaf_nodes": 31,
                       "l2_regularization": 1.0, "early_stopping": True})
    elif kind == "knn":
        k = int(preset.get("k", 15))
        # PCA cannot keep more components than samples/features (a tiny smoke set proved this)
        pca_k = int(max(1, min(int(preset.get("pca_components", 100)), X.shape[1], len(tr) - 1)))
        steps.append(("pca", PCA(n_components=pca_k, svd_solver="randomized", random_state=0)))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=2)
        params.update({"n_neighbors": k, "weights": "distance", "pca_components": pca_k})
    else:
        raise KeyError(f"unknown classical model {kind!r}")

    t0 = time.time()
    pipe = Pipeline(steps + [("clf", clf)])
    pipe.fit(X[tr], y[tr])
    if "pca" in dict(pipe.named_steps):
        params["pca_components"] = int(dict(pipe.named_steps)["pca"].n_components_)
    params["feature_dim"] = int(X.shape[1])
    params["train_samples"] = int(len(tr))
    if progress:
        print(f"  [{preset['id']}] fitted in {time.time()-t0:.1f}s", flush=True)

    def scores(idx: np.ndarray) -> np.ndarray:
        p = _predict(pipe, X[idx])
        return p[:, db.positive_index] if db.binary else p

    thr = 0.5
    if db.binary:
        thr = select_threshold(y[va], scores(va), policy=threshold_policy)
    s_te = scores(te)
    m = binary_metrics(y[te], s_te, threshold=thr) if db.binary else multiclass_metrics(y[te], s_te, db.classes)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "feature": feat,
            "feature_size": int(X.shape[1]),
            "threshold": float(thr),
            "classes": db.classes,
            "feature_names": names,
            "preprocess": {
                "resize_to": int(pixel_size if feat == "pixels" else db.cache_size),
                "feature": feat,
                "photometric": "gray" if feat == "forensic" else "none",
            },
        },
        path,
        compress=3,
    )

    one = X[te[:1]]
    for _ in range(3):
        _predict(pipe, one)
    t1 = time.perf_counter()
    for _ in range(20):
        _predict(pipe, one)
    ms = (time.perf_counter() - t1) / 20 * 1000.0
    np.save(work_dir / "test_scores.npy", s_te)
    np.save(work_dir / "val_scores.npy", scores(va))

    cand = Candidate(
        name=str(preset["id"]), family="classical", task="ai_detector" if db.binary else "multiclass",
        metrics=_jsonify(m), size_mb=round(path.stat().st_size / 1e6, 4), inference_ms=float(ms),
        pipeline="sklearn", params=params,
        notes=f"{preset.get('note','')} | descriptor={'forensic ' + str(X.shape[1]) + 'd' if feat=='forensic' else f'raw pixels {X.shape[1]}d'}".strip(" |"),
        artifacts={"model": str(path), "test_scores": str(work_dir / "test_scores.npy"), "val_scores": str(work_dir / "val_scores.npy")},
    )
    summary = {
        "id": preset["id"],
        "fit_seconds": round(time.time() - t0, 1),
        "threshold": float(thr),
        "feature_dim": int(X.shape[1]),
        "params": params,
        "descriptor": feat,
        "input_size": int(pixel_size if feat == "pixels" else db.cache_size),
    }
    return cand, summary


def _jsonify(d: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float, int)) and not isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (list, tuple, dict, str)) or v is None:
            out[k] = v
        else:
            out[k] = float(v)
    return out
