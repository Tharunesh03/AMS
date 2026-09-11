# AMS — AI-Image Authenticity Detector

Supervised **REAL vs AI-GENERATED** image classification trained on the dataset shipped in this
repository, with a model-comparison study, Grad-CAM saliency, a measurement-based explanation
engine, an analytical (non-classifier) face module, a results dashboard and a CPU-only FastAPI
service.

Everything here is CPU-only, uses free/open libraries and no external "AI detection" API. No number
in the UI, the reports or the model card is invented: metrics come from measured evaluation on the
held-out split, heatmaps come from the trained network's own gradients, and the explanation signals
are computed from the uploaded pixels.

---

## 1. What the dataset actually contains (inspected, not assumed)

`python -m ams inspect` (or step 1 of `train.py`) walks every file before anything is trained and
writes `reports/eda.md` + `models/dataset_inspection.json`. Measured on this corpus:

| property | measured value |
|---|---|
| classes on disk | `FAKE/`, `REAL/` — exactly two, 10 000 images each |
| label balance | 50 / 50 (`minority_ratio = 1.0`) |
| image format | 20 000 × JPEG, all RGB, **all 32×32** |
| corrupt / unreadable | **0** (0 rejected during validation) |
| exact duplicates | **0** |
| near-duplicates (perceptual hash) | **0** |
| capture structure | 1 000 source ids per class, 10 views each (`0001.jpg`, `0001 (2).jpg`, …) → 2 000 atomic groups |
| second label dimension | **none** — no generator names in paths or file names, no metadata files |
| photometric difference | `FAKE` mean 109.3 / std 62.4 vs `REAL` mean 122.1 / std 64.7; Laplacian-variance median **3603 (FAKE)** vs **2070 (REAL)** |

Two consequences drive the whole design:

* **The classes are `real` and `fake`, not generators.** Generator fingerprinting therefore *cannot*
  be trained here — see §4.
* **Sharpness/brightness differ systematically between the classes**, which is exactly the kind of
  shortcut a CNN exploits without learning anything about synthesis. The descriptor and the network
  input are therefore per-image z-scored (`photometric: gray` in `configs/default.yaml`), and
  resolution/colour ablations plus a raw-pixel baseline are reported so the shortcut stays visible
  (`reports/model_comparison.md`).

The images are 32×32 face crops, so the honest framing of the trained task is: *is this small face
crop an authentic camera capture or an AI-generated/synthetic render* — a presentation-attack /
synthetic-face gate, not a general "is this image AI" oracle.

## 2. Pipeline

```
python train.py --config configs/default.yaml
```

runs the whole chain reproducibly (same command regenerates every artifact below):

1. **inspect + validate** — every file opened, decoded, dimensions/format/dup/group statistics (§1)
2. **label dimensions** — binary detector enabled; generator dimension refused with a reason (§4)
3. **preprocess** — decode cache at 64 px (`artifacts/cache`, memory-mapped), per-image z-scoring,
   normalisation statistics estimated on the **training split only**
4. **classical candidates** — 698-dimensional forensic descriptor (photometric moments, tone
   histogram, high-pass energy, Laplacian/edge energy, noise floor, block-grid energy, radial
   spectrum, uniform LBP, 8×8×9 HOG, colour/saturation deltas) → LogReg / Linear-SVM / Random
   Forest / gradient boosting / k-NN, plus raw-pixel baselines
5. **deep candidates** — lightweight CNN (native 32 px) and MobileNetV3-Small / EfficientNet-B0 /
   ResNet-18 at 64 px, with train-only augmentation (flip, ±8 px shift, ±8 % scale, brightness,
   contrast, mild blur, JPEG round-trip) and early stopping on validation ROC-AUC
6. **compare + select** — weighted deployment score over the *validation*-fitted threshold
7. **export** — only the selected model into `models/` + model card
8. **reports** — comparison, training report, ROC/PR/confusion/reliability/curve figures

Splits: **70 / 15 / 15** stratified by class and **atomic per capture group** (`(class, source_id)`),
seed 13 → measured sizes `train 14 000 / val 3 000 / test 3 000`, no group ever straddles two
splits, no near-duplicate crosses a boundary. The test split is used **once**, at step 7–8.

Preprocessing detail: sources are 32×32, so 64 px is chosen for the descriptor/network grid (2×
headroom for interpolation-free augmentation) instead of 224 px — at this content size a 224 px
grid only multiplies compute by ~12 for zero new detail. The justification and the 64 px ablation
arm are recorded in `reports/model_comparison.md`.

## 3. Model selection rule (no arbitrary winner)

```
score = 0.40·F1 + 0.25·ROC-AUC + 0.15·Accuracy + 0.10·Speed + 0.10·Efficiency
```

`Speed = min(1, 3.33 ms / ms_per_image)` (a throughput target of ~300 img/s on 2 CPU threads) and
`Efficiency = 0.5·size_score + 0.5·pipeline_score` with `size_score = 1 − log10(max(MB,1))/3`.
Sub-scores, ranks and every candidate's full metric payload land in `models/metrics.json`, and
**exactly the model with the best score is promoted** — with one documented exception: when the
top two candidates are within 0.01 on both F1 and ROC-AUC, the smaller/faster artifact is selected
and the reason is written out verbatim in the report and the dashboard. Non-selected candidates stay
in `artifacts/full/` for auditing; they are never shipped.

## 4. Generator fingerprinting — reported, not fabricated

`ams/labels.py` looks for a second label dimension in two places (nested directories under a class,
generator tokens in file names) and requires ≥ 2 classes with ≥ 25 images. On this corpus it finds
nothing, so the outcome is recorded rather than invented:

```json
"generator_classifier": {"available": false, "n_classes": 0,
  "reason": "the provided dataset carries exactly one label dimension (the real/ai class folders);
             no per-generator sub-directories and no generator tokens in file names, so generator
             fingerprinting cannot be trained - it would require inventing labels"}
```

Consequences on **this** corpus: there is no generator model to gate, so the `Generator ID` tab and the
`generator_identification` field of every API response carry that same explanation, and `train.py` prints
`generator task: not trainable on this dataset (recorded, nothing invented)`.

**The second task is implemented, it is only the labels that are missing.** With per-generator folders
present (`data/FAKE/stable_diffusion/*.jpg`), `ams/generator.py` builds an independent training bundle
(generator-scoped capture groups, the same group-atomic 70/15/15 rule, re-measured normalisation), trains
the same candidate pool on it, selects with the multi-class weights
`0.35·macro-F1 + 0.20·weighted-F1 + 0.15·accuracy + 0.10·top-3 + 0.10·speed + 0.10·efficiency`
(top-3's weight moves to macro-F1 when the task has ≤ 3 classes, because a "top 3 of 3" score cannot fail),
exports the winner to `models/generator_classifier/` and reports macro/weighted F1, per-class F1,
confusion matrix and inference cost in `models/metrics.json`. Unknown-generator logic is real too:
`fit_unknown_gate` sweeps a `(min_prob, min_margin)` grid on the **validation** split and keeps the loosest
rule that still reaches 90 % precision on accepted predictions, and `apply_gate` answers `UNKNOWN` whenever
an image fails it - a name is never forced onto an image from a generator the model was not shown.
`tests/test_generator_task.py` exercises all of that end to end on a synthetic labelled corpus
(3 generators × 36 images); nothing about those numbers transfers to this dataset, which has no such labels.
Disable the step with `python train.py --no-generator-task`.

## 5. Face analysis (§11) — an analytical module, explicitly not a trained classifier

The corpus has two classes (authentic/synthetic *face crops*) and no "manipulated-face" dimension, so
a second "deepfake face" classifier would be a rename of the detector. `ams/forensics.py` instead
measures classical CV signals: Haar-cascade face localisation, eye-line geometry plausibility,
left/right mirror correlation, high-pass/micro-texture energy, per-region sharpness dispersion,
flat-region fraction, boundary gradient continuity at the face box, illumination-slope agreement with
the surround, and 8-px blockiness. It reports `is_trained_classifier: false`, `kind:
"analytical_cv_module"`, never an identity, and states that it is not a deepfake detector. For inputs
that are already tight face crops (like this corpus), it also analyses an *assumed* crop region and
labels that fact in `summary.region_source` instead of pretending a detection happened.

## 6. Explanations

* **Grad-CAM** (`ams/gradcam.py`) differentiates the *actual* selected checkpoint: the last
  convolutional activation of the loaded model, weighted by the gradient of the synthetic-class logit.
  The heatmap is produced at the model grid and upsampled with a note about that; the response
  includes the target layer name, peak/spread/mass statistics, a unit-coordinate bbox and the overlay
  PNG. If the selected model is a classical descriptor classifier there is no convolutional
  activation, and the API says so instead of faking a map.
* **Evidence engine** (`ams/explain.py`) measures 10 image signals, compares them with per-class
  distributions estimated on the **training split only** (`models/reference_stats.json`), grades each
  as LOW/MEDIUM/HIGH (combining the deviation of this image and the measured discriminative power of
  the signal) and writes both a plain-language and a technical paragraph.

## 7. Repository layout

```
ams/            dataset.py splitting.py features.py models.py transforms.py trainer.py
                metrics.py selection.py labels.py generator.py gradcam.py explain.py
                forensics.py predictor.py pipeline.py reporting.py cli.py
train.py        single entry point for the whole chain
configs/        default.yaml (full run), smoke.yaml (seconds-long sanity run)
app/            server.py (FastAPI) + static/ dashboard (no build step, no CDN)
tests/          unit + integration tests, incl. two real end-to-end training runs
models/         SHIPPED artifacts: ai_detector/{model.pth|model.joblib,classes.json,...},
                generator_classifier/ (only when the dataset carries per-generator labels),
                metrics.json, model_info.json, MODEL_CARD.md, preprocessing.json,
                dataset_inspection.json, reference_stats.json
reports/        eda.md, model_comparison.md, training_report.md, figures/*.png
samples/        a few held-out test images + manifest.json for the dashboard demo
artifacts/      caches + per-candidate runs (git-ignored; kept for auditing)
Dockerfile, render.yaml, Makefile, requirements.txt, pyproject.toml
```

`data/`, `artifacts/` and `.venv/` are git-ignored; the dataset stays on disk for the scripts.
`models/`, `samples/`, `reports/` are committed so the service can load a trained model without
retraining.

## 8. Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --config configs/smoke.yaml    # seconds: full chain on tiny data,
                                                          # writes to models_smoke/ + reports/smoke/
.venv/bin/python train.py                                 # full comparison + export (see §2)
.venv/bin/python -m pytest -q                             # tests
.venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000    # dashboard + API
```

CLI (same code paths as the app):

```bash
python -m ams inspect            # dataset validation report only
python -m ams train --smoke      # tiny end-to-end run
python -m ams reference          # rebuild the evidence-engine reference statistics
python -m ams evaluate           # re-measure the shipped artifact on the test split
python -m ams predict path/to.jpg --gradcam
python -m ams demo               # predict the bundled samples
python -m ams serve --port 8000
```

Any config key can be overridden: `python train.py --set deep_ids=[lightcnn_32] --set epochs=3`.
`--only classical|deep|ablations` narrows the pool, `--no-ablations` skips the ablation arms
(always reported, never eligible for selection), `--no-generator-task` skips the second supervised
task entirely, `--pretrained` / `--no-pretrained` control the ImageNet-weight request, and
`--rebuild-cache` re-decodes the corpus. `--inspect-only` writes `reports/eda.md` and stops.

### HTTP API

| endpoint | purpose |
|---|---|
| `GET /` | dashboard (upload/drop/paste, comparison table, model card, reports) |
| `POST /api/detect` | multipart image → probability, verdict, threshold, Grad-CAM overlay, evidence, face analysis, generator identification (only when a generator model was trainable), timings |
| `GET /api/model` | model card, classes, selection reason, full comparison, task statuses |
| `GET /api/metrics` | `models/metrics.json` verbatim |
| `GET /api/samples`, `GET /api/samples/{file}` | held-out demo images |
| `GET /api/report/{name}` | `reports/*.md` / figures |
| `GET /api/health` | model loaded? device, upload cap, available reports |

Uploads are validated (size ≤ 8 MB by default, image content types, no temp files), the model is
loaded once at startup and never retrained per request, and oversized inputs are capped with an
explicit warning in the response.

### Deployment (free tier, CPU only)

`Dockerfile` (python:3.11-slim, no CUDA) and `render.yaml` (Render free web service, health check on
`/api/health`) are included. `python -m ams serve` binds `0.0.0.0:$PORT` for platform-provided
ports. Nothing in the request path needs a GPU, a paid API or more RAM than a 512 MB instance.

## 9. Measured results

<!-- RESULTS:BEGIN -->
Measured on this machine in a single run (seed 13), full corpus, 2026-09-11T09:55:45. Regenerated from `models/metrics.json` by `python tools/sync_readme_results.py` - no number here is typed by hand.

* dataset: 20,000 usable images of 20,000 found, classes fake 10,000, real 10,000, all 32x32 px; corrupt 0, exact duplicates 38, near-duplicates 0, capture groups 2,000
* splits: train 14,000 / val 3,000 / test 3,000 (stratified by class, atomic by capture group (70/15/15))
* selected production model: **svm_forensic** (scikit-learn; model=svm, descriptor=forensic, C=2.0, loss=squared_hinge, margin_scores_mapped_to_p=sigmoid) — 64 px input, 0.02 MB, 0.2 ms/image on CPU
* transfer learning: requested, but no weights could be loaded in the training environment: all 4 deep candidates in the pool trained from scratch; no backbone was frozen, since freezing a randomly initialised one is not transfer learning (full record in §9.3)
* decision threshold 0.482 (max-F1 on the validation split (midpoint between distinct validation scores)), test split evaluated 1 time(s)

## 9.1 Comparison (test split, all candidates)

| rank | candidate | family | accuracy | F1 | ROC-AUC | recall | precision | FPR | EER | ms/img | MB | score |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **svm_forensic** | classical | 0.8887 | 0.8909 | 0.9513 | 0.9093 | 0.8732 | 0.1320 | 0.1110 | 0.2 | 0.03 | **0.9275** |
| 2 | lightcnn_32 | deep | 0.9340 | 0.9346 | 0.9837 | 0.9427 | 0.9266 | 0.0747 | 0.0697 | 1.6 | 1.92 | 0.8615 |
| 3 | logreg_forensic | classical | 0.8850 | 0.8887 | 0.9511 | 0.9180 | 0.8612 | 0.1480 | 0.1093 | 0.7 | 0.02 | 0.8586 |
| 4 | gbdt_forensic | classical | 0.9083 | 0.9102 | 0.9684 | 0.9287 | 0.8924 | 0.1120 | 0.0900 | 1.6 | 1.12 | 0.8558 |
| 5 | mobilenet_v3_small_64 | deep | 0.9063 | 0.9070 | 0.9728 | 0.9140 | 0.9002 | 0.1013 | 0.0957 | 3.3 | 3.83 | 0.8316 |
| 6 | efficientnet_b0_64 | deep | 0.9197 | 0.9211 | 0.9768 | 0.9380 | 0.9048 | 0.0987 | 0.0800 | 7.7 | 16.31 | 0.8258 |
| 7 | resnet18_64 | deep | 0.9153 | 0.9151 | 0.9771 | 0.9120 | 0.9181 | 0.0813 | 0.0840 | 6.8 | 44.78 | 0.8159 |
| 8 | knn_forensic | classical | 0.8557 | 0.8601 | 0.9265 | 0.8873 | 0.8345 | 0.1760 | 0.1433 | 2.4 | 5.51 | 0.8010 |
| 9 | rf_forensic | classical | 0.8317 | 0.8337 | 0.9137 | 0.8440 | 0.8237 | 0.1807 | 0.1707 | 45.1 | 12.90 | 0.7686 |
| 10 | logreg_pixels | classical | 0.6123 | 0.6972 | 0.7213 | 0.8927 | 0.5720 | 0.6680 | 0.3337 | 0.2 | 0.02 | 0.7460 |
| 11 | rf_pixels | classical | 0.7650 | 0.7904 | 0.8673 | 0.8860 | 0.7134 | 0.3560 | 0.2147 | 45.7 | 18.42 | 0.7271 |

Selection: deployment score 0.9275 from F1=0.8909, ROC-AUC=0.9513, accuracy=0.8887, 0.2 ms/image, 0.03 MB; beats runner-up lightcnn_32 (0.8615) by 0.0660.
The most accurate candidate on the test split was **lightcnn_32** (F1 0.9346, ROC-AUC 0.9837, accuracy 0.9340), and it was *not* selected: the deployment score also weighs latency and artifact size (1.92 MB @ 1.6 ms versus 0.03 MB @ 0.2 ms for svm_forensic). Deep models were more accurate here and a shallow one still won the stated rule - the table is the evidence, and picking the larger model would have meant ignoring the published weights.

The selected model is a linear margin classifier whose decision values are squashed through a logistic link to obtain probabilities, so the numbers are indicative rather than calibrated: measured ECE 0.1470 on the test split, brier 0.1087 (per-bin reliability table in `models/metrics.json`). The ranking of scores - what the threshold acts on - is unaffected.

## 9.2 Selected model on the held-out test split

| accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | specificity | false_positive_rate | false_negative_rate | eer | log_loss | brier | mcc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8887 | 0.8887 | 0.8732 | 0.9093 | 0.8909 | 0.9513 | 0.9387 | 0.8680 | 0.1320 | 0.0907 | 0.1110 | 0.3733 | 0.1087 | 0.7780 |

Confusion over 3000 test images, rows = true class, columns = predicted (`real`, `fake`): TN 1302, FP 198, FN 136, TP 1364. Calibration error (ECE) 0.1470.

## 9.3 Secondary tasks

* generator fingerprinting: **not trainable on this dataset** — the provided dataset carries exactly one label dimension (the real/ai class folders); no per-generator sub-directories and no generator tokens in file names, so generator fingerprinting cannot be trained - it would require inventing labels (0 generator classes found). The second task itself is implemented and tested (`ams/generator.py`, `tests/test_generator_task.py`); only the labels are missing, and no classes were invented to work around that.
* face/deepfake model: trained=False — the corpus holds exactly two classes (real / synthetic face crops) and no 'manipulated' label dimension, so a separate REAL-FACE vs DEEPFAKE-FACE classifier would be a copy of the AI detector with a new name. Rather than duplicate it, the detector is trained on the face crops themselves and face-level checks are provided by the analytical CV module (ams.forensics: face detection, landmark geometry, visual consistency), which is explicitly NOT a trained deepfake classifier.
* transfer learning: requested=True, weights actually loaded for `none`, every deep candidate in the selection pool (6) trained from scratch, frozen-backbone arms applied `none` — transfer learning was requested but the ImageNet weight host was unreachable from this environment (architecture has no published ImageNet weights; trained from scratch by design ; requested but unreachable (URLError: <urlopen error TLS/SSL connection has been closed (EOF) (_ssl.c:992)>); from scratch); freezing a randomly initialised backbone would only train a head on random features, so no arm was frozen and every deep candidate trained from scratch with all layers trainable. This is recorded per candidate in pretrained_note rather than hidden.

## 9.4 Ablations (reported, never eligible for selection)

[
  {
    "id": "ablate_lightcnn_64",
    "accuracy": 0.941,
    "f1": 0.9412545635579157,
    "roc_auc": 0.9846102222222223,
    "recall": 0.9453333333333334,
    "inference_ms": 3.7122337500174276,
    "size_mb": 1.915,
    "note": "same lightcnn at 64px input; sources are 32x32 so 64px is an interpolated enlargement, not new detail - measures whether extra input resolution changes anything"
  },
  {
    "id": "ablate_lightcnn_gray",
    "accuracy": 0.9006666666666666,
    "f1": 0.9009966777408638,
    "roc_auc": 0.9676604444444444,
    "recall": 0.904,
    "inference_ms": 1.4573294999233137,
    "size_mb": 1.9132,
    "note": "single-channel luminance input - is colour information needed at all?"
  }
]
<!-- RESULTS:END -->

## 10. Limitations, stated plainly

* Predictions are **probabilistic**; nothing here proves the origin of an image, and no claim of
  100 % accuracy is made anywhere in this project.
* Measured on 32×32 crops of faces from **one corpus**: recall on unseen generators, other
  resolutions, other pipelines or non-face content will be different — likely worse. That is a
  domain-shift fact, not something this model can compensate for.
* Heavy re-compression, screenshots, resizing, watermarks and upscaling change the measured
  evidence and can flip predictions.
* The corpus is balanced (50/50); at a realistic base rate of synthetic images the *count* of false
  accusations is what matters, so `false_positive_rate` on authentic captures and the calibration
  table (`models/metrics.json` → `reliability`) are reported next to accuracy/F1.
* Image-level explanations describe which measured signals deviate from authentic captures plus the
  network's own saliency; they are not a semantic proof about how the image was produced.
* The face module identifies no people, stores no biometrics and is not a deepfake detector.

## 11. Environment notes recorded during development

* **Transfer learning was attempted and is recorded honestly per candidate**: this sandbox cannot
  reach `download.pytorch.org`, so the ImageNet weight download fails (`URLError`) and the affected
  architectures train from scratch. `pretrained_requested` / `pretrained_applied` /
  `pretrained_note` are written into every summary and the model card, so "no transfer learning
  happened here" is visible in the artifacts rather than hidden. On a machine with network access the
  same run uses real ImageNet weights with no code change.
* **The frozen-backbone stage of the transfer-learning protocol could not be run here.** The plan was
  frozen pretrained backbone → train the head → fine-tune a few blocks if needed; freezing is skipped
  deliberately when no weights loaded (a frozen *random* backbone would only train a head on random
  features, and reporting that as "transfer learning" would be false). `models/metrics.json` records the
  whole protocol under `transfer_learning` (`requested`, `weights_applied_for`,
  `frozen_backbone_requested_for`, `frozen_backbone_applied_for`, `trainable_parameter_ratios`), where
  `trainable == total` per architecture is the measured evidence that nothing was frozen.
  `configs/default.yaml` carries the frozen-arm ablation, commented out, ready to enable where weights
  are downloadable.
* 2 CPU threads and 3.9 GB RAM forced the design of the caches (memory-mapped decode cache, chunked
  descriptor extraction) — a first version OOM-killed itself at 20 000 images.
* `opencv-python-headless` is pinned `<5` because the 5.0 wheels dropped the Haar cascade XMLs the
  face module needs.
