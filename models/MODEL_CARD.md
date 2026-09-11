# MODEL CARD

- **model_name**: vfa-ai-detector-svm_forensic
- **model_version**: 0.1.0
- **task**: binary AI/synthetic image detection (real vs generated)
- **architecture**: svm_forensic
- **framework**: scikit-learn
- **params**: {"model": "svm", "descriptor": "forensic", "C": 2.0, "loss": "squared_hinge", "margin_scores_mapped_to_p": "sigmoid", "feature_dim": 698, "train_samples": 14000}
- **image_size**: 64
- **n_classes**: 2
- **classes**: ["real", "fake"]
- **threshold**: 0.4821135216450504
- **model_size_mb**: 0.0245
- **inference_ms_per_image**: 0.226
- **pretrained_backbone**: None
- **training_date**: 2026-09-11

## Selection
- chosen from: `svm_forensic`, `lightcnn_32`, `logreg_forensic`, `gbdt_forensic`, `mobilenet_v3_small_64`, `efficientnet_b0_64`, `resnet18_64`, `knn_forensic`, `rf_forensic`, `logreg_pixels`, `rf_pixels`
- reason: deployment score 0.9275 from F1=0.8909, ROC-AUC=0.9513, accuracy=0.8887, 0.2 ms/image, 0.03 MB; beats runner-up lightcnn_32 (0.8615) by 0.0660.

## Data volumes
- training_images: 14000
- validation_images: 3000
- test_images: 3000
- dataset: `{"root": "/home/user/AMS/data", "classes_found": {"fake": 10000, "real": 10000}, "n_valid": 20000, "n_corrupt": 0, "class_balance": {"fake": 0.5, "real": 0.5}, "source_groups": 2000, "near_duplicates": 0, "perceptual_duplicates_across_sources": 39, "formats": {"JPEG": 20000}, "sizes": {"32x32": 20000}}`

## Metrics (test split, single evaluation)
| accuracy | precision | recall | f1 | roc_auc | pr_auc | specificity | false_positive_rate | eer | log_loss | brier | mcc | tp | fp | tn | fn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8887 | 0.8732 | 0.9093 | 0.8909 | 0.9513 | 0.9387 | 0.8680 | 0.1320 | 0.1110 | 0.3733 | 0.1087 | 0.7780 | 1364 | 198 | 1302 | 136 |

## Known limitations
- Trained on 32x32 upscaled crops of one corpus: it learns the artifacts of *this* data (compression, texture, spectrum), not a universal notion of 'AI-generated'.
- Performance degrades on generators, resolutions, cameras and post-processing pipelines that are not represented in the training set.
- Heavy re-compression, screenshots, resizing, watermarks and upscaling change the measured evidence and can flip predictions.
- Class balance here is 50/50; in production the real rate is far higher, so the false-positive count matters more than accuracy suggests.
- The crops are faces at very low resolution: identity is not recoverable and no identity model was trained.
- In this corpus the two classes differ measurably in sharpness and tone, which is a shortcut a classifier can exploit instead of learning synthesis artefacts. Photometric statistics are per-image z-scored in both the descriptor and the model input to limit it, and the raw-pixel baselines plus the colour/photometric ablation in reports/model_comparison.md quantify how much of the performance that shortcut could explain; a residual correlation cannot be excluded.

## Scientific honesty
- Prediction is probabilistic and is not proof of image origin.
- Generator identification is limited to patterns represented in the training dataset.
- Performance may decrease on unseen generators.
- Image editing, compression, screenshots and resizing can affect predictions.
- Face analysis does not identify people.
- No detector is 100% accurate and this one makes no such claim: an individual prediction can be wrong, and 'authentic' does not prove human authorship.

## Secondary models
- generator classifier: {"name": "generator", "available": false, "classes": {}, "n_classes": 0, "reason": "the provided dataset carries exactly one label dimension (the real/ai class folders); no per-generator sub-directories and no generator tokens in file names, so generator fingerprinting cannot be trained - it would require inventing labels", "source": "none"}
- face model: {"trained": false, "requested": true, "reason": "the corpus holds exactly two classes (real / synthetic face crops) and no 'manipulated' label dimension, so a separate REAL-FACE vs DEEPFAKE-FACE classifier would be a copy of the AI detector with a new name. Rather than duplicate it, the detector is trained on the face crops themselves and face-level checks are provided by the analytical CV module 