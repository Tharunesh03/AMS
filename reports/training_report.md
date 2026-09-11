# TRAINING REPORT

- wall-clock: **72.6 min** on Linux x86_64, CPU, Python 3.11.2, torch 2.14.0+cu130
- run: `full` | device `cpu` | threads `2` | seed `13`
- split: 70/15/15 stratified + capture-group atomic | cache grid `64px` | descriptor grid `64px`
- photometric handling: `gray` (per-image z-score for the classical descriptor)
- configs and per-run artifacts: `artifacts/<run>/` (gitignored), production model: `models/`

## Per-candidate protocol

| Candidate | id | epochs | best epoch | train time | threshold | params | input | pretrained |
|---|---|---|---|---|---|---|---|---|
| logreg_forensic | logreg_forensic | - | - | 3s | 0.400 | {'model': 'logreg', 'descriptor': 'forensic', 'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced', 'max_iter': 3000, 'feature_dim': 698, 'train_samples': 14000} | 64px | n/a |
| svm_forensic | svm_forensic | - | - | 23s | 0.482 | {'model': 'svm', 'descriptor': 'forensic', 'C': 2.0, 'loss': 'squared_hinge', 'margin_scores_mapped_to_p': 'sigmoid', 'feature_dim': 698, 'train_samples': 14000} | 64px | n/a |
| rf_forensic | rf_forensic | - | - | 27s | 0.495 | {'model': 'rf', 'descriptor': 'forensic', 'n_estimators': 250, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample', 'feature_dim': 698, 'train_samples': 14000} | 64px | n/a |
| gbdt_forensic | gbdt_forensic | - | - | 29s | 0.412 | {'model': 'gbdt', 'descriptor': 'forensic', 'max_iter': 250, 'learning_rate': 0.06, 'max_leaf_nodes': 31, 'l2_regularization': 1.0, 'early_stopping': True, 'feature_dim': 698, 'train_samples': 14000} | 64px | n/a |
| knn_forensic | knn_forensic | - | - | 2s | 0.334 | {'model': 'knn', 'descriptor': 'forensic', 'n_neighbors': 15, 'weights': 'distance', 'pca_components': 100, 'feature_dim': 698, 'train_samples': 14000} | 64px | n/a |
| logreg_pixels | logreg_pixels | - | - | 17s | 0.310 | {'model': 'logreg', 'descriptor': 'pixels', 'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced', 'max_iter': 3000, 'feature_dim': 768, 'train_samples': 14000} | 16px | n/a |
| rf_pixels | rf_pixels | - | - | 39s | 0.447 | {'model': 'rf', 'descriptor': 'pixels', 'n_estimators': 300, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample', 'feature_dim': 768, 'train_samples': 14000} | 16px | n/a |
| lightcnn_32 | lightcnn_32 | 20 | 19 | 411s | 0.839 | 473785 | 32px | no |
| mobilenet_v3_small_64 | mobilenet_v3_small_64 | 12 | 11 | 527s | 0.604 | 927585 | 64px | no |
| efficientnet_b0_64 | efficientnet_b0_64 | 8 | 7 | 1078s | 0.625 | 4008829 | 64px | no |
| resnet18_64 | resnet18_64 | 8 | 7 | 856s | 0.643 | 11177025 | 64px | no |
| ablate_lightcnn_64 | ablate_lightcnn_64 | 10 | 7 | 995s | 0.644 | 473785 | 64px | no |
| ablate_lightcnn_gray | ablate_lightcnn_gray | 12 | 11 | 250s | 0.842 | 473353 | 32px | no |

## Honest notes about this run

- `pretrained=no` means the ImageNet weight download was unreachable from this environment; the run continues from random initialisation and the fact is recorded per candidate.
- Deep models are compute-capped on this 2-core CPU box; epochs listed are what actually ran (early stopping on validation ROC-AUC), not a promise of convergence.
- Ablations are recorded separately and cannot win selection.
