# Production model re-evaluation (test split)

- model: `vfa-ai-detector-svm_forensic` (forensic-descriptor), threshold 0.482
- images: 3000 | 19.6s | 6.53 ms/image on this machine
- accuracy 0.8887 | F1 0.8909 | ROC-AUC 0.9513 | recall 0.9093 | precision 0.8732 | EER 0.1110
- FPR (false lock rate on authentic) 0.1320 | FNR (missed synthetic) 0.0907

- recomputed metrics reproduce the reported `models/metrics.json` within the stated tolerance 1e-04 (max abs delta 1.78e-06) - the deployed artifact is the same model the dashboard serves

This recomputation loads only the shipped artifact; it does not retrain anything.
