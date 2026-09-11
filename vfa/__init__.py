"""Visual Forensic AI · supervised image-provenance toolkit.

Modules
-------
``dataset``      dataset-first inspection, validation, decode cache
``splitting``    stratified + capture-group-atomic splits (no leakage)
``features``     forensic descriptors for the classical candidates
``models``       LightCNN + torchvision candidate zoo, presets
``trainer``      training/eval protocol shared by every candidate
``selection``    weighted deployment score and the selection rationale
``metrics``      binary/multiclass metrics, threshold policy, calibration
``gradcam``      Grad-CAM saliency from the actual production model
``explain``      measured-signal evidence engine + simple/technical wording
``forensics``    analytical CV face module (not a trained classifier)
``predictor``    deployment inference: loads models/, never retrains
``reporting``    markdown reports, plots, UI sample export
``cli``          ``python -m vfa {inspect,train,reference,evaluate,predict,demo,serve}``
"""

__version__ = "0.1.0"
