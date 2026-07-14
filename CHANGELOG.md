# Changelog

All notable decisions and changes to FlameGuard AI, in the order they happened.

## Sprint 1 — Project planning and data

- Compared two candidate Roboflow fire/smoke datasets. **Rejected v2**
  ("fire and smoke detection", 15,345 images): frames from all 19 source videos appear
  in train, valid *and* test, so its test split cannot measure generalisation without a
  full rebuild; it also has histogram equalisation, shear, blur and salt-and-pepper noise
  baked into every image, and almost no negatives.
- **Selected v1** ("fire and smoke", 5,300 images, CC BY 4.0): clean 640×640 images, no
  baked-in augmentation, and ~2,000 background negatives that support false-positive
  control.
- Built the dataset integrity audit (`src/dataset_validator.py`): image/label matching,
  corrupt-image decoding, malformed and out-of-range annotations, duplicate detection,
  per-class counts. Result: 0 corrupt images, 0 invalid label lines, both classes far
  above the 200-original-image requirement.
- **Found leakage in v1 as well.** 409 perceptual-hash source groups — 3,379 images, 64%
  of the dataset — had members in more than one split (mirrored/noise-augmented copies
  and sequential frames). Built `src/resplit.py`: canonical-stem grouping unioned with
  pHash near-duplicate clustering (Hamming ≤ 6), then whole groups assigned to
  train/valid/test at 70/20/10, stratified by content, seed 42. Audited to **0 spanning
  groups**.
- Full EDA package generated from the repaired split.

## Sprint 2 — Baseline and comparison

- Environment: Python 3.14, PyTorch 2.11 + CUDA 12.8, Ultralytics 8.4.95.
- **AMP auto-disabled** by Ultralytics' pre-flight check (GTX 16xx NaN-loss issue) →
  all training runs in FP32, roughly doubling epoch time. Measured: YOLOv8n @640,
  batch 16 = ~4.4 min/epoch.
- **E1 YOLOv8n baseline**: 40 epochs, 2h56m. Best epoch 39 — the run had not fully
  plateaued, which is disclosed rather than presented as convergence.
- **YOLOv8s at batch 8 does not fit**: it requested ~7.8 GB against 4 GB of VRAM, spilled
  into shared system memory, and collapsed to ~20 min/epoch (2.2 s/iter). Killed and
  re-planned at batch 4 with a short, explicitly compute-limited budget.
- Experiment protocol re-scoped to the measured hardware cost, and the fair comparison
  method changed accordingly: because validation metrics are logged every epoch, models
  trained for different numbers of epochs are compared at **equal epochs**.

## Sprint 3 — Tuning and application

- Tuning designed as a **controlled study**: a control run (`e4d`) at the probe budget,
  then three probes each changing exactly one factor — optimizer (AdamW + lr 1e-3),
  augmentation strength (HSV value, scale), and classification-loss weight.
- YOLO11n added as the primary architecture comparison: same batch, same image size and
  the same cost class as the baseline, so architecture is the only variable.
- Streamlit application built around a single cached `DetectionEngine`, shared by the
  image, video and webcam paths.
- **Bug fixed:** `torch.cuda.is_available()` can return `True` while `device_count()` is
  0 (e.g. `CUDA_VISIBLE_DEVICES=""`), which crashed the sidebar with "Invalid device id".
  `pick_device()`/`device_label()` now require a genuinely usable device. Caught by
  driving the app in a browser, not by the unit tests.
- **Bug fixed:** Ultralytics resolves a relative `path:` in `data.yaml` against its own
  global datasets directory, not against the yaml's location. A resolved copy is now
  written at runtime, so the committed `data.yaml` stays machine-independent.

## Sprint 4 — Evaluation and delivery

- Confidence threshold selected on the **validation** split (F1-optimal, with a
  documented recall bias for safety), then frozen.
- **Test split evaluated exactly once**, after the model and threshold were fixed.
- Structured error analysis with IoU-matched, class-aware categorisation:
  true positives, false positives, false negatives and localisation errors, each with a
  gallery.
- Report (DOCX + PDF), 15-slide deck, speaker notes and demo script generated
  **programmatically from the saved artefacts**, so no figure or number in the documents
  can drift from what the pipeline actually produced.
- Submission archive built and verified.
