# Sprint Summaries

## Sprint 1 - Project planning & data

**Goal.** Establish scope, validate the dataset, complete EDA.

**Committed / completed:** 22 / 22 story points (8 items).

**Delivered:**

- Analyse project requirements and rubric (FG-01)
- Define team roles and communication plan (FG-02)
- Acquire Roboflow fire/smoke dataset (v1) and verify licence (FG-03)
- Validate dataset integrity (images, labels, classes, duplicates) (FG-04)
- Detect and repair train/valid/test leakage (grouped re-split) (FG-05)
- Exploratory data analysis package (charts + review tables) (FG-06)
- Annotation-quality review and outlier report (FG-07)
- Create initial product backlog and risk register (FG-08)

## Sprint 2 - Baseline & model comparison

**Goal.** Create the baseline and compare architectures.

**Committed / completed:** 14 / 14 story points (6 items).

**Delivered:**

- Set up Python environment (CUDA PyTorch + Ultralytics) (FG-09)
- Train YOLOv8n baseline (transfer learning, 40 epochs) (FG-10)
- Train YOLOv8s comparison model (FG-11)
- Optional YOLO11n architecture comparison (FG-12)
- Analyse training curves and validation metrics (FG-13)
- Create experiment log and benchmark table (FG-14)

## Sprint 3 - Tuning & application

**Goal.** Select the final model and build the detection interface.

**Committed / completed:** 26 / 29 story points (8 items).

**Delivered:**

- Hyperparameter tuning probes (optimizer / augmentation / loss) (FG-15)
- Train tuned final model and select by recall-weighted criteria (FG-16)
- Streamlit app skeleton with sidebar controls and tabs (FG-17)
- Image upload detection with downloads (PNG/CSV/JSON) (FG-18)
- Video upload processing with progress and frame CSV (FG-19)
- Browser live webcam detection (streamlit-webrtc) (FG-20)
- OpenCV desktop webcam fallback (FG-21)
- Automated pytest suite for pipeline and app components (FG-22)

## Sprint 4 - Evaluation & delivery

**Goal.** Complete academic and submission deliverables.

**Committed / completed:** 22 / 22 story points (8 items).

**Delivered:**

- Confidence-threshold analysis and final threshold selection (FG-23)
- One-time test-set evaluation with full metric package (FG-24)
- Structured error analysis (TP/FP/FN/localization galleries) (FG-25)
- Ten sample predictions with interpretations (FG-26)
- Final report (DOCX + PDF) with real figures (FG-27)
- Presentation slides, speaker notes and demo script (FG-28)
- Scrum evidence package (board, burndown, retrospectives) (FG-29)
- Final submission ZIP and verification loop (FG-30)
