# Acceptance Criteria (major user stories)

## US2 - As a data engineer, I want a validated, leakage-free dataset, so that reported metrics reflect real generalisation.

- [x] data.yaml loads with nc=2 and Fire/Smoke classes
- [x] >=200 original images per class confirmed by script output
- [x] 0 groups spanning train/valid/test after re-split (audited)
- [x] Validation reports written to outputs/dataset_validation/

## US7 - As a safety monitor, I want to upload an image, so that I can identify visible signs of fire and smoke.

- [x] JPG/JPEG/PNG/WEBP accepted; corrupt files rejected with a clear message
- [x] Original and annotated images displayed side by side
- [x] Fire/smoke counts, max confidences and inference time shown
- [x] Annotated PNG, CSV and JSON downloads work
- [x] No-detection case shows the neutral 'no fire or smoke detected' message

## US8 - As a safety monitor, I want to process a video file, so that I can review detections frame by frame with a downloadable log.

- [x] MP4/AVI/MOV accepted; invalid videos rejected without crashing
- [x] Progress bar advances during processing
- [x] Processed video plays in the browser and downloads
- [x] Frame-level CSV includes frame number, timestamp, class, confidence, box
- [x] Temporary upload files are removed after processing

## US9 - As a presenter, I want live webcam detection, so that I can demonstrate real-time model inference in the classroom.

- [x] Browser asks for camera permission; stream starts and stops cleanly
- [x] Bounding boxes with labels render on the live video
- [x] Live fire/smoke counts, FPS and device are displayed
- [x] Status banner smooths single-frame flicker
- [x] OpenCV fallback runs standalone and quits with Q

## US10 - As an evaluator, I want honest test-set metrics and error analysis, so that the model's limits are documented.

- [x] Test set evaluated exactly once, after threshold selection on validation
- [x] Per-class precision/recall/F1/AP reported from saved files
- [x] Confusion matrices and PR/F1 curves exported
- [x] Error galleries include false positives AND false negatives

## Definition of Done (applies to every backlog item)

- Code runs end-to-end from a clean checkout with documented commands
- Outputs written to the agreed `outputs/` location
- Automated tests pass (or a test was added for the new behaviour)
- No fabricated numbers: every reported value traces to a saved artefact
- Peer-reviewed by at least one other team member
