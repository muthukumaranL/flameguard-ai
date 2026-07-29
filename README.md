# FlameGuard AI

**Real-Time Fire and Smoke Detection Using Transfer Learning**
AASD 4014 — Deep Learning II · Final Project · Group `[GROUP NUMBER]`

FlameGuard AI detects **Fire** and **Smoke** in uploaded images, uploaded videos and a
live camera stream. The detector is a YOLO model **fine-tuned on a custom fire/smoke
dataset** (transfer learning from COCO-pretrained weights — no unchanged pretrained
model is used anywhere), and it runs entirely **locally**: no paid inference API, and no
image ever leaves the machine.

> **FlameGuard AI is an educational computer-vision prototype. It is not a certified
> fire-detection or emergency-response system and must not replace smoke detectors, fire
> alarms, emergency procedures, or human supervision.**

---

## 1. Features

| Feature | What it does |
|---|---|
| **Live camera** | Browser webcam detection (streamlit-webrtc) with live boxes, fire/smoke counts, measured FPS and a flicker-smoothed status banner |
| **Image detection** | JPG/PNG/WEBP upload → annotated result, counts, confidences, timing → download PNG / CSV / JSON |
| **Video detection** | MP4/AVI/MOV upload → frame-by-frame processing with progress bar and frame-skip control → H.264 output + per-frame CSV |
| **Model performance** | Reads the *actual* saved evaluation artefacts — no hard-coded numbers |
| **OpenCV fallback** | `python src/webcam_inference.py` — native window, works with no browser |
| **Reproducible pipeline** | One command each for dataset repair, EDA, training, evaluation, error analysis, report and slides |

## 2. System architecture

```
 dataset (Roboflow, CC BY 4.0)
        │
        ▼
 scripts/validate_dataset.py ──► integrity audit + leakage-repaired re-split
        │                        (data/processed/fire_smoke_resplit)
        ▼
 scripts/run_eda.py ───────────► outputs/eda/
        │
        ▼
 scripts/train_*.py ───────────► transfer learning (Ultralytics YOLO)
        │                        outputs/training/<experiment>/
        ▼
 scripts/benchmark.py ─────────► model selection ──► models/final/best.pt
        │
        ▼
 scripts/evaluate_final.py ────► threshold choice (validation) + ONE test evaluation
 scripts/error_analysis.py ────► outputs/error_analysis/
        │
        ▼
 app.py  ──►  src/inference.py (one cached engine)
              ├── src/image_inference.py   (Image tab)
              ├── src/video_inference.py   (Video tab)
              └── src/webcam_inference.py  (Live tab + OpenCV fallback)
```

Every detection path — image, video, webcam — calls the **same** `DetectionEngine.predict`,
so behaviour cannot silently diverge between modes.

## 3. Dataset

- **Source:** [Roboflow Universe — *fire and smoke* v1](https://universe.roboflow.com/fire-detector-cqdzi/fire-and-smoke-b5lli/dataset/1)
  (`fire-detector-cqdzi/fire-and-smoke-b5lli`), **CC BY 4.0**
- **Classes:** `0 = Fire`, `1 = Smoke` · **5,300 images**, 640×640, YOLOv8 format
- **The published split leaks.** Mirrored/noise-augmented copies of the same source
  images — and sequential video frames — appear in more than one of train/valid/test.
  We measured it, then rebuilt the splits **group-wise** (perceptual-hash clustering,
  seed 42) so no source image appears in two splits. **All reported results use the
  repaired splits.** See `outputs/dataset_validation/resplit_report.json`.

### Dataset placement

The dataset is **not** in this repository. Download the v1 YOLOv8 export, extract it, then:

```bash
python scripts/validate_dataset.py --source "/path/to/fire and smoke.v1i.yolov8"
```

This copies it to `data/raw/` (originals untouched), audits it, and writes the repaired
splits to `data/processed/fire_smoke_resplit/`. If the export sits next to the project
folder, `--source` can be omitted.

## 4. Environment setup

Requires **Python 3.10–3.14**. A CUDA GPU is optional (CPU works, just slower).

**Windows**

```bat
scripts\setup_environment.bat
```

**macOS / Linux / Git Bash**

```bash
bash scripts/setup_environment.sh
```

The script creates `.venv`, installs a **CUDA build of PyTorch** if one is available for
your platform (falling back to CPU wheels automatically), then installs
`requirements.txt`. It keeps its pip cache and temp files inside the project, so it will
not fill your system drive.

Verify:

```bash
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**GPU note:** on GTX 16xx cards Ultralytics automatically disables mixed precision (AMP
produces NaN losses on that series), so training runs in FP32 and takes roughly twice as
long. This is expected, not a misconfiguration.

**CPU fallback:** everything works on CPU. `pick_device()` selects CUDA only when a
device is genuinely usable, and the app displays which device it is on.

## 5. Reproducing every result

Run in order. Each command writes its artefacts under `outputs/`.

```bash
python scripts/validate_dataset.py     # audit + leakage-repaired re-split
python scripts/run_eda.py              # EDA charts, tables, augmentation preview

python scripts/train_baseline.py       # E1 - YOLOv8n baseline (transfer learning)
python scripts/run_training_chain.py   # E4a-d tuning probes, E3 YOLO11n, E2 YOLOv8s
python scripts/train_final.py --final model=outputs/training/e1_baseline_v8n/weights/best.pt
                                       # E5 - tuned final model

python scripts/benchmark.py            # compare on validation → models/final/best.pt
python scripts/evaluate_final.py       # threshold on validation, then ONE test evaluation
python scripts/error_analysis.py       # TP/FP/FN/localisation galleries
python scripts/generate_samples.py     # the 10 sample predictions

python -m pytest tests/ -q             # automated test suite
python scripts/generate_agile.py       # Scrum artefacts
python scripts/generate_report.py      # report: Markdown + DOCX + PDF
python scripts/generate_slides.py      # deck + speaker notes + demo script
python scripts/package_submission.py --group 07
```

**The test split is evaluated exactly once**, by `evaluate_final.py`, after the model and
the confidence threshold have been fixed on validation data.

## 6. Running the application

```bash
streamlit run app.py
# or: scripts\run_app.bat   /   bash scripts/run_app.sh
```

Then open <http://localhost:8501>.

- **Webcam permissions:** the browser will ask for camera access. It only works on
  `localhost` or over HTTPS. If the camera is blocked, in use by another application, or
  the WebRTC connection fails, use the fallback:

  ```bash
  python src/webcam_inference.py                        # press Q to quit
  python src/webcam_inference.py --camera 1 --conf 0.4  # a different camera
  python src/webcam_inference.py --camera outputs/sample_inputs/demo_clip.mp4
  python src/webcam_inference.py --selftest 5           # headless check, no window
  ```

  `--camera` also accepts a video file, which makes the fallback usable as a
  headless CLI detector and lets it be verified on a machine with no camera.
  Exit codes: `0` ok, `2` model missing, `3` capture source unavailable, `4` no frames.

- **Image upload:** Image Detection tab → upload JPG/JPEG/PNG/WEBP → annotated result plus
  PNG / CSV / JSON downloads. If nothing is found, the app says so — it never claims the
  scene is safe.
- **Video upload:** Video Detection tab → MP4/AVI/MOV/MKV. Use the sidebar **frame skip**
  control to trade temporal coverage for speed. Output is re-encoded to H.264 (via the
  ffmpeg binary bundled with `imageio-ffmpeg`, so no separate install is needed).

### Output locations

| Path | Contents |
|---|---|
| `outputs/dataset_validation/` | Integrity audit, duplicate report, leakage/re-split audit |
| `outputs/eda/` | All EDA figures, `eda_summary.json`, annotation-review tables |
| `outputs/training/` | Per-experiment runs, curves, `experiment_log.csv` |
| `outputs/benchmarking/` | `benchmark_table.csv`, comparison chart, selection report |
| `outputs/evaluation/` | Test metrics, confusion matrices, PR/F1 curves, threshold sweep, speed |
| `outputs/error_analysis/` | Per-image error CSV, TP/FP/FN/localisation galleries |
| `outputs/sample_outputs/` | The 10 sample predictions + `sample_summary.csv` |
| `outputs/application_screenshots/` | UI screenshots |
| `models/final/` | `best.pt` + `model_metadata.yaml` (metrics, thresholds, config) |

## 7. Tests

```bash
python -m pytest tests/ -q
```

Covers configuration and path handling, `data.yaml` parsing, dataset structure,
annotation validation (valid *and* malformed labels), model loading, the missing-model
error path, CPU inference, the no-detection case, corrupt-image handling, video open /
invalid-video / end-to-end processing, output generation, temp-file hygiene, webcam
components (status smoothing, graceful handling of an absent camera), and that every
module and `app.py` import cleanly.

Results are saved to `outputs/test_report.txt`. Manual test cases and their evidence are
recorded in the report's appendix.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Model not found at models/final/best.pt` | Training has not been run. Run the pipeline in §5, or drop a trained `.pt` there. |
| Training is very slow | Expected on GTX 16xx: AMP is auto-disabled (NaN-loss bug), so FP32 is used. |
| CUDA out of memory | Lower `batch` in `config/training_config.yaml`. Measured on a 4 GB card, YOLOv8s needs 7.94 GB @batch 8 and 6.08 GB @batch 4 (both spill to shared RAM) — only **batch 2** fits (~1.0 GB). See `scripts/vram_probe.py`. |
| `Dataset '...' images not found` | Run `scripts/validate_dataset.py` first; it writes the resolved `data.yaml`. |
| Browser camera never starts | Not on localhost/HTTPS, permission denied, camera in use, or WebRTC blocked. Use `python src/webcam_inference.py`. |
| Processed video will not play in the browser | The H.264 conversion failed. Check that `imageio-ffmpeg` is installed; the file still downloads and plays in VLC. |
| Model Performance tab says "Result file not available" | Correct behaviour — the evaluation pipeline has not been run yet. |

## 9. Limitations

- Trained on ~5.3k images skewed toward outdoor wildfire and web imagery. Industrial
  CCTV, thermal ranges and unusual camera geometries are **unvalidated**.
- **The model has learned a strong colour prior, not the structure of flame.** A plain
  orange rectangle — no fire, no texture, no structure — is detected as *Fire* with high
  confidence, while random noise is correctly ignored. This is measured, not guessed:
  see `outputs/error_analysis/colour_prior_probe.json` (produced by
  `scripts/error_analysis.py`) and §10.3.1 of the report. It is the single best
  explanation for the false positives below.
- **Known false positives:** cloud banks, haze, sunset glow, warm artificial lighting,
  flat warm-toned surfaces.
- **Known false negatives:** thin/transparent smoke, small distant flames, night-time fires.
- Single-frame detection only — no temporal reasoning, so static haze cannot be
  distinguished from a stationary plume by motion.
- Detects the *visual signature* of fire, so it will happily fire on a photograph of a fire.

## 10. Ethics and safety

- **Local processing.** No frame, image or video leaves the machine. Nothing is uploaded.
- **No identity recognition.** The model has exactly two output classes, Fire and Smoke.
  It performs no face detection and no person tracking, and is structurally incapable of
  doing so.
- **No false reassurance.** "No Hazard Detected" means nothing was found above the
  threshold in that frame — *not* that the area is safe. The UI never says "safe".
- **Error asymmetry is a value judgement, made explicitly.** A missed fire is worse than a
  false alarm, so model selection and the default threshold are biased toward recall.
- FlameGuard AI is an **educational prototype**, not a safety product. See
  [LICENSE_NOTES.md](LICENSE_NOTES.md) for dataset and model licensing.
