# FlameGuard AI - Classroom Demo Script

**Before the session (do this, do not skip it)**

1. `cd FlameGuard_AI` and activate the environment.
2. `python -m pytest tests/ -q` - confirm the suite passes.
3. `streamlit run app.py` - confirm the app loads and the sidebar shows
   "Model loaded" and the correct device.
4. Have `outputs/sample_inputs/` open in a file browser - the demo images are there.
5. Have `presentation/backup_demo/` ready in case the live demo fails.
6. Plug in the laptop. GPU inference on battery is throttled.

**Demo sequence (about 4 minutes)**

| # | Action | What to say |
|---|--------|-------------|
| 1 | Show the sidebar | "The model is loaded once and cached; it is running on the GPU. Confidence is at 0.30 - we chose that on validation data, not by accepting a default." |
| 2 | Image tab → upload a fire-only sample | "Original on the left, detections on the right. Counts, peak confidence and inference time are measured, not estimated." |
| 3 | Upload a smoke-only sample | "Smoke is the harder class - it is semi-transparent and has no crisp boundary. Watch the box hug the dense core." |
| 4 | Upload a both-classes sample | "Fire and smoke together, which is the realistic case: smoke is usually the earlier signal." |
| 5 | Upload a negative or difficult sample | "This one it gets wrong - and this is the slide I want you to remember. It reads the cloud bank as smoke. We know this failure mode because we measured it." |
| 6 | Drag the confidence slider up | "Raising the threshold kills that false positive - and also starts killing real detections. That trade-off is the actual product decision." |
| 7 | Video tab → short clip, frame skip = 2 | "One frame at a time, so memory stays flat regardless of video length. Progress bar is real. The per-frame CSV downloads with timestamps." |
| 8 | Live Camera → START, allow permission | "This is the same engine, running on the webcam feed. Fire and smoke counts and measured FPS update live. The status banner is smoothed over five frames so it does not strobe." |
| 9 | Hold up a phone showing a fire image | "It fires on a screen showing fire - which is worth noting: the model detects the visual signature of fire, not fire itself." |
| 10 | Model Performance tab | "Every number here is read from the evaluation files. If we had not run the pipeline, it would say so rather than showing a number." |
| 11 | About tab | Read the disclaimer aloud. |

**Fallbacks (rehearse these too)**

- Browser will not grant camera access → `python src/webcam_inference.py`
  (native OpenCV window, press Q to quit).
- No camera at all → play the pre-processed video from `presentation/backup_demo/`.
- Everything is slow → set frame skip to "every 3rd frame", drop to a smaller clip.
- The app will not start → walk through `outputs/sample_outputs/sample_grid.png`
  and the screenshots in `outputs/application_screenshots/`.

**Never** debug live in front of the class. Announce the fallback and continue.
