# FlameGuard AI - Verification Report

**27/27 checks passed.**

| Check | Result | Detail |
|---|---|---|
| Dataset: repaired split exists | PASS | train/valid/test = 3707/1060/533 |
| Dataset: >=200 original images/class | PASS | {'Fire': 2535, 'Smoke': 1644} |
| Dataset: zero cross-split leakage after repair | PASS | 3,379 leaked images repaired -> 0 spanning groups |
| EDA: figures generated | PASS | 16 figures |
| Model: final best.pt exists | PASS | models\final\best.pt (6.0 MB) |
| Model: metadata with real metrics | PASS | conf=0.3, model=Tuned final |
| Model: loads and infers on CPU | PASS | CPU inference 2528 ms, empty-path OK |
| Training: >=3 experiments logged | PASS | 8 runs: e1_baseline_v8n, e4d_probe_baseline, e4a_probe_adamw, e4b_probe_augment, e4c_probe_loss, e3_compare_11n, e5a_naive_restart, e5_final |
| Evaluation: test metrics present | PASS | mAP50=0.492 R=0.426 on 533 images |
| Evaluation: threshold chosen on validation | PASS | 5 thresholds swept |
| Evaluation: measured inference speed | PASS | gpu: 34.4 FPS; cpu: 4.1 FPS |
| Benchmark: table + selection report | PASS | 3 models, winner=e5_final |
| Error analysis: failures documented | PASS | FP=118 FN=306, 4 galleries |
| Samples: >=10 with successes AND failures | PASS | 10 samples: 8 correct, 2 incorrect/partial |
| App: imports cleanly | PASS | app.py compiles |
| App: screenshots captured | PASS | 14 screenshots |
| Manual tests recorded | PASS | 13 manual tests, 0 failed |
| Automated tests passed | PASS | ======================== 65 passed, 1 skipped in 9.84s ======================== |
| Report: DOCX opens | PASS | 298 paragraphs, 13 tables, 28 figures |
| Report: PDF opens | PASS | 39 pages |
| Report: no unfilled metric placeholders | PASS | stray placeholders: none |
| Presentation: PPTX opens with >=13 slides | PASS | 15 slides, 15 with speaker notes |
| Presentation: notes + demo script | PASS | notes=True demo_script=True backup_assets=True |
| Scrum: artefacts complete | PASS | 10/10 present |
| Notebooks present | PASS | 7 notebooks |
| No absolute personal paths in source | PASS | files with hard-coded user paths: none |
| Submission ZIP: Group##_FlameGuard_AI_Final_Submission.zip | PASS | 539 entries, 171.1 MB, missing=none, forbidden=0 |
