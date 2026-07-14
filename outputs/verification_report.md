# FlameGuard AI - Verification Report

**13/26 checks passed.**

| Check | Result | Detail |
|---|---|---|
| Dataset: repaired split exists | PASS | train/valid/test = 3707/1060/533 |
| Dataset: >=200 original images/class | PASS | {'Fire': 2535, 'Smoke': 1644} |
| Dataset: zero cross-split leakage after repair | PASS | 3,379 leaked images repaired -> 0 spanning groups |
| EDA: figures generated | PASS | 16 figures |
| Model: final best.pt exists | PASS | models\final\best.pt (6.0 MB) |
| Model: metadata with real metrics | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\models\\final\\model_metadata.yaml' |
| Model: loads and infers on CPU | PASS | CPU inference 4082 ms, empty-path OK |
| Training: >=3 experiments logged | PASS | 3 runs: e1_baseline_v8n, e4d_probe_baseline, e4a_probe_adamw |
| Evaluation: test metrics present | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\evaluation\\metrics_test.json' |
| Evaluation: threshold chosen on validation | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\evaluation\\threshold_analysis.csv' |
| Evaluation: measured inference speed | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\evaluation\\inference_speed.json' |
| Benchmark: table + selection report | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\benchmarking\\benchmark_table.csv' |
| Error analysis: failures documented | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\error_analysis\\error_summary.json' |
| Samples: >=10 with successes AND failures | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\sample_outputs\\sample_summary.csv' |
| App: imports cleanly | PASS | app.py compiles |
| App: screenshots captured | PASS | 14 screenshots |
| Manual tests recorded | PASS | 13 manual tests, 0 failed |
| Automated tests passed | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\outputs\\test_report.txt' |
| Report: DOCX opens | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\report\\FlameGuard_AI_Final_Report.docx' |
| Report: PDF opens | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\report\\FlameGuard_AI_Final_Report.pdf' |
| Report: no unfilled metric placeholders | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\report\\report_content.md' |
| Presentation: PPTX opens with >=13 slides | **FAIL** | FileNotFoundError: [Errno 2] No such file or directory: 'E:\\college\\pg\\second course\\new\\Applied A.I. Solutions Development\\Deep Learning 2\\FlameGuard_AI\\presentation\\FlameGuard_AI_Presentation.pptx' |
| Presentation: notes + demo script | **FAIL** | notes=False demo_script=False backup_assets=True |
| Scrum: artefacts complete | PASS | 10/10 present |
| Notebooks present | PASS | 7 notebooks |
| No absolute personal paths in source | PASS | files with hard-coded user paths: none |
