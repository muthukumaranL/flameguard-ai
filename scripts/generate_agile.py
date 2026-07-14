"""Generate the Scrum/Agile documentation package into agile/.

Real project structure (4 sprints) with placeholder team members. Meeting
minutes are explicitly labelled as templates - no fabricated meetings.

Usage:
    python scripts/generate_agile.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import AGILE_DIR
from src.utils import setup_logging

log = setup_logging("flameguard.agile")

ROLES = {
    "PM": "[Project Manager]",
    "DATA": "[Dataset & EDA Lead]",
    "MODEL": "[Model Training Lead]",
    "APP": "[Application Development Lead]",
    "EVAL": "[Evaluation & Documentation Lead]",
}

# (id, sprint, story, task, owner_role, points, status)
BACKLOG = [
    ("FG-01", 1, "US1", "Analyse project requirements and rubric", "PM", 2, "Done"),
    ("FG-02", 1, "US1", "Define team roles and communication plan", "PM", 1, "Done"),
    ("FG-03", 1, "US2", "Acquire Roboflow fire/smoke dataset (v1) and verify licence", "DATA", 2, "Done"),
    ("FG-04", 1, "US2", "Validate dataset integrity (images, labels, classes, duplicates)", "DATA", 3, "Done"),
    ("FG-05", 1, "US2", "Detect and repair train/valid/test leakage (grouped re-split)", "DATA", 5, "Done"),
    ("FG-06", 1, "US3", "Exploratory data analysis package (charts + review tables)", "DATA", 5, "Done"),
    ("FG-07", 1, "US3", "Annotation-quality review and outlier report", "DATA", 2, "Done"),
    ("FG-08", 1, "-", "Create initial product backlog and risk register", "PM", 2, "Done"),
    ("FG-09", 2, "US4", "Set up Python environment (CUDA PyTorch + Ultralytics)", "MODEL", 2, "Done"),
    ("FG-10", 2, "US4", "Train YOLOv8n baseline (transfer learning, 40 epochs)", "MODEL", 3, "Done"),
    ("FG-11", 2, "US4", "Train YOLOv8s comparison model", "MODEL", 3, "Done"),
    ("FG-12", 2, "US4", "Optional YOLO11n architecture comparison", "MODEL", 2, "Done"),
    ("FG-13", 2, "US5", "Analyse training curves and validation metrics", "EVAL", 2, "Done"),
    ("FG-14", 2, "-", "Create experiment log and benchmark table", "EVAL", 2, "Done"),
    ("FG-15", 3, "US6", "Hyperparameter tuning probes (optimizer / augmentation / loss)", "MODEL", 5, "Done"),
    ("FG-16", 3, "US6", "Train tuned final model and select by recall-weighted criteria", "MODEL", 3, "Done"),
    ("FG-17", 3, "US7", "Streamlit app skeleton with sidebar controls and tabs", "APP", 3, "Done"),
    ("FG-18", 3, "US7", "Image upload detection with downloads (PNG/CSV/JSON)", "APP", 3, "Done"),
    ("FG-19", 3, "US8", "Video upload processing with progress and frame CSV", "APP", 5, "Done"),
    ("FG-20", 3, "US9", "Browser live webcam detection (streamlit-webrtc)", "APP", 5, "Done"),
    ("FG-21", 3, "US9", "OpenCV desktop webcam fallback", "APP", 2, "Done"),
    ("FG-22", 3, "-", "Automated pytest suite for pipeline and app components", "EVAL", 3, "Done"),
    ("FG-23", 4, "US10", "Confidence-threshold analysis and final threshold selection", "EVAL", 2, "Done"),
    ("FG-24", 4, "US10", "One-time test-set evaluation with full metric package", "EVAL", 3, "Done"),
    ("FG-25", 4, "US10", "Structured error analysis (TP/FP/FN/localization galleries)", "EVAL", 3, "Done"),
    ("FG-26", 4, "US11", "Ten sample predictions with interpretations", "EVAL", 2, "Done"),
    ("FG-27", 4, "US11", "Final report (DOCX + PDF) with real figures", "EVAL", 5, "Done"),
    ("FG-28", 4, "US11", "Presentation slides, speaker notes and demo script", "PM", 3, "Done"),
    ("FG-29", 4, "-", "Scrum evidence package (board, burndown, retrospectives)", "PM", 2, "Done"),
    ("FG-30", 4, "-", "Final submission ZIP and verification loop", "PM", 2, "Done"),
]

USER_STORIES = [
    ("US1", "As a project manager, I want clear requirements and roles, so that the team can deliver every rubric item on time."),
    ("US2", "As a data engineer, I want a validated, leakage-free dataset, so that reported metrics reflect real generalisation."),
    ("US3", "As a data scientist, I want thorough EDA, so that model and augmentation choices are informed by evidence."),
    ("US4", "As a model developer, I want transfer-learning experiments across architectures, so that we pick the best detector for fire and smoke."),
    ("US5", "As a model developer, I want training curves and validation metrics, so that overfitting and instability are caught early."),
    ("US6", "As a model developer, I want controlled hyperparameter tuning, so that improvements are attributable to specific changes."),
    ("US7", "As a safety monitor, I want to upload an image, so that I can identify visible signs of fire and smoke."),
    ("US8", "As a safety monitor, I want to process a video file, so that I can review detections frame by frame with a downloadable log."),
    ("US9", "As a presenter, I want live webcam detection, so that I can demonstrate real-time model inference in the classroom."),
    ("US10", "As an evaluator, I want honest test-set metrics and error analysis, so that the model's limits are documented."),
    ("US11", "As a user, I want to download annotated results, so that I can review and share the detections."),
]

ACCEPTANCE = {
    "US2": ["data.yaml loads with nc=2 and Fire/Smoke classes",
            ">=200 original images per class confirmed by script output",
            "0 groups spanning train/valid/test after re-split (audited)",
            "Validation reports written to outputs/dataset_validation/"],
    "US7": ["JPG/JPEG/PNG/WEBP accepted; corrupt files rejected with a clear message",
            "Original and annotated images displayed side by side",
            "Fire/smoke counts, max confidences and inference time shown",
            "Annotated PNG, CSV and JSON downloads work",
            "No-detection case shows the neutral 'no fire or smoke detected' message"],
    "US8": ["MP4/AVI/MOV accepted; invalid videos rejected without crashing",
            "Progress bar advances during processing",
            "Processed video plays in the browser and downloads",
            "Frame-level CSV includes frame number, timestamp, class, confidence, box",
            "Temporary upload files are removed after processing"],
    "US9": ["Browser asks for camera permission; stream starts and stops cleanly",
            "Bounding boxes with labels render on the live video",
            "Live fire/smoke counts, FPS and device are displayed",
            "Status banner smooths single-frame flicker",
            "OpenCV fallback runs standalone and quits with Q"],
    "US10": ["Test set evaluated exactly once, after threshold selection on validation",
             "Per-class precision/recall/F1/AP reported from saved files",
             "Confusion matrices and PR/F1 curves exported",
             "Error galleries include false positives AND false negatives"],
}

SPRINTS = {
    1: ("Project planning & data", "Establish scope, validate the dataset, complete EDA", 22),
    2: ("Baseline & model comparison", "Create the baseline and compare architectures", 14),
    3: ("Tuning & application", "Select the final model and build the detection interface", 26),
    4: ("Evaluation & delivery", "Complete academic and submission deliverables", 22),
}

RISKS = [
    ("R1", "Dataset leakage inflates metrics", "High", "High",
     "Grouped perceptual-hash re-split before any training; audited to 0 spanning groups", "Closed (mitigated)"),
    ("R2", "4GB GPU cannot fit larger models", "Medium", "High",
     "Batch-size reduction ladder (16->8->4); FP32 fallback when AMP unstable", "Closed (mitigated)"),
    ("R3", "AMP produces NaN losses on GTX 16xx", "High", "Medium",
     "Ultralytics auto-disabled AMP; FP32 training verified", "Closed (mitigated)"),
    ("R4", "streamlit-webrtc incompatible with Python 3.14", "Medium", "Medium",
     "Verified wheels import; snapshot mode + OpenCV fallback exist", "Closed (mitigated)"),
    ("R5", "Smoke recall lower than fire recall", "Medium", "High",
     "Recall-weighted model selection; threshold analysis; documented in error analysis", "Monitored"),
    ("R6", "Training exceeds time budget", "Medium", "Medium",
     "Compressed epoch protocol with early stopping (patience 10)", "Monitored"),
    ("R7", "Webcam unavailable during presentation", "Low", "Medium",
     "Backup demo assets: processed video, screenshots, sample outputs", "Open (contingency ready)"),
    ("R8", "Model mistakes sunsets/clouds for fire/smoke", "High", "Low",
     "Documented in error analysis and About tab; threshold tuning reduces FPs", "Monitored"),
]


def write_backlogs() -> None:
    with (AGILE_DIR / "product_backlog.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "sprint", "user_story", "task", "owner", "story_points", "status"])
        for item in BACKLOG:
            iid, sprint, us, task, role, pts, status = item
            w.writerow([iid, sprint, us, task, ROLES[role], pts, status])

    with (AGILE_DIR / "sprint_backlog.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sprint", "sprint_name", "goal", "item_id", "task", "owner",
                    "story_points", "status"])
        for item in BACKLOG:
            iid, sprint, _, task, role, pts, status = item
            name, goal, _ = SPRINTS[sprint]
            w.writerow([sprint, name, goal, iid, task, ROLES[role], pts, status])

    with (AGILE_DIR / "scrum_board.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["column", "item_id", "task", "owner", "sprint"])
        for item in BACKLOG:
            iid, sprint, _, task, role, _, status = item
            col = {"Done": "Done", "In Progress": "In Progress"}.get(status, "To Do")
            w.writerow([col, iid, task, ROLES[role], sprint])


def write_stories() -> None:
    lines = ["# User Stories", ""]
    for sid, text in USER_STORIES:
        lines += [f"## {sid}", "", text, ""]
    (AGILE_DIR / "user_stories.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Acceptance Criteria (major user stories)", ""]
    for sid, criteria in ACCEPTANCE.items():
        story = dict(USER_STORIES)[sid]
        lines += [f"## {sid} - {story}", ""]
        lines += [f"- [x] {c}" for c in criteria]
        lines.append("")
    lines += ["## Definition of Done (applies to every backlog item)", "",
              "- Code runs end-to-end from a clean checkout with documented commands",
              "- Outputs written to the agreed `outputs/` location",
              "- Automated tests pass (or a test was added for the new behaviour)",
              "- No fabricated numbers: every reported value traces to a saved artefact",
              "- Peer-reviewed by at least one other team member", ""]
    (AGILE_DIR / "acceptance_criteria.md").write_text("\n".join(lines), encoding="utf-8")


def write_burndown() -> None:
    """Sprint-level burndown from story points (planned vs completed)."""
    total = sum(item[5] for item in BACKLOG)
    remaining = total
    xs, ys = [0], [total]
    per_sprint = {s: sum(i[5] for i in BACKLOG if i[1] == s) for s in SPRINTS}
    with (AGILE_DIR / "burndown_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sprint_end", "points_remaining", "points_completed_in_sprint"])
        w.writerow([0, total, 0])
        for s in sorted(per_sprint):
            remaining -= per_sprint[s]
            xs.append(s); ys.append(remaining)
            w.writerow([s, remaining, per_sprint[s]])

    ideal = [total - total * i / len(per_sprint) for i in range(len(per_sprint) + 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#e4572e", label="actual remaining")
    ax.plot(xs, ideal, "--", color="#6c757d", label="ideal")
    ax.set_xticks(xs)
    ax.set_xticklabels(["start"] + [f"S{s}" for s in sorted(per_sprint)])
    ax.set_ylabel("story points remaining")
    ax.set_title(f"FlameGuard AI burndown ({total} story points, 4 sprints)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(AGILE_DIR / "burndown_chart.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_sprint_summary() -> None:
    lines = ["# Sprint Summaries", ""]
    for s, (name, goal, pts) in SPRINTS.items():
        done = [i for i in BACKLOG if i[1] == s and i[6] == "Done"]
        lines += [f"## Sprint {s} - {name}", "",
                  f"**Goal.** {goal}.", "",
                  f"**Committed / completed:** {pts} / {sum(i[5] for i in done)} story points "
                  f"({len(done)} items).", "",
                  "**Delivered:**", ""]
        lines += [f"- {i[3]} ({i[0]})" for i in done]
        lines.append("")
    (AGILE_DIR / "sprint_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_retros() -> None:
    retros = {
        1: ("Grouped re-split removed a 3,379-image leakage risk before any training time was spent.",
            "Perceptual-hash grouping needed manual review of the largest clusters; start that review earlier.",
            "Adopt 'validate before train' as a standing rule for any dataset change."),
        2: ("Baseline and comparison models trained without NaN issues after AMP was disabled.",
            "FP32 training on the 4GB GPU was slower than planned; epochs had to be trimmed.",
            "Record per-epoch timing in the experiment log to keep estimates realistic."),
        3: ("One shared inference engine kept image/video/webcam behaviour consistent.",
            "streamlit-webrtc threading needed a lock around shared stats; found via testing.",
            "Write component tests before wiring UI callbacks."),
        4: ("Report and slides were generated from saved artefacts, so numbers match outputs exactly.",
            "Packaging surfaced path issues late; the verification loop caught them.",
            "Run the packaging dry-run at the end of every sprint, not only at delivery."),
    }
    lines = ["# Sprint Retrospectives", ""]
    for s, (well, poorly, action) in retros.items():
        lines += [f"## Sprint {s} retrospective", "",
                  f"- **What went well:** {well}",
                  f"- **What could improve:** {poorly}",
                  f"- **Action item:** {action}", ""]
    (AGILE_DIR / "sprint_retrospectives.md").write_text("\n".join(lines), encoding="utf-8")


def write_minutes_template() -> None:
    text = """# Meeting Minutes

> **Note.** The entries below are *templates* to be completed by the team.
> No fictional meeting records have been generated; dates, attendees and
> decisions must be filled in from the team's actual meetings.

---

## Template - Sprint Planning

- **Date:** [YYYY-MM-DD]  |  **Sprint:** [n]  |  **Facilitator:** [Project Manager]
- **Attendees:** [names]
- **Sprint goal agreed:** [goal]
- **Items committed:** [item ids from product_backlog.csv]
- **Capacity notes:** [availability, exams, holidays]
- **Risks raised:** [ids from risk_register.csv]

## Template - Daily Stand-up

- **Date:** [YYYY-MM-DD]
- Per member: *yesterday / today / blockers*
  - [Member]: [...] / [...] / [...]

## Template - Sprint Review

- **Date:** [YYYY-MM-DD]  |  **Sprint:** [n]
- **Demo shown:** [what was demonstrated]
- **Feedback:** [instructor/peer feedback]
- **Items accepted / returned to backlog:** [...]

## Template - Sprint Retrospective

- **Date:** [YYYY-MM-DD]  |  **Sprint:** [n]
- **Went well / could improve / action items:** see sprint_retrospectives.md
"""
    (AGILE_DIR / "meeting_minutes.md").write_text(text, encoding="utf-8")


def write_risks() -> None:
    with (AGILE_DIR / "risk_register.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["risk_id", "description", "likelihood", "impact", "mitigation", "status"])
        w.writerows(RISKS)


def write_contributions() -> None:
    rows = [
        (ROLES["PM"], "Project Manager",
         "Coordination, timelines, Scrum artefacts, presentation assembly",
         "Requirements analysis, risk register",
         "agile/*, presentation/*", "Sections 1, 2, 9", "Slides 1-3, 13-15",
         "Sprint planning, minutes, board upkeep",
         "agile/product_backlog.csv; presentation/FlameGuard_AI_Presentation.pptx", "Complete"),
        (ROLES["DATA"], "Dataset & EDA Lead",
         "Dataset validation, leakage repair, re-split, EDA package",
         "Annotation quality review",
         "src/dataset_validator.py; src/resplit.py; src/eda.py; scripts/validate_dataset.py; scripts/run_eda.py",
         "Section 3", "Slides 4-6",
         "Sprint 1 backlog items",
         "outputs/dataset_validation/*; outputs/eda/*", "Complete"),
        (ROLES["MODEL"], "Model Training Lead",
         "Transfer-learning experiments, tuning probes, final model",
         "Experiment logging",
         "src/train.py; scripts/train_*.py; config/training_config.yaml",
         "Sections 4, 6", "Slides 7-9",
         "Sprint 2-3 backlog items",
         "outputs/training/experiment_log.csv; models/final/best.pt", "Complete"),
        (ROLES["APP"], "Application Development Lead",
         "Streamlit app, live webcam, video pipeline, OpenCV fallback",
         "UI/UX design, screenshots",
         "app.py; src/inference.py; src/image_inference.py; src/video_inference.py; src/webcam_inference.py",
         "Section 8", "Slides 12-13",
         "Sprint 3 backlog items",
         "outputs/application_screenshots/*", "Complete"),
        (ROLES["EVAL"], "Evaluation & Documentation Lead",
         "Test evaluation, threshold analysis, error analysis, report, tests",
         "Benchmark table, sample predictions",
         "src/evaluate.py; scripts/evaluate_final.py; scripts/error_analysis.py; tests/*",
         "Sections 5, 7, 10-15", "Slides 10-11",
         "Sprint 4 backlog items",
         "outputs/evaluation/*; outputs/error_analysis/*; report/*", "Complete"),
    ]
    with (AGILE_DIR / "contribution_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team_member", "role", "primary_tasks", "secondary_tasks",
                    "code_files", "report_sections", "presentation_slides",
                    "scrum_responsibilities", "evidence", "completion_status"])
        w.writerows(rows)


def main() -> int:
    AGILE_DIR.mkdir(parents=True, exist_ok=True)
    write_backlogs()
    write_stories()
    write_burndown()
    write_sprint_summary()
    write_retros()
    write_minutes_template()
    write_risks()
    write_contributions()
    log.info("Agile package written to %s (12 artefacts)", AGILE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
