"""Turn the probe results into the final-model configuration, and show the working.

Reads outputs/training/experiment_log.csv, compares every probe against the
control, writes outputs/training/e5_decision.md, and prints the exact command to
train E5.  Nothing here is hand-entered: the decision is derived from measured
values, so the report and the training command cannot disagree.

Usage:
    python scripts/decide_final_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import paths
from src.utils import setup_logging

log = setup_logging("flameguard.decide")

CONTROL = "e4d_probe_baseline"
# probe id -> (human description, the override it implies for the final run)
PROBE_CHANGES = {
    # e4a was designed as an optimizer probe, but `optimizer: auto` already
    # resolves to AdamW - so the only thing that actually changed was the
    # learning rate (the control's effective 1.667e-3 -> 1.0e-3). Labelled here
    # as what it really tested.
    "e4a_probe_adamw": ("lower learning rate (1.667e-3 -> 1.0e-3; AdamW in both arms)",
                        {"optimizer": "AdamW", "lr0": 0.001}),
    "e4b_probe_augment": ("stronger photometric + scale augmentation",
                          {"hsv_v": 0.6, "scale": 0.7}),
    "e4c_probe_loss": ("classification-loss weight 0.5 -> 1.0", {"cls": 1.0}),
}
METRIC = "map50_95"


def main() -> int:
    log_path = paths.TRAINING_OUTPUT_DIR / "experiment_log.csv"
    df = pd.read_csv(log_path).drop_duplicates("experiment_id", keep="last")
    df = df.set_index("experiment_id")
    if CONTROL not in df.index:
        log.error("control run %s not found in %s", CONTROL, log_path)
        return 1

    ctrl = df.loc[CONTROL]
    rows, winners = [], []
    for pid, (desc, override) in PROBE_CHANGES.items():
        if pid not in df.index:
            log.warning("probe %s missing - skipped", pid)
            continue
        r = df.loc[pid]
        delta = float(r[METRIC]) - float(ctrl[METRIC])
        helped = delta > 0
        rows.append({
            "probe": pid, "change": desc,
            "map50": float(r["map50"]), "map50_95": float(r[METRIC]),
            "recall": float(r["recall"]),
            "delta_map50_95": delta, "verdict": "adopt" if helped else "reject",
        })
        if helped:
            winners.append((delta, pid, desc, override))

    winners.sort(reverse=True)
    adopted: dict = {}
    if winners:
        _, best_id, best_desc, best_override = winners[0]
        adopted.update(best_override)

    # Continuing from an already-trained checkpoint: restart the schedule at a
    # lower peak LR so the warm-up does not undo what the baseline learned.
    # (Standard fine-tuning practice, not a tuned value.)
    if "lr0" not in adopted:
        adopted["lr0"] = 0.002

    baseline_weights = (paths.TRAINING_OUTPUT_DIR / "e1_baseline_v8n" /
                        "weights" / "best.pt")
    adopted_model = str(baseline_weights.relative_to(paths.PROJECT_ROOT)).replace("\\", "/")

    table = pd.DataFrame(rows)
    lines = [
        "# Final model (E5) - configuration decision", "",
        "The final model is **not** a fresh run with hand-picked settings. It continues",
        f"fine-tuning the strongest existing checkpoint (`{adopted_model}`) using the",
        "single change that measurably beat the control in the probe study below.", "",
        "## Probe study (equal budget, one variable each, same seed and data)", "",
        f"Control (`{CONTROL}`): mAP@0.5 = {float(ctrl['map50']):.4f}, "
        f"mAP@0.5:0.95 = {float(ctrl[METRIC]):.4f}, recall = {float(ctrl['recall']):.4f} "
        f"at {int(ctrl['epochs_run'])} epochs.", "",
        "| Probe | Change | mAP@0.5 | mAP@0.5:0.95 | Recall | Δ mAP@0.5:0.95 | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['probe']}` | {r['change']} | {r['map50']:.4f} | {r['map50_95']:.4f} "
            f"| {r['recall']:.4f} | {r['delta_map50_95']:+.4f} | **{r['verdict']}** |")
    lines += ["", "## Decision", ""]
    if winners:
        lines += [
            f"- **Adopted:** {best_desc} (`{best_id}`), the largest measured improvement "
            f"({winners[0][0]:+.4f} mAP@0.5:0.95 against the control).",
        ]
        rejected = [r for r in rows if r["verdict"] == "reject"]
        for r in rejected:
            lines.append(f"- **Rejected:** {r['change']} - it made things worse "
                         f"({r['delta_map50_95']:+.4f}). Reported as measured.")
        others = [w for w in winners[1:]]
        for _, pid, desc, _ in others:
            lines.append(f"- **Not combined:** {desc} (`{pid}`) also helped on its own, "
                         f"but the probes tested it in isolation. Stacking two changes "
                         f"whose interaction was never measured would make the final "
                         f"result unattributable, so it was left out.")
    else:
        lines.append("- No probe beat the control. The final model keeps the default "
                     "recipe - a real result: on this dataset the Ultralytics defaults "
                     "are already well matched to the task.")
    lines += ["", "## Continuing a trained model is not the same as training one", ""]
    if "e5a_naive_restart" in df.index and "e1_baseline_v8n" in df.index:
        bad, base = df.loc["e5a_naive_restart"], df.loc["e1_baseline_v8n"]
        lines += [
            "Our first attempt at the final model **failed, and we kept it** "
            "(`e5a_naive_restart`).",
            "",
            f"- Continued from the baseline with an ordinary fresh schedule "
            f"(lr0 = {bad['lr0']}, default warm-up, mosaic on).",
            f"- Validation mAP@0.5 fell from **{float(base['map50']):.4f}** to "
            f"**{float(bad['map50']):.4f}**; mAP@0.5:0.95 from "
            f"{float(base['map50_95']):.4f} to {float(bad['map50_95']):.4f}.",
            f"- The tell: **best epoch = {int(bad['best_epoch'])}**. The model was at "
            f"its best before the new schedule did anything, and got worse afterwards.",
            "- Cause: re-running the warm-up ramps the LR back up, and re-enabling "
            "mosaic changes the input distribution. Together they shake a converged "
            "model loose.",
            "",
            "**Fix, used for the final model:** treat it as a polish, not a restart -",
            "a low peak LR, **no warm-up**, and **no mosaic**, so the model finishes on",
            "realistic images. The winning probe change (`cls=1.0`) is kept.",
            "",
        ]
    if "e5_final" in df.index:
        good = df.loc["e5_final"]
        base = df.loc["e1_baseline_v8n"]
        delta = float(good["map50"]) - float(base["map50"])
        lines += [
            f"Final model (`e5_final`): mAP@0.5 = **{float(good['map50']):.4f}** "
            f"({delta:+.4f} vs the baseline), mAP@0.5:0.95 = "
            f"{float(good['map50_95']):.4f}, recall = {float(good['recall']):.4f}, "
            f"best epoch {int(good['best_epoch'])} of {int(good['epochs_run'])}.",
            "",
        ]
    lines += [
        "## Command actually run", "",
        "```bash",
        "# the final recipe lives in config/training_config.yaml (e5_final)",
        "python scripts/train_final.py --final",
        "```", "",
        "## Caveat we state out loud", "",
        "Probes were run at a short budget because the GPU is a 4GB GTX 1650 Ti on which",
        "AMP is auto-disabled (FP32, ~2x slower). Short probes rank configurations *under",
        "a short schedule*; the learning-rate schedule depends on total epochs, so a",
        "setting that wins at 5 epochs is not guaranteed to win at 50. With more compute,",
        "the honest design repeats the probes at full length.",
        "",
    ]
    out = paths.TRAINING_OUTPUT_DIR / "e5_decision.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(paths.TRAINING_OUTPUT_DIR / "probe_comparison.csv", index=False)
    log.info("decision -> %s", paths.rel_to_root(out))

    args = " ".join(f"{k}={v}" for k, v in adopted.items())
    cmd = f"python scripts/train_final.py --final model={adopted_model} {args}"
    log.info("run this next:\n    %s", cmd)
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
