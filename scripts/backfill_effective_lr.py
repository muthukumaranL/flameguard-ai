"""Correct the experiment log to record the optimizer/LR that were ACTUALLY used.

Runs launched before we discovered the `optimizer: auto` behaviour logged their
*requested* lr0 (e.g. 0.01) even though Ultralytics silently substituted its own
(AdamW at 1.667e-3). This reads the optimizer line that each run printed to its
console log and writes the real values back into experiment_log.csv.

This corrects a reporting error - it does not change any measured metric.

Usage:
    python scripts/backfill_effective_lr.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import paths
from src.utils import setup_logging

log = setup_logging("flameguard.backfill")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUN_START = re.compile(r"===\s+(\w+)\s+\|\s+model=")
OPTIMIZER = re.compile(r"optimizer:\s*(AdamW|SGD|Adam|RMSProp|NAdam|RAdam)\(lr=([0-9.eE+-]+)")

CONSOLE_LOGS = ["e1_console.log", "chain_console.log", "e5_console.log"]


def parse_console_logs() -> dict[str, tuple[str, float]]:
    """experiment_id -> (optimizer_name, effective_lr) from the console output."""
    found: dict[str, tuple[str, float]] = {}
    for name in CONSOLE_LOGS:
        path = paths.TRAINING_OUTPUT_DIR / name
        if not path.exists():
            continue
        text = ANSI.sub("", path.read_text(encoding="utf-8", errors="ignore"))
        current: str | None = None
        for line in text.splitlines():
            if m := RUN_START.search(line):
                current = m.group(1)
            elif (m := OPTIMIZER.search(line)) and current:
                found[current] = (m.group(1), float(m.group(2)))
                current = None          # take the first optimizer line per run
    return found


def main() -> int:
    log_path = paths.TRAINING_OUTPUT_DIR / "experiment_log.csv"
    if not log_path.exists():
        log.error("no experiment log at %s", log_path)
        return 1

    effective = parse_console_logs()
    if not effective:
        log.warning("no optimizer lines found in the console logs - nothing to do")
        return 0

    df = pd.read_csv(log_path)
    changed = 0
    for exp_id, (opt, lr) in effective.items():
        mask = df.experiment_id == exp_id
        if not mask.any():
            continue
        old_opt = df.loc[mask, "optimizer"].iloc[0]
        old_lr = df.loc[mask, "lr0"].iloc[0]
        if str(old_opt) != opt or not pd.isna(old_lr) and abs(float(old_lr) - lr) > 1e-9:
            log.info("%-20s requested(%s, lr0=%s) -> ACTUAL(%s, lr=%s)",
                     exp_id, old_opt, old_lr, opt, lr)
            df.loc[mask, "optimizer"] = opt
            df.loc[mask, "lr0"] = lr
            changed += 1

    if changed:
        df.to_csv(log_path, index=False)
        log.info("corrected %d rows in %s", changed, paths.rel_to_root(log_path))
    else:
        log.info("experiment log already records the effective optimizer/LR")
    print(df[["experiment_id", "optimizer", "lr0", "map50", "map50_95",
              "recall"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
