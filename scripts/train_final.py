"""Sprint 3: tuning probes (E4a-c) and the tuned final model (E5).

Usage:
    python scripts/train_final.py --probes            # run the three 20-epoch probes
    python scripts/train_final.py --final [k=v ...]   # run E5, optionally overriding
                                                      # config values from probe review
Example:
    python scripts/train_final.py --final optimizer=AdamW lr0=0.001 epochs=80

Probe review is a human step: results land in outputs/training/experiment_log.csv
and the chosen overrides are recorded in outputs/training/e5_decision.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import run_experiment
from src.utils import setup_logging

log = setup_logging("flameguard.train-final")

PROBES = ["e4a_probe_adamw", "e4b_probe_augment", "e4c_probe_loss"]


def _parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not _:
            raise ValueError(f"override '{pair}' is not key=value")
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", action="store_true", help="run E4a-c")
    parser.add_argument("--final", action="store_true", help="run E5")
    parser.add_argument("overrides", nargs="*", help="k=v overrides for E5")
    args = parser.parse_args()

    if not args.probes and not args.final:
        parser.error("pass --probes and/or --final")

    if args.probes:
        for exp_id in PROBES:
            run_experiment(exp_id)

    if args.final:
        overrides = _parse_overrides(args.overrides)
        if overrides:
            log.info("E5 overrides from probe review: %s", overrides)
        run_experiment("e5_final", overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
