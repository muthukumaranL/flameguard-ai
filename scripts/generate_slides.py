"""Generate the presentation deck, speaker notes and demo script.

Usage:
    python scripts/generate_slides.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PRESENTATION_DIR
from src.report_generator import Artifacts
from src.slide_generator import build_deck, write_demo_script, write_speaker_notes
from src.utils import setup_logging

log = setup_logging("flameguard.slides-runner")


def main() -> int:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        art = Artifacts.load()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run the full pipeline before generating slides.")
        return 1

    prs, notes = build_deck(art)
    deck = PRESENTATION_DIR / "FlameGuard_AI_Presentation.pptx"
    prs.save(deck)
    log.info("deck -> %s (%d slides)", deck, len(prs.slides.__iter__.__self__._sldIdLst))

    write_speaker_notes(notes, PRESENTATION_DIR / "speaker_notes.md")
    write_demo_script(art, PRESENTATION_DIR / "demo_script.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
