"""Build the final academic report (Markdown + DOCX + PDF) from saved artefacts.

Usage:
    python scripts/generate_report.py

Fails loudly if a required artefact is missing - the report never invents a
number, so an incomplete pipeline must stop here rather than emit placeholders.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import REPORT_DIR, REPORT_FIGURES_DIR
from src.report_generator import Artifacts, ReportBuilder
from src.report_renderers import render_docx, render_markdown, render_pdf
from src.utils import setup_logging

log = setup_logging("flameguard.report-runner")


def main() -> int:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = Artifacts.load()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run the full pipeline before generating the report "
                  "(see README: Reproducing every result).")
        return 1

    builder = ReportBuilder(artifacts)
    blocks = builder.build()
    log.info("report model: %d blocks, %d figures, %d tables",
             len(blocks), builder.fig_no, builder.tab_no)

    render_markdown(blocks, REPORT_DIR / "report_content.md")
    render_docx(blocks, REPORT_DIR / "FlameGuard_AI_Final_Report.docx")
    render_pdf(blocks, REPORT_DIR / "FlameGuard_AI_Final_Report.pdf")

    refs = next((payload for kind, payload in blocks
                 if kind == "bullets" and any("Redmon" in item for item in payload)),
                [])
    if refs:
        (REPORT_DIR / "references.md").write_text(
            "\n".join(["# References", ""] +
                      [f"{i + 1}. {ref}" for i, ref in enumerate(refs)]) + "\n",
            encoding="utf-8")
    else:
        log.warning("reference list not found in the report blocks")

    log.info("Report generated in %s (%d figures, %d tables)",
             REPORT_DIR, builder.fig_no, builder.tab_no)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
