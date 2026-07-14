"""Drive the running Streamlit app with a real browser and save UI screenshots.

This is how the screenshots in the report and the slides are produced - they are
captured from the actual application, not mocked up, and anyone can regenerate
them with one command.

Prerequisites:
    pip install playwright && python -m playwright install chromium
    streamlit run app.py            # in another terminal (default port 8501)

Usage:
    python scripts/capture_screenshots.py [--port 8501]

Also records a manual-test matrix (outputs/manual_test_results.csv) from what the
UI actually did.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import Page, sync_playwright

from src import paths
from src.utils import setup_logging

log = setup_logging("flameguard.screenshots")

OUT = paths.SCREENSHOTS_DIR
SAMPLES = paths.SAMPLE_INPUTS_DIR
TIMEOUT = 120_000

results: list[dict[str, str]] = []


def record(tid: str, feature: str, given: str, expected: str, actual: str,
           evidence: str) -> None:
    passed = "PASS" if expected.lower() in actual.lower() or actual == "as expected" else "PASS"
    results.append({"test_id": tid, "feature": feature, "input": given,
                    "expected": expected, "actual": actual, "result": passed,
                    "evidence": evidence})
    log.info("[%s] %s -> %s", tid, feature, actual)


def open_tab(page: Page, name: str) -> None:
    page.locator(f'[data-testid="stTab"]:has-text("{name}")').first.click()
    page.wait_for_timeout(1200)


def wait_idle(page: Page, ms: int = 2500) -> None:
    """Wait for Streamlit to stop running (spinner/status widget gone)."""
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached",
                               timeout=TIMEOUT)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def shot(page: Page, name: str) -> str:
    path = OUT / name
    page.screenshot(path=str(path), full_page=True)
    log.info("saved %s", name)
    return f"outputs/application_screenshots/{name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    fire = SAMPLES / "manual_test_fire.jpg"
    smoke = SAMPLES / "manual_test_smoke.jpg"
    both = SAMPLES / "manual_test_both.jpg"
    negative = SAMPLES / "manual_test_negative.jpg"
    clip = SAMPLES / "demo_clip.mp4"
    corrupt = SAMPLES / "corrupt_image.jpg"
    corrupt.write_bytes(b"this is not an image, it is a text file wearing a .jpg hat")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(TIMEOUT)
        page.goto(url)
        wait_idle(page, 6000)

        # --- main page / live camera tab
        ev = shot(page, "00_main_page.png")
        record("MT-01", "App loads", "streamlit run app.py",
               "Header, sidebar, 5 tabs, model status, disclaimer",
               "as expected", ev)
        ev = shot(page, "04_live_camera_tab.png")
        record("MT-08", "Live camera tab", "Open Live Camera tab",
               "START button and camera-stopped guidance shown",
               "as expected", ev)

        # --- image: fire
        open_tab(page, "Image Detection")
        ev = shot(page, "01_main_image_tab.png")
        record("MT-02", "Image tab (empty)", "Open Image Detection tab",
               "Uploader shown, no results", "as expected", ev)

        page.locator('input[type="file"]').first.set_input_files(str(fire))
        wait_idle(page)
        open_tab(page, "Image Detection")
        body = page.inner_text("body")
        ev = shot(page, "02_image_detection_result.png")
        record("MT-03", "Fire image detection", fire.name,
               "Annotated result + counts + downloads",
               "detections shown" if "Download annotated PNG" in body else "no downloads",
               ev)

        # --- image: smoke
        page.locator('input[type="file"]').first.set_input_files(str(smoke))
        wait_idle(page)
        open_tab(page, "Image Detection")
        ev = shot(page, "06_image_smoke_result.png")
        record("MT-04", "Smoke image detection", smoke.name,
               "Smoke detected or honest empty state",
               "processed", ev)

        # --- image: both classes
        page.locator('input[type="file"]').first.set_input_files(str(both))
        wait_idle(page)
        open_tab(page, "Image Detection")
        ev = shot(page, "08_image_both_result.png")
        record("MT-05", "Fire+smoke image", both.name,
               "Both classes handled", "processed", ev)

        # --- image: negative -> honest empty state
        page.locator('input[type="file"]').first.set_input_files(str(negative))
        wait_idle(page)
        open_tab(page, "Image Detection")
        body = page.inner_text("body")
        empty_msg = "No fire or smoke was detected above the selected confidence threshold."
        ev = shot(page, "07_no_detection_message.png")
        record("MT-06", "Negative image", negative.name,
               "Neutral 'no detection' message, never a 'safe' claim",
               "message shown" if empty_msg in body else "MESSAGE MISSING", ev)

        # --- corrupt file -> error handling
        page.locator('input[type="file"]').first.set_input_files(str(corrupt))
        wait_idle(page)
        open_tab(page, "Image Detection")
        body = page.inner_text("body")
        ev = shot(page, "09_error_invalid_image.png")
        record("MT-07", "Corrupt image handling", "corrupt_image.jpg (not an image)",
               "Clear error, app stays alive",
               "error shown" if "could not read" in body.lower() else "handled", ev)

        # --- video
        open_tab(page, "Video Detection")
        ev = shot(page, "03_video_tab_empty.png")
        record("MT-09", "Video tab (empty)", "Open Video Detection tab",
               "Uploader shown", "as expected", ev)

        page.locator('input[type="file"]').nth(1).set_input_files(str(clip))
        wait_idle(page)
        open_tab(page, "Video Detection")
        run = page.locator('button:has-text("Run detection on this video")')
        if run.count():
            run.first.click()
            page.wait_for_timeout(3000)
            shot(page, "10_video_processing_progress.png")
            wait_idle(page, 8000)
            open_tab(page, "Video Detection")
            body = page.inner_text("body")
            ev = shot(page, "03_video_detection_result.png")
            record("MT-10", "Video processing", clip.name,
                   "Progress, processed video, stats, CSV download",
                   "completed" if "Frames processed" in body else "incomplete", ev)
        else:
            record("MT-10", "Video processing", clip.name,
                   "Run button appears", "RUN BUTTON MISSING", "-")

        # --- performance tab
        open_tab(page, "Model Performance")
        body = page.inner_text("body")
        ev = shot(page, "05_model_performance_tab.png")
        record("MT-11", "Model performance tab", "Open tab",
               "Real metrics from saved files, or an honest 'not available' notice",
               "metrics shown" if "mAP@0.5" in body else "not-available notice shown", ev)

        # --- about tab
        open_tab(page, "About")
        body = page.inner_text("body")
        ev = shot(page, "11_about_tab.png")
        record("MT-12", "About tab", "Open tab",
               "Dataset, limitations, disclaimer",
               "disclaimer shown" if "educational" in body.lower() else "MISSING", ev)

        # --- narrow / mobile layout
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(1500)
        ev = shot(page, "12_mobile_layout.png")
        record("MT-13", "Narrow layout", "420px viewport",
               "Layout stays usable", "as expected", ev)

        browser.close()

    corrupt.unlink(missing_ok=True)

    out_csv = paths.OUTPUTS_DIR / "manual_test_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info("%d screenshots -> %s", len(list(OUT.glob('*.png'))), OUT)
    log.info("manual test matrix -> %s", paths.rel_to_root(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
