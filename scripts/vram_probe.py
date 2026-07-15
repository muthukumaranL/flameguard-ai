"""Measure what YOLOv8s actually costs on this GPU, and save it as evidence.

The YOLOv8s capacity experiment could not be completed within the project's GPU
budget. Rather than assert that from memory, this script MEASURES it: for each
batch size it starts a real training run, lets a handful of iterations execute,
records peak reserved VRAM and the wall-clock time per iteration, then aborts.

Output: outputs/training/vram_probe.json  (cited by the report; nothing about the
YOLOv8s limitation is claimed without a number from this file.)

Usage:
    python scripts/vram_probe.py [--model yolov8s.pt] [--batches 2 4 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import paths
from src.train import ultralytics_data_yaml
from src.utils import setup_logging

log = setup_logging("flameguard.vram")

WARMUP_ITERS = 3      # ignore allocator warm-up
MEASURE_ITERS = 8     # then time this many iterations


class _StopAfter(Exception):
    """Control-flow signal used to abort a probe run once measured."""


def probe(model_name: str, batch: int, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = {"count": 0, "t0": None, "elapsed": None}

    def on_batch_end(trainer) -> None:
        state["count"] += 1
        if state["count"] == WARMUP_ITERS:
            state["t0"] = time.perf_counter()
        elif state["count"] >= WARMUP_ITERS + MEASURE_ITERS:
            state["elapsed"] = time.perf_counter() - state["t0"]
            raise _StopAfter

    model = YOLO(model_name)
    model.add_callback("on_train_batch_end", on_batch_end)
    try:
        model.train(data=str(data_yaml), epochs=1, batch=batch, imgsz=640,
                    device="cuda:0", workers=2, val=False, plots=False, save=False,
                    project=str(paths.PROJECT_ROOT / ".tmp"), name=f"vram_b{batch}",
                    exist_ok=True, verbose=False)
    except _StopAfter:
        pass
    except torch.cuda.OutOfMemoryError as exc:
        log.warning("batch %d: CUDA OOM - %s", batch, exc)
        return {"batch": batch, "status": "cuda_out_of_memory",
                "peak_reserved_gb": None, "seconds_per_iteration": None}

    peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    s_per_it = state["elapsed"] / MEASURE_ITERS if state["elapsed"] else None
    # Windows/WDDM lets torch reserve beyond physical VRAM by paging into shared
    # system memory over PCIe. That is the failure we are documenting: it does not
    # crash, it just becomes an order of magnitude slower.
    spilling = peak_gb > total_gb * 0.98
    result = {
        "batch": batch,
        "status": "spilling_to_shared_memory" if spilling else "fits_in_vram",
        "peak_reserved_gb": round(peak_gb, 2),
        "gpu_total_gb": round(total_gb, 2),
        "seconds_per_iteration": round(s_per_it, 3) if s_per_it else None,
        "images_per_second": round(batch / s_per_it, 1) if s_per_it else None,
    }
    log.info("batch %-2d -> %.2f GB peak (%s), %.3f s/iter, %.1f img/s",
             batch, peak_gb, result["status"], s_per_it or 0,
             result["images_per_second"] or 0)
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--baseline-model", default="yolov8n.pt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log.error("no CUDA device - this probe measures GPU memory")
        return 1

    data_yaml = ultralytics_data_yaml(paths.PROCESSED_DATA_YAML)
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "imgsz": 640,
        "precision": "FP32 (AMP auto-disabled by Ultralytics on GTX 16xx)",
        "note": ("Peak reserved VRAM measured over real training iterations. On "
                 "Windows/WDDM, exceeding physical VRAM does not crash - PyTorch "
                 "pages into shared system memory over PCIe and throughput collapses. "
                 "That is the failure mode documented here."),
        "runs": [],
    }
    for b in args.batches:
        results["runs"].append({"model": args.model, **probe(args.model, b, data_yaml)})
    # one reference point from the model that DID train, for comparison
    results["runs"].append({"model": args.baseline_model,
                            **probe(args.baseline_model, 16, data_yaml)})

    out = paths.TRAINING_OUTPUT_DIR / "vram_probe.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("evidence -> %s", paths.rel_to_root(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
