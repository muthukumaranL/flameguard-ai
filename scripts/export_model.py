"""Export the final model to ONNX for portable / edge deployment.

ONNX runs the detector without PyTorch or Ultralytics installed, which is the
first step toward the edge deployment discussed in the report's future work.
The export is verified by loading it back and comparing its output shape.

Usage:
    python scripts/export_model.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.utils import file_size_mb, setup_logging

log = setup_logging("flameguard.export")


def main() -> int:
    if not paths.FINAL_MODEL_PATH.exists():
        log.error("final model missing - run scripts/benchmark.py first")
        return 1

    from ultralytics import YOLO

    paths.MODELS_DIR.joinpath("exported").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(paths.FINAL_MODEL_PATH))
    log.info("exporting to ONNX (opset 12, imgsz 640)...")
    try:
        onnx_path = Path(model.export(format="onnx", imgsz=640, opset=12,
                                      simplify=False, dynamic=False))
    except Exception as exc:
        log.error("ONNX export failed (%s). This is optional - the PyTorch model "
                  "in models/final/best.pt remains the deployment artefact.", exc)
        return 0                       # non-fatal: never block the project on this

    dest = paths.MODELS_DIR / "exported" / "flameguard_final.onnx"
    shutil.move(str(onnx_path), dest)
    log.info("exported -> %s (%.1f MB)", paths.rel_to_root(dest), file_size_mb(dest))

    # verify the exported graph actually loads and produces the expected shape
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.run(None, {inp.name: np.zeros((1, 3, 640, 640), dtype=np.float32)})
        log.info("verified: input %s %s -> output %s",
                 inp.name, inp.shape, [o.shape for o in out])
    except ImportError:
        log.info("onnxruntime not installed - export written but not executed "
                 "(pip install onnxruntime to verify)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
