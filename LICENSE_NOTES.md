# License and Attribution Notes

## Dataset

- **Name:** fire and smoke - v1 (Roboflow Universe)
- **Author/workspace:** `fire-detector-cqdzi`, project `fire-and-smoke-b5lli`
- **Version:** 1 (exported 2023-08-23)
- **License:** **CC BY 4.0** (per the export's `data.yaml`)
- **URL:** <https://universe.roboflow.com/fire-detector-cqdzi/fire-and-smoke-b5lli/dataset/1>
- The original ZIP and extracted export are preserved unmodified outside this
  repository; `data/raw/` holds an exact copy (see `data/raw/fire_and_smoke_v1/PROVENANCE.json`).
- This project rebuilt the train/valid/test split (leakage repair) but did not
  alter any image or annotation content.

## Models

- **Ultralytics YOLOv8 / YOLO11** - AGPL-3.0 licensed open-source software.
  Used here for a non-commercial academic course project. COCO-pretrained
  weights are downloaded automatically by the `ultralytics` package and
  fine-tuned locally (transfer learning).

## Third-party packages

Installed from PyPI under their respective licenses (see `requirements.txt`);
notably PyTorch (BSD-3), OpenCV (Apache-2.0), Streamlit (Apache-2.0),
streamlit-webrtc (MIT), imageio-ffmpeg (BSD-2 + LGPL ffmpeg binary).

## Scope

This is an educational prototype for AASD 4014 (Deep Learning II). It is not a
certified fire-safety product and must not be used as one.
