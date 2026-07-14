"""FlameGuard AI - Streamlit application.

Run:
    streamlit run app.py

Tabs: Live Camera | Image Detection | Video Detection | Model Performance | About
The app loads the locally fine-tuned model from models/final/best.pt (cached)
and shares one inference engine across all detection modes.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_app_config, resolve_model_path
from src.image_inference import (InvalidImageError, annotated_png_bytes,
                                 detect_image, load_image_bytes,
                                 records_csv_bytes, records_json_bytes)
from src.inference import (DetectionEngine, MissingModelError, StatusSmoother,
                           detections_to_records)
from src.paths import EVALUATION_OUTPUT_DIR, OUTPUTS_DIR, TRAINING_OUTPUT_DIR
from src.utils import device_label

CFG = load_app_config()
ACCENT = "#e4572e"

# Temporary files stay inside the project (the system drive may be nearly full)
# and are cleaned up by the context manager that creates them.
TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="FlameGuard AI", page_icon="🔥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2.2rem; max-width: 1150px; }}
    h1 {{ font-size: 2.1rem; margin-bottom: 0; }}
    .fg-subtitle {{ color: #444; font-size: 1.15rem; margin-top: 0.15rem; }}
    .fg-caption {{ color: #8a8a8a; font-size: 0.85rem; margin-bottom: 0.8rem; }}
    .fg-status {{ padding: 0.55rem 1rem; border-radius: 6px; font-weight: 600;
                 display: inline-block; margin: 0.3rem 0; }}
    .fg-ok {{ background: #e7f5ec; color: #1b6e3c; border: 1px solid #bfe3cd; }}
    .fg-alert {{ background: #fdecea; color: #a4161a; border: 1px solid #f5c2c0; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; }}
    .stButton>button, .stDownloadButton>button {{
        border-radius: 6px; border: 1px solid #d0d0d0;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading detection model...")
def get_engine() -> DetectionEngine:
    return DetectionEngine(resolve_model_path())


def engine_or_none() -> DetectionEngine | None:
    try:
        return get_engine()
    except MissingModelError:
        return None


# ------------------------------------------------------------------- header
st.title("FlameGuard AI")
st.markdown('<div class="fg-subtitle">Real-Time Fire and Smoke Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="fg-caption">Deep-learning object detection using '
            'transfer learning.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
engine = engine_or_none()
defaults = CFG["defaults"]

with st.sidebar:
    st.subheader("Model")
    if engine is not None:
        st.success(f"Model loaded: `{resolve_model_path().name}`")
        st.caption(f"Device: **{device_label()}**")
    else:
        st.error("Model not found at `models/final/best.pt`.\n\n"
                 "Run the training pipeline first (see README).")

    st.subheader("Detection settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90,
                     float(defaults["confidence_threshold"]), 0.05)
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90,
                    float(defaults["iou_threshold"]), 0.05)
    show_labels = st.toggle("Show class labels", value=bool(defaults["show_labels"]))
    show_conf = st.toggle("Show confidence scores", value=bool(defaults["show_confidence"]))
    line_width = st.slider("Box line thickness", 1, 5, int(defaults["line_thickness"]))
    frame_skip = st.selectbox("Video frame skip",
                              options=[1, 2, 3],
                              format_func=lambda n: {1: "Process every frame",
                                                     2: "Every 2nd frame",
                                                     3: "Every 3rd frame"}[n],
                              index=int(defaults["video_frame_skip"]) - 1,
                              help="Skipping frames speeds up video processing "
                                   "but reduces temporal coverage.")
    st.divider()
    st.caption(CFG["disclaimer"])

tab_live, tab_image, tab_video, tab_perf, tab_about = st.tabs(
    ["Live Camera", "Image Detection", "Video Detection", "Model Performance", "About"])


# -------------------------------------------------------------- live camera
with tab_live:
    st.markdown("Start the camera to run continuous fire/smoke detection in "
                "your browser. Frames are processed locally on this machine.")
    if engine is None:
        st.warning("Live detection requires a trained model.")
    else:
        webrtc_available = True
        try:
            import av  # noqa: F401
            from streamlit_webrtc import WebRtcMode, webrtc_streamer
        except Exception:
            webrtc_available = False

        if webrtc_available:
            import av
            import threading

            class _LiveProcessor:
                """streamlit-webrtc frame callback holder with shared stats."""

                def __init__(self) -> None:
                    self.lock = threading.Lock()
                    self.conf = conf
                    self.iou = iou
                    self.show_labels = show_labels
                    self.show_conf = show_conf
                    self.line_width = line_width
                    self.smoother = StatusSmoother(
                        window=int(CFG["defaults"]["status_smoothing_frames"]),
                        min_hits=2)
                    self.stats = {"fire": 0, "smoke": 0, "total": 0,
                                  "fps": 0.0, "status": "No Hazard Detected"}
                    self._last = time.perf_counter()

                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    img = frame.to_ndarray(format="bgr24")
                    result = engine.predict(
                        img, conf=self.conf, iou=self.iou, draw=True,
                        show_labels=self.show_labels, show_conf=self.show_conf,
                        line_width=self.line_width,
                        max_size=int(CFG["defaults"]["webcam_process_width"]))
                    now = time.perf_counter()
                    inst_fps = 1.0 / max(now - self._last, 1e-6)
                    self._last = now
                    with self.lock:
                        s = self.stats
                        s["fire"], s["smoke"] = result.counts["fire"], result.counts["smoke"]
                        s["total"] = result.counts["total"]
                        s["fps"] = 0.9 * s["fps"] + 0.1 * inst_fps if s["fps"] else inst_fps
                        s["status"] = self.smoother.update(result)
                    return av.VideoFrame.from_ndarray(result.annotated_bgr, format="bgr24")

            ctx = webrtc_streamer(
                key="flameguard-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=_LiveProcessor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )
            status_box = st.empty()
            metric_box = st.empty()
            if ctx.state.playing and ctx.video_processor:
                ctx.video_processor.conf = conf
                ctx.video_processor.iou = iou
                ctx.video_processor.show_labels = show_labels
                ctx.video_processor.show_conf = show_conf
                ctx.video_processor.line_width = line_width
                while ctx.state.playing:
                    with ctx.video_processor.lock:
                        s = dict(ctx.video_processor.stats)
                    css = "fg-ok" if s["status"] == "No Hazard Detected" else "fg-alert"
                    status_box.markdown(
                        f'<div class="fg-status {css}">{s["status"]}</div>',
                        unsafe_allow_html=True)
                    with metric_box.container():
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Fire", s["fire"])
                        c2.metric("Smoke", s["smoke"])
                        c3.metric("Detections", s["total"])
                        c4.metric("FPS", f"{s['fps']:.1f}")
                        c5.metric("Device", device_label().split(" - ")[0])
                    time.sleep(0.4)
            else:
                st.caption("Camera is stopped. Click START and allow camera "
                           "access in your browser. If the camera does not "
                           "start, check that no other application is using it.")
        else:
            st.info("Browser live streaming (streamlit-webrtc) is unavailable in "
                    "this environment. Snapshot mode is active instead - or run "
                    "the desktop fallback: `python src/webcam_inference.py`.")
            snap = st.camera_input("Take a snapshot to analyse")
            if snap is not None:
                img = load_image_bytes(snap.getvalue())
                result = detect_image(engine, img, conf=conf, iou=iou,
                                      show_labels=show_labels, show_conf=show_conf,
                                      line_width=line_width)
                st.image(result.annotated_bgr[:, :, ::-1],
                         caption=result.status, width="stretch")


# ------------------------------------------------------------ image upload
with tab_image:
    upload = st.file_uploader("Upload an image (JPG, JPEG, PNG, WEBP)",
                              type=CFG["supported_image_types"])
    if upload is not None and engine is not None:
        size_mb = upload.size / (1024 * 1024)
        if size_mb > CFG["limits"]["max_image_mb"]:
            st.error(f"Image is {size_mb:.1f} MB - the limit is "
                     f"{CFG['limits']['max_image_mb']} MB.")
        else:
            try:
                image_bgr = load_image_bytes(upload.getvalue())
            except InvalidImageError as exc:
                st.error(f"Could not read this file as an image. {exc}")
            else:
                try:
                    result = detect_image(engine, image_bgr, conf=conf, iou=iou,
                                          show_labels=show_labels,
                                          show_conf=show_conf,
                                          line_width=line_width)
                except Exception as exc:  # inference failure surface, not crash
                    st.error(f"Prediction failed: {exc}")
                    result = None
                if result is not None:
                    col1, col2 = st.columns(2)
                    col1.image(image_bgr[:, :, ::-1], caption="Original",
                               width="stretch")
                    col2.image(result.annotated_bgr[:, :, ::-1],
                               caption="Detections", width="stretch")

                    if not result.detections:
                        st.info("No fire or smoke was detected above the "
                                "selected confidence threshold.")
                    else:
                        css = "fg-alert"
                        st.markdown(f'<div class="fg-status {css}">{result.status}</div>',
                                    unsafe_allow_html=True)
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Detections", result.counts["total"])
                    m2.metric("Fire", result.counts["fire"])
                    m3.metric("Smoke", result.counts["smoke"])
                    fire_conf = result.max_confidence(0)
                    smoke_conf = result.max_confidence(1)
                    m4.metric("Max fire conf", f"{fire_conf:.2f}" if fire_conf else "-")
                    m5.metric("Max smoke conf", f"{smoke_conf:.2f}" if smoke_conf else "-")
                    m6.metric("Inference", f"{result.inference_ms:.0f} ms")
                    st.caption(f"Image size: {result.image_width} x {result.image_height} px")

                    records = detections_to_records(result, upload.name)
                    d1, d2, d3 = st.columns(3)
                    d1.download_button("Download annotated PNG",
                                       annotated_png_bytes(result),
                                       file_name=f"{Path(upload.name).stem}_pred.png",
                                       mime="image/png")
                    d2.download_button("Download detections CSV",
                                       records_csv_bytes(records) or b"no detections\n",
                                       file_name=f"{Path(upload.name).stem}_detections.csv",
                                       mime="text/csv")
                    d3.download_button("Download detections JSON",
                                       records_json_bytes(records),
                                       file_name=f"{Path(upload.name).stem}_detections.json",
                                       mime="application/json")
    elif upload is not None:
        st.warning("Image detection requires a trained model.")


# ------------------------------------------------------------ video upload
with tab_video:
    vupload = st.file_uploader("Upload a video (MP4, AVI, MOV, MKV)",
                               type=CFG["supported_video_types"])
    if vupload is not None and engine is not None:
        vsize_mb = vupload.size / (1024 * 1024)
        if vsize_mb > CFG["limits"]["max_video_mb"]:
            st.error(f"Video is {vsize_mb:.0f} MB - the limit is "
                     f"{CFG['limits']['max_video_mb']} MB.")
        else:
            from src.video_inference import InvalidVideoError, probe_video, process_video

            # Identify this upload. A different file invalidates any previous
            # result, so stale statistics can never be shown next to a new video.
            upload_key = f"{vupload.name}:{vupload.size}"
            if st.session_state.get("video_key") != upload_key:
                for k in ("video_stats", "video_out", "video_name"):
                    st.session_state.pop(k, None)
                st.session_state["video_key"] = upload_key

            try:
                # Processing happens inside a context manager, so the temporary
                # input/output files are always removed - including on failure.
                # (Streamlit reruns this script on every widget interaction, so a
                # naive mkdtemp() here would leak a directory per interaction.)
                with tempfile.TemporaryDirectory(
                        prefix="flameguard_", dir=str(TMP_DIR)) as tmp:
                    tmp_dir = Path(tmp)
                    in_path = tmp_dir / vupload.name
                    in_path.write_bytes(vupload.getvalue())
                    out_path = tmp_dir / f"{in_path.stem}_pred.mp4"

                    meta = probe_video(in_path)
                    st.caption(f"`{vupload.name}` - {meta['width']}x{meta['height']} px, "
                               f"{meta['fps']:.1f} fps, {meta['frames']} frames")
                    if st.button("Run detection on this video", type="primary"):
                        progress = st.progress(0.0, text="Processing video...")
                        stats = process_video(
                            engine, in_path, out_path, conf=conf, iou=iou,
                            frame_skip=frame_skip, show_labels=show_labels,
                            show_conf=show_conf, line_width=line_width,
                            progress_cb=lambda f: progress.progress(
                                f, text=f"Processing video... {f:.0%}"))
                        progress.progress(1.0, text="Done")
                        # keep the result in memory; the files on disk go away
                        st.session_state["video_stats"] = stats
                        st.session_state["video_out"] = out_path.read_bytes()
                        st.session_state["video_name"] = out_path.name
            except InvalidVideoError as exc:
                st.error(f"This video cannot be processed: {exc}")
            except Exception as exc:            # surface, never crash the app
                st.error(f"Video processing failed: {exc}")

            stats = st.session_state.get("video_stats")
            if stats is not None and st.session_state.get("video_out"):
                st.video(st.session_state["video_out"])
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Frames processed",
                          f"{stats.frames_processed}/{stats.total_frames}")
                s2.metric("Processing FPS", f"{stats.processing_fps:.1f}")
                s3.metric("Frames w/ fire", stats.frames_with_fire)
                s4.metric("Frames w/ smoke", stats.frames_with_smoke)
                s5, s6, s7, s8 = st.columns(4)
                s5.metric("Max fire conf",
                          f"{stats.max_fire_confidence:.2f}" if stats.max_fire_confidence else "-")
                s6.metric("Max smoke conf",
                          f"{stats.max_smoke_confidence:.2f}" if stats.max_smoke_confidence else "-")
                s7.metric("Total detections", stats.total_detections)
                s8.metric("Processing time", f"{stats.processing_seconds:.1f}s")
                if stats.total_detections == 0:
                    st.info("No fire or smoke was detected above the selected "
                            "confidence threshold in the processed frames.")
                v1, v2 = st.columns(2)
                v1.download_button("Download annotated video",
                                   st.session_state["video_out"],
                                   file_name=st.session_state["video_name"],
                                   mime="video/mp4")
                frame_csv = pd.DataFrame(stats.records)
                v2.download_button("Download frame detections CSV",
                                   frame_csv.to_csv(index=False).encode("utf-8")
                                   if not frame_csv.empty else b"no detections\n",
                                   file_name=f"{Path(stats.source_name).stem}_frames.csv",
                                   mime="text/csv")
    elif vupload is not None:
        st.warning("Video detection requires a trained model.")


# -------------------------------------------------------- model performance
with tab_perf:
    st.markdown("All values below are read from files produced by the actual "
                "training and evaluation runs.")
    metrics_file = EVALUATION_OUTPUT_DIR / "metrics_test.json"
    if not metrics_file.exists():
        st.info("Result file not available. Run the evaluation pipeline first.")
    else:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        st.subheader("Final model - held-out test set")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Precision", f"{data['precision']:.3f}")
        p2.metric("Recall", f"{data['recall']:.3f}")
        p3.metric("F1", f"{data['f1']:.3f}")
        p4.metric("mAP@0.5", f"{data['map50']:.3f}")
        p5.metric("mAP@0.5:0.95", f"{data['map50_95']:.3f}")
        per_class = data.get("per_class", {})
        if per_class:
            st.dataframe(pd.DataFrame(per_class).T.round(3),
                         width="stretch")
        p6, p7, p8 = st.columns(3)
        p6.metric("Model size", f"{data['model_size_mb']:.1f} MB")
        p7.metric("Inference / image", f"{data['total_ms_per_image']:.1f} ms")
        fps_val = data.get("fps_estimate")
        p8.metric("Estimated FPS", f"{fps_val:.1f}" if fps_val else "-")

        figures = [
            ("Confusion matrix", EVALUATION_OUTPUT_DIR / "test_confusion_matrix.png"),
            ("Normalized confusion matrix",
             EVALUATION_OUTPUT_DIR / "test_confusion_matrix_normalized.png"),
            ("Precision-Recall curve", EVALUATION_OUTPUT_DIR / "test_BoxPR_curve.png"),
            ("F1-confidence curve", EVALUATION_OUTPUT_DIR / "test_BoxF1_curve.png"),
            ("Threshold analysis", EVALUATION_OUTPUT_DIR / "threshold_analysis.png"),
            ("Model comparison", OUTPUTS_DIR / "benchmarking" / "benchmark_chart.png"),
            ("Training curves (final model)",
             TRAINING_OUTPUT_DIR / "e5_final" / "results.png"),
        ]
        cols = st.columns(2)
        shown = 0
        for title, fig_path in figures:
            if fig_path.exists():
                cols[shown % 2].image(str(fig_path), caption=title,
                                      width="stretch")
                shown += 1
        if shown == 0:
            st.info("Evaluation figures not found. Run the evaluation pipeline first.")


# -------------------------------------------------------------------- about
with tab_about:
    st.subheader("About FlameGuard AI")
    st.markdown(f"""
**Problem.** Early visual detection of fire and smoke can shorten emergency
response times. This project trains a custom object detector that finds
*Fire* and *Smoke* regions in images, videos and live camera streams.

**Dataset.** Roboflow Universe - *fire and smoke* (v1) by `fire-detector-cqdzi`,
licensed **CC BY 4.0**
([dataset link](https://universe.roboflow.com/fire-detector-cqdzi/fire-and-smoke-b5lli/dataset/1)).
The original release contained duplicated and mirrored copies of source images
across the train/validation/test folders; the splits were rebuilt group-wise in
this project so that evaluation images are never seen during training.

**Model.** Ultralytics YOLO (single-stage detector). COCO-pretrained weights
are **fine-tuned by transfer learning** on the fire/smoke dataset - the
pretrained backbone provides generic visual features, while training on the
custom data adapts the detection head to the two target classes.

**Classes.** `0 - Fire`, `1 - Smoke`.

**Limitations.** The model can confuse sunsets, orange lighting and reflections
with fire, and fog, steam or clouds with smoke. Thin or distant smoke and small
flames may be missed, particularly in low light. Detection quality depends on
camera quality and scene conditions.

**Privacy.** All processing runs locally on this machine. No image, video or
camera frame leaves the computer, and no identity recognition is performed.

**Team.** Group `[GROUP NUMBER]` - `[Project Manager]`, `[Dataset & EDA Lead]`,
`[Model Training Lead]`, `[Application Lead]`, `[Evaluation & Documentation Lead]`.

**Course.** AASD 4014 - Deep Learning II, Final Project.
""")
    st.warning(CFG["disclaimer"])
