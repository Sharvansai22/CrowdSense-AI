git add app.py detector.py requirements.txt README.md .gitignorefrom __future__ import annotations

import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import analyze_frame, classify_crowd, load_model

st.set_page_config(page_title="CrowdSense AI | Command Center", page_icon="👥", layout="wide", initial_sidebar_state="expanded")

# ---------- visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d18; --panel:#101827; --panel2:#131f31; --line:#26364c; --muted:#9aabc2; --text:#edf4ff; --cyan:#53e0d0; --blue:#6e9cff; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #142744 0%, #080d18 48%, #080d18 100%); color:var(--text); font-family:'Manrope',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,13,24,.75); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1500px; }
section[data-testid="stSidebar"] { background:#0b1220; border-right:1px solid #233149; }
section[data-testid="stSidebar"] * { color:#e9f2ff; }
.hero { border:1px solid #2a4562; border-radius:24px; padding:26px 30px; background:linear-gradient(115deg,rgba(25,55,88,.95),rgba(14,26,45,.94) 55%,rgba(19,48,66,.95)); box-shadow:0 18px 60px rgba(0,0,0,.18); position:relative; overflow:hidden; }
.hero:after { content:'◉'; position:absolute; right:32px; top:12px; font-size:92px; color:rgba(83,224,208,.12); }
.eyebrow { color:var(--cyan); font-family:'DM Mono',monospace; letter-spacing:2px; font-size:11px; text-transform:uppercase; }
.hero h1 { color:#f5f9ff; font-size:clamp(30px,4vw,44px); line-height:1.1; letter-spacing:-1.7px; margin:8px 0 10px; font-weight:800; }
.hero p { color:#b9cce3; font-size:14px; margin:0; max-width:720px; }
.pill { display:inline-block; border:1px solid #31526d; color:#bdece8; background:#102a3a; padding:5px 10px; border-radius:999px; font:11px 'DM Mono',monospace; margin:16px 6px 0 0; }
.section-label { color:#8da6c5; font:11px 'DM Mono',monospace; letter-spacing:1.8px; text-transform:uppercase; margin:20px 0 10px; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#142238,#101827); border:1px solid #2a3b52; padding:17px 19px; border-radius:16px; box-shadow:0 8px 24px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color:#9db2ce !important; font-size:12px; }
[data-testid="stMetricValue"] { color:#f2f7ff !important; font-weight:800; }
[data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; }
div[data-testid="stTabs"] button { color:#b9cce3; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#53e0d0; }
.stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #35516e; font-weight:700; }
.stButton button[kind="primary"] { background:linear-gradient(90deg,#237e8b,#2764a5); border:0; }
div[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #2a3b52; border-radius:12px; overflow:hidden; }
.small-muted { color:#91a5bf; font-size:12px; }
.status-box { border:1px solid #2a3b52; border-radius:14px; padding:15px; background:#101827; }
footer { visibility:hidden; }
</style>
<div class="hero">
  <div class="eyebrow">Computer vision · Live intelligence · Crowd safety support</div>
  <h1>CrowdSense <span style="color:#53e0d0">AI</span></h1>
  <p>Event crowd monitoring and visual analytics, powered by YOLO person detection. Upload a scene, inspect the detections, and understand how crowd levels compare with your configured venue limits.</p>
  <span class="pill">YOLO PERSON DETECTION</span><span class="pill">SPATIAL HEATMAP</span><span class="pill">CAPACITY ALERTS</span><span class="pill">LOCAL PROCESSING</span>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "model_error" not in st.session_state:
    st.session_state.model_error = None

with st.sidebar:
    st.markdown("### ⚙️ Control room")
    st.caption("Configure the venue and detector before analysing media.")
    venue_name = st.text_input("Venue / event name", "Campus Event")
    capacity = st.number_input("Venue capacity", min_value=1, max_value=100000, value=100, step=10)
    st.markdown("---")
    st.markdown("**Crowd classification thresholds**")
    low_limit = st.number_input("Low: maximum count", min_value=0, max_value=100000, value=20, step=5)
    moderate_limit = st.number_input("Moderate: starts at", min_value=1, max_value=100000, value=21, step=5)
    high_limit = st.number_input("High: starts at", min_value=1, max_value=100000, value=75, step=5)
    confidence = st.slider("YOLO confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    image_size = st.select_slider("Inference resolution", options=[416, 512, 640, 768, 960], value=640)
    st.markdown("---")
    show_heatmap = st.toggle("Show spatial heatmap", value=True)
    st.caption("Model weights: yolov8n.pt. First run requires internet to download weights.")
    if st.button("🧹 Clear session history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Ensure thresholds remain logically ordered, without crashing if user changes them.
moderate_limit = max(int(moderate_limit), int(low_limit) + 1)
high_limit = max(int(high_limit), int(moderate_limit) + 1)

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("yolov8n.pt")

def add_history(source: str, count: int, status: str, ratio: float, avg_conf: float, frames: int = 1):
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue": venue_name,
        "source": source,
        "people_detected": int(count),
        "crowd_status": status,
        "capacity_usage_percent": round(float(ratio), 1),
        "mean_detection_confidence": round(float(avg_conf), 3),
        "frames_analysed": int(frames),
    })

def status_message(status: str, count: int, ratio: float):
    if status == "CRITICAL":
        st.error(f"🚨 CRITICAL CROWD LEVEL — {count} people detected; estimated capacity usage is {ratio:.1f}%. Ask event staff to verify immediately.")
    elif status == "HIGH":
        st.warning(f"⚠️ HIGH CROWD LEVEL — {count} people detected. Check crowd flow and venue conditions.")
    elif status == "MODERATE":
        st.info(f"Crowd level is MODERATE — {count} people detected. Continue monitoring.")
    else:
        st.success(f"Crowd level is LOW — {count} people detected in the analysed view.")

def metrics_row(count: int, ratio: float, status: str, avg_conf: float):
    a, b, c, d = st.columns(4)
    a.metric("People detected", f"{count:,}")
    b.metric("Capacity used", f"{ratio:.1f}%")
    c.metric("Crowd status", status)
    d.metric("Mean confidence", f"{avg_conf:.0%}")

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Dashboard summary
hist_df = pd.DataFrame(st.session_state.history)
st.markdown('<div class="section-label">Operational overview</div>', unsafe_allow_html=True)
if not hist_df.empty:
    total_analyses = len(hist_df)
    latest = hist_df.iloc[-1]
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Analyses this session", total_analyses)
    top2.metric("Latest detected count", int(latest["people_detected"]))
    top3.metric("Peak session count", int(hist_df["people_detected"].max()))
    top4.metric("High / critical alerts", int(hist_df["crowd_status"].isin(["HIGH", "CRITICAL"]).sum()))
else:
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Analyses this session", "0")
    top2.metric("Latest detected count", "—")
    top3.metric("Peak session count", "—")
    top4.metric("High / critical alerts", "0")

image_tab, video_tab, history_tab, about_tab = st.tabs(["🖼️ Image analysis", "🎞️ Video analysis", "📈 History & reports", "🧠 How it works"])

with image_tab:
    st.markdown('<div class="section-label">Single-scene analysis</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a crowd or event image", type=["jpg", "jpeg", "png", "webp"], key="image_upload", help="Supported formats: JPG, JPEG, PNG, WEBP")
    if upload is not None:
        raw = np.frombuffer(upload.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("This image could not be decoded. Try a different JPG or PNG file.")
        else:
            col_a, col_b = st.columns([1, 1])
            with st.spinner("Loading YOLO and analysing the image…"):
                try:
                    model = get_model()
                    result = analyze_frame(frame, model, confidence, image_size)
                except Exception as exc:
                    st.error(f"AI analysis failed: {exc}")
                    st.info("Check your internet connection for the first model download, then retry. Confirm all packages installed successfully.")
                    result = None
            if result is not None:
                status, emoji, color = classify_crowd(result.count, capacity, low_limit, moderate_limit, high_limit)
                ratio = result.count / max(int(capacity), 1) * 100
                metrics_row(result.count, ratio, status, result.confidence_mean)
                status_message(status, result.count, ratio)
                with col_a:
                    st.markdown("**AI annotated image**")
                    st.image(bgr_to_rgb(result.annotated_bgr), use_container_width=True)
                    ok, encoded = cv2.imencode(".jpg", result.annotated_bgr)
                    if ok:
                        st.download_button("⬇ Download annotated image", encoded.tobytes(), "crowdsense_annotated.jpg", "image/jpeg", use_container_width=True)
                with col_b:
                    st.markdown("**Spatial detection heatmap**" if show_heatmap else "**Original image**")
                    visual = result.heatmap_bgr if show_heatmap else frame
                    st.image(bgr_to_rgb(visual), use_container_width=True)
                    if show_heatmap:
                        ok, heat_encoded = cv2.imencode(".jpg", result.heatmap_bgr)
                        if ok:
                            st.download_button("⬇ Download heatmap", heat_encoded.tobytes(), "crowdsense_heatmap.jpg", "image/jpeg", use_container_width=True)
                add_history(f"Image: {upload.name}", result.count, status, ratio, result.confidence_mean)
                with st.expander("Detection details"):
                    if result.boxes:
                        details = pd.DataFrame([{"Person": i + 1, "Confidence": round(b["confidence"], 3), "Center X": int((b["x1"] + b["x2"]) / 2), "Center Y": int((b["y1"] + b["y2"]) / 2)} for i, b in enumerate(result.boxes)])
                        st.dataframe(details, use_container_width=True, hide_index=True)
                    else:
                        st.write("No person detections above the selected confidence threshold.")

with video_tab:
    st.markdown('<div class="section-label">Video timeline analysis</div>', unsafe_allow_html=True)
    video_upload = st.file_uploader("Upload an event video", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
    sample_every = st.slider("Analyse one frame every N frames", min_value=1, max_value=60, value=10, help="Higher values are faster but provide a less detailed timeline.")
    max_frames = st.number_input("Maximum frames to analyse (0 = no limit)", min_value=0, max_value=100000, value=600, step=100)
    if video_upload is not None and st.button("▶ Analyse video", type="primary", use_container_width=True):
        temp_path = None
        try:
            suffix = Path(video_upload.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_upload.getvalue())
                temp_path = tmp.name
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("OpenCV could not open this video. Try MP4 (H.264) or a smaller video file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                st.caption(f"Video: {video_upload.name} · {total_frames:,} frames · approximately {duration:.1f} seconds · sampling every {sample_every} frame(s)")
                try:
                    model = get_model()
                except Exception as exc:
                    st.error(f"Could not load YOLO model: {exc}")
                    model = None
                if model is not None:
                    progress = st.progress(0)
                    preview = st.empty()
                    live_status = st.empty()
                    records = []
                    frame_index = 0
                    processed = 0
                    started = time.time()
                    latest_result = None
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if frame_index % int(sample_every) == 0:
                            # Scale large frames to make inference more practical.
                            h, w = frame.shape[:2]
                            if max(h, w) > 1280:
                                scale = 1280 / max(h, w)
                                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                            try:
                                result = analyze_frame(frame, model, confidence, image_size)
                            except Exception as exc:
                                st.error(f"Frame {frame_index} failed: {exc}")
                                break
                            status, _, _ = classify_crowd(result.count, capacity, low_limit, moderate_limit, high_limit)
                            ratio = result.count / max(int(capacity), 1) * 100
                            timestamp = frame_index / fps if fps > 0 else 0
                            records.append({"time_seconds": round(timestamp, 2), "people_detected": result.count, "capacity_usage_percent": round(ratio, 1), "crowd_status": status, "mean_confidence": round(result.confidence_mean, 3)})
                            latest_result = result
                            preview.image(bgr_to_rgb(result.annotated_bgr), caption=f"Frame {frame_index} · {result.count} people · {status}", use_container_width=True)
                            live_status.info(f"Processed sample {len(records)} · source frame {frame_index}/{total_frames or '?'} · elapsed {time.time() - started:.1f}s")
                            processed += 1
                            if max_frames and processed >= int(max_frames):
                                break
                        frame_index += 1
                        if total_frames > 0:
                            progress.progress(min(frame_index / total_frames, 1.0))
                    cap.release()
                    if records:
                        video_df = pd.DataFrame(records)
                        peak = int(video_df["people_detected"].max())
                        average = float(video_df["people_detected"].mean())
                        last = int(video_df.iloc[-1]["people_detected"])
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Peak detected", peak)
                        m2.metric("Average detected", f"{average:.1f}")
                        m3.metric("Last sample", last)
                        m4.metric("Samples analysed", len(video_df))
                        st.line_chart(video_df.set_index("time_seconds")["people_detected"], height=260)
                        st.dataframe(video_df, use_container_width=True, hide_index=True)
                        st.download_button("⬇ Download video timeline CSV", video_df.to_csv(index=False).encode("utf-8"), "crowdsense_video_timeline.csv", "text/csv", use_container_width=True)
                        maxrow = video_df.loc[video_df["people_detected"].idxmax()]
                        add_history(f"Video: {video_upload.name}", peak, str(maxrow["crowd_status"]), peak / max(int(capacity), 1) * 100, float(video_df["mean_confidence"].mean()), len(video_df))
                        st.success(f"Video analysis complete: {len(video_df)} sampled frames in {time.time() - started:.1f} seconds.")
                    else:
                        st.warning("No frames were successfully analysed. Try another video or lower the inference resolution.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            st.info("Use a short MP4 video first. Large videos may need more memory or processing time.")
        finally:
            if 'cap' in locals():
                cap.release()
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

with history_tab:
    st.markdown('<div class="section-label">Session records & export</div>', unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.history)
    if hist_df.empty:
        st.info("No analyses recorded yet. Analyse an image or video to populate this report.")
    else:
        st.line_chart(hist_df.reset_index().set_index("index")["people_detected"], height=240)
        st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("⬇ Export complete history (CSV)", hist_df.to_csv(index=False).encode("utf-8"), "crowdsense_session_history.csv", "text/csv", use_container_width=True)
        st.caption("History is held in this Streamlit session and may reset when the app restarts or the session is cleared.")

with about_tab:
    st.markdown('<div class="section-label">System overview</div>', unsafe_allow_html=True)
    st.markdown("""
    ### From pixels to crowd-level guidance
    1. **Input capture:** the user uploads an image or video.
    2. **Person detection:** YOLO predicts bounding boxes; only the COCO `person` class is retained.
    3. **Crowd count:** the number of retained boxes is used as the detected-person estimate for that image or sampled video frame.
    4. **Capacity comparison:** detected count is compared with the venue capacity and configurable count thresholds.
    5. **Spatial visualization:** detected-person box centers contribute to an illustrative heatmap.
    6. **Reporting:** counts, timestamps, confidence estimates, and status labels can be exported as CSV.

    **Important limitations**
    - People may be missed or counted incorrectly due to occlusion, lighting, distance, camera angle, and crowded scenes.
    - The heatmap represents detected locations, not calibrated physical density per square metre.
    - A single frame is not a reliable estimate of total venue occupancy when the camera sees only part of a venue.
    - Alerts are advisory and require human verification; this prototype is not a certified safety system.
    - Uploaded media is processed by this local Python app, but the pretrained model may be downloaded from the internet on first use.
    """)

st.markdown("---")
st.markdown('<div class="small-muted">CROWDSENSE AI · BTech CSE–AIML prototype · YOLO person detections are estimates, not certified occupancy measurements.</div>', unsafe_allow_html=True)
