# CrowdSense AI — Advanced Event Crowd Detector

A local Streamlit dashboard for educational crowd monitoring using YOLO person detection, configurable crowd thresholds, capacity usage, spatial heatmaps, video sampling, session history, and CSV exports.

## Requirements
- Windows 10/11
- Python 3.12 recommended
- Internet for the first YOLO model-weight download (`yolov8n.pt`)
- VS Code (recommended)

## Run on Windows with VS Code
1. Extract the ZIP to a normal folder (do not open the ZIP as a text file).
2. In VS Code, choose **File → Open Folder** and select `CrowdSense_AI_Advanced`.
3. Choose **Terminal → New Terminal** and run these commands one at a time:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run this in the same terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If the `py -3.12` command is not recognised, install Python 3.12 and ensure the Python Launcher is selected during installation.

## Features
- YOLO pretrained person detection (COCO class 0)
- Image analysis with annotated image and illustrative spatial heatmap
- Video analysis by configurable frame sampling
- Low / Moderate / High / Critical crowd classifications
- Venue capacity utilisation and configurable thresholds
- Mean detection confidence metric
- Session analysis history, trend chart, and CSV exports
- Downloadable annotated image and heatmap
- Friendly error messages for invalid images, model loading, and video decoding

## Notes and limitations
- The first run may download `yolov8n.pt`; keep the internet connection active.
- Video processing time depends on video duration, sampling interval, model resolution, and computer performance.
- Detection counts are estimates and can be wrong due to occlusion, lighting, camera angle, or dense crowds.
- The heatmap is based on detected-person box centers and is not a calibrated density map.
- History is stored in Streamlit session state, not a permanent database.
- This educational prototype is not a certified occupancy or public-safety system. A human must verify alerts.
