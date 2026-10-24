"""Core computer-vision functions for CrowdSense AI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionResult:
    annotated_bgr: np.ndarray
    heatmap_bgr: np.ndarray
    count: int
    confidence_mean: float
    boxes: list[dict[str, float]]


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Load a pretrained Ultralytics YOLO model (downloads weights on first run)."""
    return YOLO(model_name)


def classify_crowd(count: int, capacity: int, low_limit: int, moderate_limit: int, high_limit: int) -> tuple[str, str, str]:
    """Return (label, emoji, color) using thresholds and configured venue capacity."""
    capacity = max(int(capacity), 1)
    ratio = count / capacity
    if ratio >= 1:
        return "CRITICAL", "🔴", "#ff4d5e"
    if count >= high_limit:
        return "HIGH", "🟠", "#ff9f43"
    if count >= moderate_limit or count > low_limit:
        return "MODERATE", "🟡", "#ffd166"
    return "LOW", "🟢", "#39d98a"


def make_heatmap(frame: np.ndarray, boxes: list[dict[str, float]]) -> np.ndarray:
    """Create an illustrative spatial heatmap from detected person-box centers."""
    h, w = frame.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    sigma = max(12, min(h, w) // 24)
    for box in boxes:
        cx = int((box["x1"] + box["x2"]) / 2)
        cy = int((box["y1"] + box["y2"]) / 2)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(density, (cx, cy), sigma, 1.0, -1)
    if density.max() > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
        density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(density, cv2.COLORMAP_TURBO)
        mask = density > 8
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame, 0.35, colored, 0.65, 0)[mask]
        return overlay
    return frame.copy()


def analyze_frame(frame: np.ndarray, model: YOLO, confidence: float = 0.35, image_size: int = 640) -> DetectionResult:
    """Detect only the COCO person class and return annotation and heatmap."""
    if frame is None or frame.size == 0:
        raise ValueError("The supplied frame is empty or could not be decoded.")
    result: Any = model.predict(source=frame, conf=float(confidence), classes=[0], imgsz=int(image_size), verbose=False)[0]
    annotated = result.plot()
    boxes: list[dict[str, float]] = []
    scores: list[float] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for coords, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [float(v) for v in coords]
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(score)})
            scores.append(float(score))
    heatmap = make_heatmap(frame, boxes)
    return DetectionResult(annotated, heatmap, len(boxes), float(np.mean(scores)) if scores else 0.0, boxes)
