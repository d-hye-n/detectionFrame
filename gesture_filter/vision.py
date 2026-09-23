"""Model adapters for MediaPipe Tasks and Ultralytics YOLO."""
from dataclasses import dataclass
from pathlib import Path
import os
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / ".cache" / "models"
HAND_MODEL = ASSETS / "hand_landmarker.task"
YOLO_MODEL = ASSETS / "yolo11n.pt"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def configure_cache():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "Ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    for key in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def download_models():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for path, url in ((HAND_MODEL, HAND_URL), (YOLO_MODEL, YOLO_URL)):
        if path.exists():
            print(f"Already available: {path.name}")
            continue
        print(f"Downloading {path.name} ...", flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    label: str
    confidence: float


class Detector:
    def __init__(self, model, device, targets, confidence, imgsz):
        configure_cache()
        import torch
        from ultralytics import YOLO
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = YOLO(str(model))
        self.confidence, self.imgsz = confidence, imgsz
        self.set_targets(targets)

    def set_targets(self, targets):
        requested = {item.strip() for item in targets.split(",") if item.strip()}
        if targets == "all":
            self.classes = None
        else:
            unknown = requested - set(self.model.names.values())
            if unknown or not requested:
                raise ValueError(f"Unknown/empty target classes: {sorted(unknown)}. Use COCO names or 'all'.")
            self.classes = [key for key, label in self.model.names.items() if label in requested]
        self.targets = targets

    def detect(self, frame, roi):
        from .roi import global_box
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            return []
        result = self.model.predict(crop, device=self.device, imgsz=self.imgsz,
                                    conf=self.confidence, classes=self.classes, verbose=False)[0]
        detections = []
        for left, top, right, bottom, confidence, class_id in result.boxes.data.cpu().tolist():
            box = global_box((left, top, right, bottom), roi)
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(box, self.model.names[int(class_id)], confidence))
        return detections


class HandTracker:
    def __init__(self, model):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp, self.last_timestamp = mp, -1
        self.tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=1))

    def detect(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1_000_000)
        self.last_timestamp = timestamp
        result = self.tracker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self):
        self.tracker.close()
