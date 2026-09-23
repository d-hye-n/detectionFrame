"""Opt-in actual model test, using the image bundled with Ultralytics, no camera."""
from pathlib import Path
import time

import cv2
import numpy as np

from gesture_filter.app import main
from gesture_filter.filters import apply_filter
from gesture_filter.vision import Detector, HandTracker, HAND_MODEL, YOLO_MODEL, ROOT, configure_cache


def run():
    configure_cache()
    import ultralytics
    sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    frame = cv2.imread(str(sample))
    assert frame is not None, f"Missing package sample: {sample}"
    height, width = frame.shape[:2]
    roi = (20, 200, width-20, height-20)
    detector = Detector(YOLO_MODEL, "auto", "person", .35, 640)
    detections = detector.detect(frame, roi)
    assert detections, "Expected people in bundled bus sample"
    for d in detections:
        assert d.label == "person"
        assert roi[0] <= d.box[0] < d.box[2] <= roi[2]
        assert roi[1] <= d.box[1] < d.box[3] <= roi[3]
    output = apply_filter(frame, [d.box for d in detections], "pixelate")
    outside = np.ones(frame.shape[:2], dtype=bool)
    for d in detections:
        x1, y1, x2, y2 = d.box
        outside[y1:y2, x1:x2] = False
    np.testing.assert_array_equal(output[outside], frame[outside])
    assert not np.array_equal(output, frame)
    artifacts = ROOT / ".cache" / "smoke"
    artifacts.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(artifacts / "filtered.jpg"), output)
    hands = HandTracker(HAND_MODEL)
    try:
        hands.detect(frame)
        hands.detect(frame)  # Verify strictly increasing VIDEO timestamps.
    finally:
        hands.close()
    started = time.perf_counter()
    for _ in range(5):
        detector.detect(frame, roi)
    print(f"GPU/device {detector.device}: {len(detections)} people; warm mean {(time.perf_counter()-started)*200:.1f} ms")
    video = artifacts / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(frame)
    finally:
        writer.release()
    saved = artifacts / f"output-{time.time_ns()}.mp4"
    status = main(["--source", str(video), "--headless", "--no-hands", "--roi",
                   *map(str, roi), "--max-frames", "3", "--output", str(saved)])
    assert status == 0, "Headless application failed"
    capture = cv2.VideoCapture(str(saved))
    count = 0
    try:
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    assert count == 3, f"Expected 3 saved frames, got {count}"
    print(f"PASS: real YOLO, MediaPipe, filters, headless video pipeline. Artifacts: {artifacts}")


if __name__ == "__main__":
    run()
