"""Local webcam application with keyboard, gesture and mouse controls."""
import argparse
from pathlib import Path
import sys
import time

from .roi import ROIController, State, rectangle
from .vision import Detector, HandTracker, HAND_MODEL, YOLO_MODEL, download_models

WINDOW = "Gesture-Guided Object Filter"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or local video path")
    parser.add_argument("--model", type=Path, default=YOLO_MODEL)
    parser.add_argument("--hand-model", type=Path, default=HAND_MODEL)
    parser.add_argument("--download-models", action="store_true", help="Download official weights and exit")
    parser.add_argument("--device", default="auto", help="auto, cpu, or GPU index (0)")
    parser.add_argument("--classes", default="person", help="Comma-separated COCO classes, or all")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--filter", choices=("blur", "pixelate", "gray", "edge"), default="blur")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-hands", action="store_true", help="Mouse/keyboard ROI only")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam frames")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--headless", action="store_true", help="Process without a display window")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--output", type=Path, help="Optional processed .mp4 video")
    args = parser.parse_args(argv)
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")
    if min(args.width, args.height, args.imgsz) <= 0 or args.max_frames < 0:
        parser.error("Dimensions must be positive and --max-frames must be nonnegative")
    if args.headless and not args.roi and not args.download_models:
        parser.error("--headless requires --roi X1 Y1 X2 Y2")
    if args.output and not args.download_models:
        if args.output.exists():
            parser.error("--output already exists; choose a new path to preserve the existing file")
        if not args.source.isdecimal() and args.output.resolve() == Path(args.source).resolve():
            parser.error("--output must not be the input video")
    return args


def overlay(frame, controller, detections, landmarks, fps, inference_ms, mode, target, device):
    import cv2
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), (60, 200, 255), 2)
        cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}", (x1, max(18, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 200, 255), 1, cv2.LINE_AA)
    roi = controller.preview(width, height)
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), (180, 230, 60), 2)
    if landmarks:
        for index in (4, 8):
            p = landmarks[index]
            cv2.circle(frame, (int(p.x*width), int(p.y*height)), 7, (180, 230, 60), -1)
    lines = [
        f"{controller.state.value} | {mode} | {target} | targets {len(detections)}",
        f"FPS {fps:.1f} | detect {inference_ms:.1f} ms | device {device}",
        "Pinch + drag, release to lock | Mouse drag also works",
        "R reset  F full frame  1-4 filters  C class  H HUD  Q quit",
    ]
    cv2.rectangle(frame, (0, 0), (width, 105), (25, 28, 32), -1)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (12, 22+24*index), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (225, 230, 235), 1, cv2.LINE_AA)


def run(args):
    import cv2
    from .filters import FILTERS, apply_filter
    for path in [args.model] + ([] if args.no_hands else [args.hand_model]):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model: {path}. Run: python main.py --download-models")
    detector = Detector(args.model, args.device, args.classes, args.confidence, args.imgsz)
    hands = capture = writer = None
    controller = ROIController()
    mouse = {"active": False, "owns": False}
    dimensions = [args.width, args.height]
    mode, show_hud, frames, fps = args.filter, True, 0, 0.0
    camera = args.source.isdecimal()
    source = int(args.source) if camera else args.source
    cycle = list(dict.fromkeys([args.classes, "person", "cup", "bottle", "all"]))

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            controller.reset()
            controller.begin((x, y))
            mouse.update(active=True, owns=True)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["active"]:
            controller.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and mouse["active"]:
            controller.finish((x, y), *dimensions)
            mouse.update(active=False, owns=controller.state == State.LOCKED)

    try:
        if not args.no_hands:
            hands = HandTracker(args.hand_model)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source {source!r}. Check camera permissions or --source 1.")
        if camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, on_mouse)
        print(f"Ready: device={detector.device}, classes={detector.targets}, filter={mode}", flush=True)
        session_start = time.perf_counter()
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if camera or frames == 0:
                    raise RuntimeError("No frame received from source.")
                break
            if camera and not args.no_mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            dimensions[:] = [width, height]
            if frames == 0 and args.roi:
                box = rectangle(args.roi[:2], args.roi[2:], width, height)
                controller.begin(box[:2])
                controller.finish(box[2:], width, height)
                if controller.locked is None:
                    raise ValueError("--roi must cover at least 40 x 40 pixels inside the actual frame")
            landmarks = None
            if hands and controller.state != State.LOCKED and not mouse["owns"]:
                landmarks = hands.detect(frame)
                controller.update(landmarks, width, height, time.monotonic())
            detect_start = time.perf_counter()
            detections = detector.detect(frame, controller.locked) if controller.locked else []
            inference_ms = (time.perf_counter()-detect_start)*1000 if controller.locked else 0.0
            output = apply_filter(frame, [d.box for d in detections], mode)
            if show_hud:
                overlay(output, controller, detections, landmarks, fps, inference_ms, mode,
                        detector.targets, detector.device)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps if 1 <= source_fps <= 240 else 30, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot write video: {args.output}")
                writer.write(output)
            key = -1
            if not args.headless:
                cv2.imshow(WINDOW, output)
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frames += 1
            elapsed = max(time.perf_counter()-started, 1e-6)
            fps = 1/elapsed if not fps else .9*fps+.1/elapsed
            if key in (27, ord("q")) or (args.max_frames and frames >= args.max_frames):
                break
            if key == ord("r"):
                controller.reset()
                mouse.update(active=False, owns=False)
            elif key == ord("f"):
                controller.reset()
                controller.begin((0, 0))
                controller.finish((width, height), width, height)
                mouse.update(active=False, owns=False)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                mode = FILTERS[key-ord("1")]
            elif key == ord("c"):
                detector.set_targets(cycle[(cycle.index(detector.targets)+1) % len(cycle)])
            elif key == ord("h"):
                show_hud = not show_hud
        print(f"Processed {frames} frames in {time.perf_counter()-session_start:.2f}s")
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if hands is not None:
            hands.close()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    args = arguments(argv)
    try:
        if args.download_models:
            download_models()
        else:
            run(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
