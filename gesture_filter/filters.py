"""Apply effects only within target boxes without changing the input."""
import cv2
from .roi import rectangle

FILTERS = ("blur", "pixelate", "gray", "edge")


def apply_filter(frame, boxes, mode):
    if mode not in FILTERS:
        raise ValueError(f"Unknown filter: {mode}")
    output = frame.copy()
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = rectangle(box[:2], box[2:], width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = frame[y1:y2, x1:x2]
        if mode == "blur":
            processed = cv2.GaussianBlur(patch, (0, 0), sigmaX=15)
        elif mode == "pixelate":
            tiny = cv2.resize(patch, (max(1, (x2-x1)//18), max(1, (y2-y1)//18)), interpolation=cv2.INTER_AREA)
            processed = cv2.resize(tiny, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        elif mode == "gray":
            processed = cv2.cvtColor(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        else:
            processed = cv2.cvtColor(cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 80, 160), cv2.COLOR_GRAY2BGR)
        output[y1:y2, x1:x2] = processed
    return output
