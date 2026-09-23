import unittest
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gesture_filter.filters import FILTERS, apply_filter
from gesture_filter.roi import ROIController, State, global_box, rectangle
from gesture_filter.vision import Detector
from gesture_filter.app import arguments


def hand(x, y, gap=10):
    points = [SimpleNamespace(x=x/640, y=y/480) for _ in range(21)]
    points[0] = SimpleNamespace(x=x/640, y=(y+100)/480)
    points[9] = SimpleNamespace(x=x/640, y=y/480)
    points[4] = SimpleNamespace(x=(x-gap/2)/640, y=y/480)
    points[8] = SimpleNamespace(x=(x+gap/2)/640, y=y/480)
    return points


class GestureTests(unittest.TestCase):
    def test_pinch_debounce_release_and_lock(self):
        roi = ROIController()
        roi.update(hand(100, 100), 640, 480, 0)
        self.assertEqual(roi.state, State.IDLE)
        roi.update(hand(100, 100), 640, 480, .11)
        self.assertEqual(roi.state, State.DRAWING)
        roi.update(hand(400, 350), 640, 480, .2)
        corner = roi.cursor
        roi.update(hand(400, 350, 70), 640, 480, .3)
        roi.update(hand(400, 350, 70), 640, 480, .42)
        self.assertEqual(roi.state, State.LOCKED)
        self.assertEqual(roi.locked, (100, 100, *corner))
        locked = roi.locked
        roi.update(hand(500, 400), 640, 480, .6)
        self.assertEqual(roi.locked, locked)

    def test_hysteresis_ignores_middle_distance(self):
        roi = ROIController()
        roi.update(hand(100, 100), 640, 480, 0)
        roi.update(hand(100, 100), 640, 480, .11)
        for t in (.2, .4, .6):
            roi.update(hand(300, 300, 35), 640, 480, t)
        self.assertEqual(roi.state, State.DRAWING)

    def test_tracking_loss_cancels_not_locks(self):
        roi = ROIController()
        roi.update(hand(100, 100), 640, 480, 0)
        roi.update(hand(100, 100), 640, 480, .11)
        roi.update(None, 640, 480, .2)
        self.assertEqual(roi.state, State.DRAWING)
        roi.update(None, 640, 480, .6)
        self.assertEqual(roi.state, State.IDLE)
        self.assertIsNone(roi.locked)

    def test_small_selection_and_reset(self):
        roi = ROIController()
        roi.begin((10, 10))
        roi.finish((20, 20), 640, 480)
        self.assertEqual(roi.state, State.IDLE)
        roi.begin((300, 300))
        roi.finish((100, 100), 640, 480)
        self.assertEqual(roi.locked, (100, 100, 300, 300))
        roi.reset()
        self.assertIsNone(roi.preview(640, 480))


class GeometryAndFilterTests(unittest.TestCase):
    def test_cli_rejects_existing_output_and_headless_without_roi(self):
        for args in (["--output", __file__], ["--headless"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    arguments(args)
                self.assertEqual(error.exception.code, 2)

    def test_clipping_and_remapping(self):
        self.assertEqual(rectangle((700, 500), (-20, -10), 640, 480), (0, 0, 640, 480))
        self.assertEqual(global_box((10, 20, 80, 90), (100, 200, 300, 400)), (110, 220, 180, 290))
        self.assertEqual(global_box((-10, -20, 400, 400), (100, 200, 300, 400)), (100, 200, 300, 400))

    def test_all_filters_preserve_pixels_outside_box_and_input(self):
        frame = np.random.default_rng(1).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        original = frame.copy()
        outside = np.ones(frame.shape[:2], dtype=bool)
        outside[20:65, 15:70] = False
        for mode in FILTERS:
            with self.subTest(mode=mode):
                output = apply_filter(frame, [(15, 20, 70, 65)], mode)
                np.testing.assert_array_equal(output[outside], frame[outside])
                np.testing.assert_array_equal(frame, original)
                self.assertFalse(np.array_equal(output[~outside], frame[~outside]))

    def test_empty_and_tiny_regions(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        for mode in FILTERS:
            output = apply_filter(frame, [(4, 4, 4, 4), (0, 0, 1, 1)], mode)
            self.assertEqual(output.shape, frame.shape)
            np.testing.assert_array_equal(apply_filter(frame, [], mode), frame)

    def test_detector_receives_only_crop_and_returns_global_box(self):
        detector = Detector.__new__(Detector)
        detector.model = Mock()
        detector.model.names = {0: "person", 41: "cup"}
        detector.device, detector.confidence, detector.imgsz = "cpu", .35, 640
        detector.set_targets("person")
        detector.model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=Mock()))]
        detector.model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
            [10, 15, 60, 80, .9, 0]]
        frame = np.zeros((200, 300, 3), np.uint8)
        result = detector.detect(frame, (100, 50, 200, 150))
        self.assertEqual(detector.model.predict.call_args.args[0].shape, (100, 100, 3))
        self.assertEqual(detector.model.predict.call_args.kwargs["classes"], [0])
        self.assertEqual(result[0].box, (110, 65, 160, 130))
        with self.assertRaises(ValueError):
            detector.set_targets("unknown")


if __name__ == "__main__":
    unittest.main()
