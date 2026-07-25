"""Characterization tests for crop geometry and fixed/free crop state."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True

# The application deliberately preloads Torch before importing PyQt6.  Import it
# before asking the shared helper for a QApplication so this test preserves that
# production import order.
import gazou_kiritori as application

try:
    from tests.helpers import get_qapplication
except ModuleNotFoundError:
    from helpers import get_qapplication  # type: ignore[no-redef]

from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


class _MouseMoveEvent:
    """Minimal event surface used by CropLabel's ordinary drag path."""

    def __init__(self, x: float, y: float) -> None:
        self._position = QtCore.QPointF(x, y)

    def position(self) -> QtCore.QPointF:
        return QtCore.QPointF(self._position)

    def buttons(self) -> QtCore.Qt.MouseButton:
        return QtCore.Qt.MouseButton.LeftButton

    def modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return QtCore.Qt.KeyboardModifier.NoModifier


class _StopAfterCrop(RuntimeError):
    pass


class _CropRecordingImage:
    """Records the Pillow box chosen by save_cropped, then stops before I/O."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.recorded_box: tuple[int, int, int, int] | None = None

    def crop(self, box: tuple[int, int, int, int]) -> Any:
        self.recorded_box = box
        raise _StopAfterCrop("crop box recorded before save side effects")


class _CropLabelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qapplication = get_qapplication()

    def setUp(self) -> None:
        self.mainwin = QtWidgets.QWidget()
        self.mainwin.zoom_scale = 1.0
        self.mainwin.base_display_width = 10
        self.mainwin.base_display_height = 8
        self.mainwin.constrain_crop_to_image = False
        self.mainwin.multiple_lock_enabled = False
        self.mainwin._crop_rect = None
        self.mainwin._crop_rect_img = None
        self.mainwin._panel_hidden_by_user = False
        self.mainwin._adjust_mode = False
        self.mainwin._nudge_overlay = None
        self.mainwin._hide_action_panel = lambda: None
        self.mainwin.clip_notice_count = 0

        def record_clip_notice() -> None:
            self.mainwin.clip_notice_count += 1

        self.mainwin.show_crop_clip_notice = record_clip_notice
        self.image = Image.new("RGB", (10, 8), (20, 40, 60))
        self.mainwin.image = self.image
        self.label = application.CropLabel(self.mainwin)
        self._configure_view()

    def tearDown(self) -> None:
        self.label.close()
        self.mainwin.close()
        self.label.deleteLater()
        self.mainwin.deleteLater()
        self.image.close()
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.Type.DeferredDelete,
        )
        self.qapplication.processEvents()

    def _set_image_size(self, size: tuple[int, int]) -> None:
        self.image.close()
        self.image = Image.new("RGB", size, (20, 40, 60))
        self.mainwin.image = self.image

    def _configure_view(
        self,
        *,
        image_size: tuple[int, int] = (10, 8),
        base_size: tuple[int, int] | None = None,
        zoom: float = 1.0,
        pixmap_size: tuple[int, int] | None = None,
        label_size: tuple[int, int] | None = None,
        view_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._set_image_size(image_size)
        base_w, base_h = base_size or image_size
        self.mainwin.base_display_width = base_w
        self.mainwin.base_display_height = base_h
        self.mainwin.zoom_scale = zoom

        full_scaled_size = (
            max(1, int(round(base_w * zoom))),
            max(1, int(round(base_h * zoom))),
        )
        pixmap_w, pixmap_h = pixmap_size or full_scaled_size
        label_w, label_h = label_size or (pixmap_w, pixmap_h)
        self.label.resize(label_w, label_h)
        pixmap = QtGui.QPixmap(pixmap_w, pixmap_h)
        pixmap.fill(QtGui.QColor(20, 40, 60))
        self.label.setPixmap(pixmap)
        if view_rect is None:
            self.label._view_rect_scaled = QtCore.QRect(
                0,
                0,
                full_scaled_size[0],
                full_scaled_size[1],
            )
        else:
            self.label._view_rect_scaled = QtCore.QRect(*view_rect)

    def assertRect(
        self,
        rect: QtCore.QRect | None,
        expected: tuple[int, int, int, int],
    ) -> None:
        self.assertIsInstance(rect, QtCore.QRect)
        assert rect is not None
        self.assertEqual(rect.getRect(), expected)


class CropLabelGeometryCharacterizationTests(_CropLabelTestCase):
    def test_initial_state_has_no_selection_and_uses_free_crop_mode(self) -> None:
        self.assertFalse(self.label.fixed_crop_mode)
        self.assertIsNone(self.label.drag_rect_img)
        self.assertIsNone(self.label.fixed_crop_rect_img)
        self.assertIsNone(self.label.fixed_crop_size)
        self.assertFalse(self.label._aspect_lock)
        self.assertIsNone(self.label._aspect_ratio)
        self.assertFalse(self.label.adjust_mode)

    def test_constrained_drag_normalizes_all_four_directions(self) -> None:
        self._configure_view(image_size=(10, 10))
        expected = (2, 3, 5, 6)
        for start, end in (
            ((2, 3), (7, 9)),
            ((7, 9), (2, 3)),
            ((7, 3), (2, 9)),
            ((2, 9), (7, 3)),
        ):
            with self.subTest(start=start, end=end):
                rect = self.label._build_constrained_drag_rect(*start, *end)
                self.assertRect(rect, expected)

    def test_constrained_drag_rejects_zero_axes_and_accepts_one_pixel(self) -> None:
        self.assertIsNone(
            self.label._build_constrained_drag_rect(4, 5, 4, 7),
        )
        self.assertIsNone(
            self.label._build_constrained_drag_rect(4, 5, 6, 5),
        )
        self.assertRect(
            self.label._build_constrained_drag_rect(4, 5, 5, 6),
            (4, 5, 1, 1),
        )

    def test_constrained_drag_clamps_negative_and_overflow_coordinates(self) -> None:
        self.assertRect(
            self.label._build_constrained_drag_rect(-5, -6, 20, 30),
            (0, 0, 10, 8),
        )
        boundary = self.label._build_constrained_drag_rect(0, 0, 10, 8)
        self.assertRect(boundary, (0, 0, 10, 8))
        assert boundary is not None
        self.assertEqual((boundary.right(), boundary.bottom()), (9, 7))

    def test_unconstrained_drag_keeps_negative_and_overflow_coordinates(self) -> None:
        # Ordinary free-drag construction is inline in mouseMoveEvent, so this
        # minimal event exercises that production path without showing a window.
        self.mainwin.constrain_crop_to_image = False
        self.label._drag_start_img = (-2, -1)
        self.label.mouseMoveEvent(_MouseMoveEvent(12, 10))  # type: ignore[arg-type]

        self.assertRect(self.label.drag_rect_img, (-2, -1, 14, 11))

    def test_existing_rect_slides_before_it_is_clipped(self) -> None:
        moved, clipped = self.label._adjust_existing_rect_into_image(
            QtCore.QRect(-2, 1, 4, 3),
        )
        self.assertRect(moved, (0, 1, 4, 3))
        self.assertFalse(clipped)

        oversized, clipped = self.label._adjust_existing_rect_into_image(
            QtCore.QRect(-3, -2, 20, 15),
        )
        self.assertRect(oversized, (0, 0, 10, 8))
        self.assertTrue(clipped)

    def test_free_edge_resize_changes_width_and_height_independently(self) -> None:
        self.mainwin.constrain_crop_to_image = True
        width_change = self.label._clamp_edge_resize_rect(
            QtCore.QRect(QtCore.QPoint(2, 2), QtCore.QPoint(8, 4)),
            "r",
        )
        height_change = self.label._clamp_edge_resize_rect(
            QtCore.QRect(QtCore.QPoint(2, 2), QtCore.QPoint(5, 7)),
            "b",
        )

        self.assertRect(width_change, (2, 2, 7, 3))
        self.assertRect(height_change, (2, 2, 4, 6))

    def test_small_odd_even_landscape_and_portrait_image_bounds(self) -> None:
        for size in ((1, 1), (2, 2), (7, 3), (3, 7), (5, 7), (6, 8)):
            with self.subTest(size=size):
                self._configure_view(image_size=size)
                width, height = size
                bounds = self.label._crop_image_bounds()
                self.assertRect(bounds, (0, 0, width, height))
                assert bounds is not None
                self.assertEqual(
                    (bounds.right(), bounds.bottom()),
                    (width - 1, height - 1),
                )
                self.assertRect(
                    self.label._build_constrained_drag_rect(
                        0,
                        0,
                        width,
                        height,
                    ),
                    (0, 0, width, height),
                )

    def test_qrect_edges_become_half_open_pillow_crop_boxes(self) -> None:
        cases = (
            (QtCore.QRect(0, 0, 5, 4), (0, 0, 5, 4), (5, 4)),
            (QtCore.QRect(4, 3, 1, 1), (4, 3, 5, 4), (1, 1)),
            (QtCore.QRect(3, 1, 2, 3), (3, 1, 5, 4), (2, 3)),
            (QtCore.QRect(-2, -1, 5, 4), (0, 0, 3, 3), (3, 3)),
        )
        source = Image.new("RGB", (5, 4), (1, 2, 3))
        try:
            for rect, expected_box, expected_size in cases:
                with self.subTest(rect=rect.getRect()):
                    recorder = _CropRecordingImage(5, 4)
                    fake_app = SimpleNamespace(
                        image=recorder,
                        label=SimpleNamespace(fixed_crop_rect_img=rect),
                        _crop_rect_img=None,
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = application.CropperApp.save_cropped(fake_app, None)

                    self.assertFalse(result[0])
                    self.assertEqual(recorder.recorded_box, expected_box)
                    with source.crop(expected_box) as cropped:
                        self.assertEqual(cropped.size, expected_size)

            full = cases[0][0]
            self.assertEqual(
                (
                    full.left(),
                    full.top(),
                    full.right(),
                    full.bottom(),
                    full.width(),
                    full.height(),
                ),
                (0, 0, 4, 3, 5, 4),
            )
        finally:
            source.close()

    def test_identity_coordinate_conversion_includes_edges_and_outside(self) -> None:
        for image_point, label_point in (
            ((0, 0), (0, 0)),
            ((5, 4), (5, 4)),
            ((10, 8), (10, 8)),
            ((-1, -2), (-1, -2)),
            ((11, 9), (11, 9)),
        ):
            with self.subTest(image_point=image_point):
                self.assertEqual(
                    self.label.image_to_label_coords(*image_point),
                    label_point,
                )
                self.assertEqual(
                    self.label.label_to_image_coords(*label_point),
                    image_point,
                )

    def test_zoom_in_and_out_use_python_rounding(self) -> None:
        self._configure_view(
            image_size=(10, 10),
            base_size=(10, 10),
            zoom=2.0,
        )
        self.assertEqual(
            self.label.image_to_label_coords(1.25, 2.75),
            (2, 6),
        )
        self.assertEqual(self.label.label_to_image_coords(2, 6), (1, 3))

        self._configure_view(
            image_size=(10, 10),
            base_size=(10, 10),
            zoom=0.5,
        )
        self.assertEqual(self.label.image_to_label_coords(1, 1), (0, 0))
        self.assertEqual(self.label.label_to_image_coords(1, 1), (2, 2))
        self.assertEqual(self.label.image_to_label_coords(10, 10), (5, 5))

    def test_centering_offset_is_added_and_removed(self) -> None:
        self._configure_view(
            image_size=(10, 8),
            pixmap_size=(10, 8),
            label_size=(30, 20),
        )
        self.assertEqual(self.label.image_to_label_coords(0, 0), (10, 6))
        self.assertEqual(self.label.image_to_label_coords(5, 4), (15, 10))
        self.assertEqual(self.label.image_to_label_coords(10, 8), (20, 14))
        self.assertEqual(self.label.label_to_image_coords(9, 5), (-1, -1))

    def test_viewport_offset_records_zoom_and_pan_projection(self) -> None:
        # _view_rect_scaled is set directly because show_image also performs
        # file/UI work; this is the smallest state consumed by both converters.
        self._configure_view(
            image_size=(100, 80),
            base_size=(100, 80),
            zoom=2.0,
            pixmap_size=(80, 60),
            label_size=(100, 80),
            view_rect=(50, 40, 80, 60),
        )
        self.assertEqual(self.label.image_to_label_coords(25, 20), (10, 10))
        self.assertEqual(self.label.image_to_label_coords(45, 35), (50, 40))
        self.assertEqual(self.label.image_to_label_coords(65, 50), (90, 70))
        self.assertEqual(self.label.image_to_label_coords(0, 0), (-40, -30))
        self.assertEqual(self.label.label_to_image_coords(10, 10), (25, 20))
        self.assertEqual(self.label.label_to_image_coords(90, 70), (65, 50))

    def test_shrunk_view_round_trip_currently_loses_at_most_one_pixel(self) -> None:
        self._configure_view(
            image_size=(7, 5),
            base_size=(10, 6),
            zoom=0.5,
        )
        maximum_error = 0
        for x in range(8):
            for y in range(6):
                label_point = self.label.image_to_label_coords(x, y)
                round_trip = self.label.label_to_image_coords(*label_point)
                error = max(abs(round_trip[0] - x), abs(round_trip[1] - y))
                maximum_error = max(maximum_error, error)
                self.assertLessEqual(error, 1)
        self.assertEqual(maximum_error, 1)

    def test_inclusive_imgrect_helper_collapses_one_pixel_selection(self) -> None:
        one_pixel = QtCore.QRect(2, 3, 1, 1)
        converted = self.label._imgrect_to_labelrect(one_pixel)
        self.assertRect(converted, (2, 3, 0, 0))

        self.label.drag_rect_img = QtCore.QRect(one_pixel)
        exclusive_edge_conversion = self.label._drag_rect_labelcoords()
        self.assertRect(exclusive_edge_conversion, (2, 3, 1, 1))


class CropModeStateCharacterizationTests(_CropLabelTestCase):
    def test_fixed_crop_is_centered_for_small_odd_even_and_oriented_sizes(
        self,
    ) -> None:
        cases = (
            ((1, 1), (1, 1), (0, 0, 1, 1)),
            ((9, 7), (3, 5), (3, 1, 3, 5)),
            ((10, 8), (4, 2), (3, 3, 4, 2)),
            ((11, 6), (7, 2), (2, 2, 7, 2)),
            ((6, 11), (2, 7), (2, 2, 2, 7)),
        )
        for image_size, crop_size, expected in cases:
            with self.subTest(image_size=image_size, crop_size=crop_size):
                self._configure_view(image_size=image_size)
                self.label.start_fixed_crop(crop_size)
                self.assertTrue(self.label.fixed_crop_mode)
                self.assertEqual(self.label.fixed_crop_size, crop_size)
                self.assertRect(self.label.fixed_crop_rect_img, expected)

    def test_oversized_fixed_crop_remains_outside_when_unconstrained(self) -> None:
        self._configure_view(image_size=(5, 4))
        self.mainwin.constrain_crop_to_image = False
        self.label.start_fixed_crop((8, 7))

        self.assertRect(self.label.fixed_crop_rect_img, (-2, -2, 8, 7))
        assert self.label.fixed_crop_rect_img is not None
        self.assertEqual(
            (
                self.label.fixed_crop_rect_img.right(),
                self.label.fixed_crop_rect_img.bottom(),
            ),
            (5, 4),
        )
        self.assertEqual(self.mainwin.clip_notice_count, 0)

    def test_oversized_fixed_crop_is_full_image_when_constrained(self) -> None:
        self._configure_view(image_size=(5, 4))
        self.mainwin.constrain_crop_to_image = True
        self.label.start_fixed_crop((8, 7))

        self.assertRect(self.label.fixed_crop_rect_img, (0, 0, 5, 4))
        self.assertEqual(self.label.fixed_crop_size, (5, 4))
        self.assertEqual(self.mainwin.clip_notice_count, 1)

    def test_fixed_start_keeps_free_copy_but_fixed_rect_takes_precedence(
        self,
    ) -> None:
        free_rect = QtCore.QRect(1, 1, 4, 3)
        self.label.drag_rect_img = QtCore.QRect(free_rect)
        self.label.start_fixed_crop((2, 2))

        self.assertRect(self.label.drag_rect_img, free_rect.getRect())
        self.assertRect(self.label.fixed_crop_rect_img, (4, 3, 2, 2))
        self.assertRect(self.label._edit_rect_img(), (4, 3, 2, 2))

    def test_unfix_moves_current_fixed_rect_to_free_state(self) -> None:
        self.label.start_fixed_crop((4, 2))
        fixed_rect = QtCore.QRect(self.label.fixed_crop_rect_img)
        fake_app = SimpleNamespace(label=self.label, _crop_rect_img=None)

        application.CropperApp.unfix_fixed_mode(fake_app)

        self.assertFalse(self.label.fixed_crop_mode)
        self.assertIsNone(self.label.fixed_crop_rect_img)
        self.assertRect(self.label.drag_rect_img, fixed_rect.getRect())
        self.assertRect(fake_app._crop_rect_img, fixed_rect.getRect())
        self.assertEqual(self.label.fixed_crop_size, (4, 2))

    def test_aspect_lock_promotes_free_rect_and_off_keeps_fixed_mode(self) -> None:
        free_rect = QtCore.QRect(1, 2, 6, 3)
        self.label.drag_rect_img = QtCore.QRect(free_rect)
        fake_app = SimpleNamespace(label=self.label, _crop_rect_img=None)

        application.CropperApp.set_aspect_lock(fake_app, True)

        self.assertTrue(self.label._aspect_lock)
        self.assertEqual(self.label._aspect_base_wh, (6, 3))
        self.assertEqual(self.label._aspect_ratio, 2.0)
        self.assertTrue(self.label.fixed_crop_mode)
        self.assertRect(self.label.fixed_crop_rect_img, free_rect.getRect())
        self.assertIsNone(self.label.drag_rect_img)
        self.assertEqual(self.label.fixed_crop_size, (6, 3))
        self.assertRect(fake_app._crop_rect_img, free_rect.getRect())

        application.CropperApp.set_aspect_lock(fake_app, False)

        self.assertFalse(self.label._aspect_lock)
        self.assertIsNone(self.label._aspect_base_wh)
        self.assertIsNone(self.label._aspect_ratio)
        self.assertTrue(self.label.fixed_crop_mode)
        self.assertRect(self.label.fixed_crop_rect_img, free_rect.getRect())

    def test_finalize_synchronizes_croplabel_and_cropperapp_rect_copies(
        self,
    ) -> None:
        free_rect = QtCore.QRect(2, 1, 5, 4)
        self.label.drag_rect_img = QtCore.QRect(free_rect)
        self.label._finalize_adjust_interaction(QtCore.QPoint(7, 5))
        self.assertRect(self.mainwin._crop_rect_img, free_rect.getRect())

        fixed_rect = QtCore.QRect(3, 2, 4, 2)
        self.label.fixed_crop_mode = True
        self.label.fixed_crop_rect_img = QtCore.QRect(fixed_rect)
        self.mainwin._crop_rect_img = None
        self.label._finalize_adjust_interaction(QtCore.QPoint(7, 4))

        self.assertRect(self.mainwin._crop_rect_img, fixed_rect.getRect())
        self.assertRect(
            self.label.fixed_crop_rect_img_base,
            fixed_rect.getRect(),
        )
        self.assertEqual(self.label.fixed_crop_size, (4, 2))


def tearDownModule() -> None:
    """Remove bytecode produced before this test module disabled cache writes."""
    if not TEST_BYTECODE_CACHE.is_dir():
        return
    for bytecode_file in TEST_BYTECODE_CACHE.glob("*.py[co]"):
        bytecode_file.unlink()
    try:
        TEST_BYTECODE_CACHE.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()
