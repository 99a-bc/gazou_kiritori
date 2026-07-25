"""Characterization tests for thumbnail-row navigation and path selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True

# The application deliberately preloads Torch before importing PyQt6.
import gazou_kiritori as application

try:
    from tests.helpers import get_qapplication
except ModuleNotFoundError:
    from helpers import get_qapplication  # type: ignore[no-redef]

from PyQt6 import QtCore, QtWidgets


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


class _RecordingPool:
    """QThreadPool stand-in that records work without starting worker threads."""

    def __init__(self) -> None:
        self.tasks: list[Any] = []
        self.clear_count = 0

    def start(self, task: Any) -> None:
        self.tasks.append(task)

    def clear(self) -> None:
        self.clear_count += 1
        self.tasks.clear()


class _CurrentIndex:
    def __init__(self, row: int = -1, *, valid: bool = False) -> None:
        self._row = row
        self._valid = valid

    def isValid(self) -> bool:
        return self._valid

    def row(self) -> int:
        return self._row


class _RecordingListView:
    """Small QListView surface used by CropperApp._move_thumb_focus."""

    def __init__(self, current: _CurrentIndex | QtCore.QModelIndex) -> None:
        self.current = current
        self.selected_rows: list[int] = []
        self.scrolled_rows: list[int] = []
        self.focus_reasons: list[QtCore.Qt.FocusReason] = []

    def currentIndex(self) -> _CurrentIndex | QtCore.QModelIndex:
        return self.current

    def setCurrentIndex(self, index: QtCore.QModelIndex) -> None:
        self.current = index
        self.selected_rows.append(index.row())

    def scrollTo(
        self,
        index: QtCore.QModelIndex,
        _hint: QtWidgets.QAbstractItemView.ScrollHint,
    ) -> None:
        self.scrolled_rows.append(index.row())

    def setFocus(self, reason: QtCore.Qt.FocusReason) -> None:
        self.focus_reasons.append(reason)


class _KeyEvent:
    """Minimal key event accepted by CustomListView's handled branches."""

    def __init__(
        self,
        key: QtCore.Qt.Key,
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> None:
        self._key = key
        self._modifiers = modifiers
        self.accepted = False
        self.ignored = False

    def key(self) -> QtCore.Qt.Key:
        return self._key

    def modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class _NavigationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qapplication = get_qapplication()

    def setUp(self) -> None:
        self._models: list[application.ThumbnailListModel] = []
        self._widgets: list[QtWidgets.QWidget] = []

    def tearDown(self) -> None:
        for widget in self._widgets:
            widget.close()
            widget.deleteLater()
        for model in self._models:
            model.deleteLater()
        self.qapplication.processEvents()
        self.assertFalse(
            any(widget.isVisible() for widget in self.qapplication.topLevelWidgets())
        )

    def make_model(
        self,
        paths: list[str],
    ) -> tuple[application.ThumbnailListModel, _RecordingPool]:
        model = application.ThumbnailListModel(paths)
        pool = _RecordingPool()
        model._pool = pool
        self._models.append(model)
        return model, pool

    def make_navigator(
        self,
        paths: list[str],
        *,
        current_row: int = -1,
        current_valid: bool = False,
        loop: bool = False,
    ) -> tuple[SimpleNamespace, _RecordingListView]:
        model, _pool = self.make_model(paths)
        listview = _RecordingListView(
            _CurrentIndex(current_row, valid=current_valid)
        )
        previewed_rows: list[int] = []
        subject = SimpleNamespace(
            model=model,
            listview=listview,
            _thumb_loop_enabled=loop,
            _preview_from_thumb_index=lambda index: previewed_rows.append(
                index.row()
            ),
            previewed_rows=previewed_rows,
        )
        return subject, listview


class ThumbnailRowNavigationTests(_NavigationTestCase):
    def test_empty_list_does_not_select_scroll_focus_or_preview(self) -> None:
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                subject, listview = self.make_navigator([], loop=False)

                application.CropperApp._move_thumb_focus(subject, delta)

                self.assertEqual(listview.selected_rows, [])
                self.assertEqual(listview.scrolled_rows, [])
                self.assertEqual(listview.focus_reasons, [])
                self.assertEqual(subject.previewed_rows, [])

    def test_non_looping_navigation_stops_at_ends_and_moves_in_middle(
        self,
    ) -> None:
        paths = ["first.png", "middle.png", "last.png"]
        cases = (
            (0, -1, None),
            (0, 1, 1),
            (1, -1, 0),
            (1, 1, 2),
            (2, -1, 1),
            (2, 1, None),
        )

        for current_row, delta, expected_row in cases:
            with self.subTest(
                current_row=current_row,
                delta=delta,
                expected_row=expected_row,
            ):
                subject, listview = self.make_navigator(
                    paths,
                    current_row=current_row,
                    current_valid=True,
                    loop=False,
                )

                application.CropperApp._move_thumb_focus(subject, delta)

                expected = [] if expected_row is None else [expected_row]
                self.assertEqual(listview.selected_rows, expected)
                self.assertEqual(listview.scrolled_rows, expected)
                self.assertEqual(subject.previewed_rows, expected)
                self.assertEqual(
                    len(listview.focus_reasons),
                    0 if expected_row is None else 1,
                )

    def test_looping_navigation_wraps_at_both_ends(self) -> None:
        paths = ["first.png", "middle.png", "last.png"]
        cases = ((0, -1, 2), (2, 1, 0))

        for current_row, delta, expected_row in cases:
            with self.subTest(current_row=current_row, delta=delta):
                subject, listview = self.make_navigator(
                    paths,
                    current_row=current_row,
                    current_valid=True,
                    loop=True,
                )

                application.CropperApp._move_thumb_focus(subject, delta)

                self.assertEqual(listview.selected_rows, [expected_row])
                self.assertEqual(subject.previewed_rows, [expected_row])

    def test_one_item_reselection_depends_on_loop_setting(self) -> None:
        for loop, expected in ((False, []), (True, [0])):
            for delta in (-1, 1):
                with self.subTest(loop=loop, delta=delta):
                    subject, listview = self.make_navigator(
                        ["only.png"],
                        current_row=0,
                        current_valid=True,
                        loop=loop,
                    )

                    application.CropperApp._move_thumb_focus(subject, delta)

                    self.assertEqual(listview.selected_rows, expected)
                    self.assertEqual(subject.previewed_rows, expected)

    def test_currently_unselected_list_uses_row_zero_as_navigation_origin(
        self,
    ) -> None:
        # Current behavior is asymmetric: "next" selects row 1, while
        # "previous" clamps to row 0 and returns without selecting it.
        paths = ["first.png", "middle.png", "last.png"]

        next_subject, next_view = self.make_navigator(paths)
        application.CropperApp._move_thumb_focus(next_subject, 1)
        self.assertEqual(next_view.selected_rows, [1])
        self.assertEqual(next_subject.previewed_rows, [1])

        previous_subject, previous_view = self.make_navigator(paths)
        application.CropperApp._move_thumb_focus(previous_subject, -1)
        self.assertEqual(previous_view.selected_rows, [])
        self.assertEqual(previous_subject.previewed_rows, [])

    def test_out_of_range_current_row_is_clamped_or_ignored_before_preview(
        self,
    ) -> None:
        paths = ["first.png", "middle.png", "last.png"]
        cases = ((99, 1, 2), (-5, -1, None))

        for current_row, delta, expected_row in cases:
            with self.subTest(current_row=current_row, delta=delta):
                subject, listview = self.make_navigator(
                    paths,
                    current_row=current_row,
                    current_valid=True,
                )

                application.CropperApp._move_thumb_focus(subject, delta)

                expected = [] if expected_row is None else [expected_row]
                self.assertEqual(listview.selected_rows, expected)
                self.assertEqual(subject.previewed_rows, expected)


class ImageIndexSynchronizationTests(_NavigationTestCase):
    def make_preview_subject(
        self,
        *,
        image_path: str,
        image_list: list[str],
        current_index: int | None = -1,
    ) -> tuple[SimpleNamespace, dict[str, list[Any]]]:
        calls: dict[str, list[Any]] = {
            "opened": [],
            "preserved": [],
            "synced": [],
        }
        attributes: dict[str, Any] = {
            "image_path": image_path,
            "image_list": image_list,
            "_norm_path": application.norm_vpath,
            "_prepare_preserve_for_nav": lambda: calls["preserved"].append(True),
            "open_image_from_path": lambda path: calls["opened"].append(path),
            "_sync_thumb_selection": lambda: calls["synced"].append(True),
        }
        if current_index is not None:
            attributes["current_index"] = current_index
        return SimpleNamespace(**attributes), calls

    def test_preview_maps_same_basename_to_exact_normalized_path(self) -> None:
        first = r"C:\first\same name.png"
        second = r"C:\second\same name.png"
        model, _pool = self.make_model([first, second])
        subject, calls = self.make_preview_subject(
            image_path="",
            image_list=[first, second],
            current_index=99,
        )

        application.CropperApp._preview_from_thumb_index(
            subject,
            model.index(1, 0),
        )

        self.assertEqual(subject.current_index, 1)
        self.assertEqual(calls["opened"], [second])
        self.assertEqual(calls["preserved"], [True])
        self.assertEqual(calls["synced"], [True])
        self.assertEqual(subject._suspend_chain_clear, 0)

    def test_preview_sets_unset_or_out_of_range_image_index_by_path(self) -> None:
        paths = [r"C:\images\first.png", r"C:\images\second.png"]
        model, _pool = self.make_model(paths)

        for initial_index in (None, -1, 20):
            with self.subTest(initial_index=initial_index):
                subject, calls = self.make_preview_subject(
                    image_path="",
                    image_list=paths,
                    current_index=initial_index,
                )

                application.CropperApp._preview_from_thumb_index(
                    subject,
                    model.index(1, 0),
                )

                self.assertEqual(subject.current_index, 1)
                self.assertEqual(calls["opened"], [paths[1]])

    def test_preview_of_path_missing_from_image_list_opens_with_index_minus_one(
        self,
    ) -> None:
        selected = r"C:\browser-only\selected.png"
        model, _pool = self.make_model([selected])
        subject, calls = self.make_preview_subject(
            image_path="",
            image_list=[],
            current_index=7,
        )

        application.CropperApp._preview_from_thumb_index(
            subject,
            model.index(0, 0),
        )

        self.assertEqual(subject.current_index, -1)
        self.assertEqual(calls["opened"], [selected])

    def test_reselecting_current_image_returns_before_repairing_index(
        self,
    ) -> None:
        selected = r"C:\images\selected.png"
        model, _pool = self.make_model([selected])
        subject, calls = self.make_preview_subject(
            image_path=selected,
            image_list=[selected],
            current_index=42,
        )

        application.CropperApp._preview_from_thumb_index(
            subject,
            model.index(0, 0),
        )

        # Current behavior suppresses the duplicate open before current_index
        # is synchronized, so an already-stale index remains unchanged.
        self.assertEqual(subject.current_index, 42)
        self.assertEqual(calls["opened"], [])
        self.assertEqual(calls["preserved"], [])
        self.assertEqual(calls["synced"], [])

    def test_selection_resync_tracks_remaining_path_after_reorder_and_clears_on_reset(
        self,
    ) -> None:
        first = r"C:\one\same.png"
        selected = r"C:\two\same.png"
        model, _pool = self.make_model([first, selected])
        listview = QtWidgets.QListView()
        listview.setModel(model)
        self._widgets.append(listview)
        subject = SimpleNamespace(
            image_path=selected,
            model=model,
            listview=listview,
        )

        application.CropperApp._sync_thumb_selection(subject)
        self.assertEqual(listview.currentIndex().row(), 1)

        model.reset_items([selected, first])
        application.CropperApp._sync_thumb_selection(subject)
        self.assertEqual(listview.currentIndex().row(), 0)

        model.reset_items([first])
        application.CropperApp._sync_thumb_selection(subject)
        self.assertFalse(listview.currentIndex().isValid())

    def test_selection_resync_normalizes_zip_case_and_slashes(self) -> None:
        stored = application.make_zip_uri(
            r"C:\Archives\Mixed Case.zip",
            "Folder/Photo.PNG",
        )
        alternate = (
            "zip://"
            + stored[len("zip://") :].upper().replace("/", "\\")
        )
        model, _pool = self.make_model([stored])
        listview = QtWidgets.QListView()
        listview.setModel(model)
        self._widgets.append(listview)
        subject = SimpleNamespace(
            image_path=alternate,
            model=model,
            listview=listview,
        )

        application.CropperApp._sync_thumb_selection(subject)

        self.assertTrue(listview.currentIndex().isValid())
        self.assertEqual(listview.currentIndex().row(), 0)


class CustomListViewKeyBehaviorTests(_NavigationTestCase):
    def test_ctrl_arrow_is_consumed_but_does_nothing_with_current_construction(
        self,
    ) -> None:
        # CropperApp currently constructs CustomListView() without passing
        # itself, leaving mainwin=None.
        view = application.CustomListView()
        self._widgets.append(view)
        event = _KeyEvent(
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        view.keyPressEvent(event)  # type: ignore[arg-type]

        self.assertIsNone(view.mainwin)
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)
        self.assertFalse(view.isVisible())

    def test_ctrl_arrow_calls_preserve_then_move_when_mainwin_is_attached(
        self,
    ) -> None:
        calls: list[Any] = []
        mainwin = SimpleNamespace(
            _prepare_preserve_for_nav=lambda: calls.append("preserve"),
            _move_thumb_focus=lambda delta: calls.append(("move", delta)),
        )
        view = application.CustomListView()
        view.mainwin = mainwin
        self._widgets.append(view)

        for key, delta in (
            (QtCore.Qt.Key.Key_Left, -1),
            (QtCore.Qt.Key.Key_Right, 1),
        ):
            with self.subTest(key=key):
                calls.clear()
                event = _KeyEvent(
                    key,
                    QtCore.Qt.KeyboardModifier.ControlModifier,
                )

                view.keyPressEvent(event)  # type: ignore[arg-type]

                self.assertEqual(calls, ["preserve", ("move", delta)])
                self.assertTrue(event.accepted)

    def test_plain_left_and_right_are_ignored(self) -> None:
        view = application.CustomListView()
        self._widgets.append(view)

        for key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            with self.subTest(key=key):
                event = _KeyEvent(
                    key,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )

                view.keyPressEvent(event)  # type: ignore[arg-type]

                self.assertTrue(event.ignored)
                self.assertFalse(event.accepted)


def tearDownModule() -> None:
    """Remove bytecode created while unittest discovered this module."""
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
