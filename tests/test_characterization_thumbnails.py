"""Characterization tests for thumbnail models, generation, and stale results."""

from __future__ import annotations

import gc
import os
import sys
import unittest
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

sys.dont_write_bytecode = True

# The application deliberately preloads Torch before importing PyQt6.
import gazou_kiritori as application

try:
    from tests.helpers import (
        create_test_image,
        get_qapplication,
        temporary_directory,
    )
except ModuleNotFoundError:
    from helpers import (  # type: ignore[no-redef]
        create_test_image,
        get_qapplication,
        temporary_directory,
    )

from PIL import Image
from PyQt6 import QtCore, QtGui


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


def _png_bytes(
    color: tuple[int, ...],
    *,
    size: tuple[int, int] = (4, 3),
    mode: str = "RGB",
) -> bytes:
    output = BytesIO()
    image = Image.new(mode, size, color)
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    return output.getvalue()


def _content_bounds(
    image: Image.Image,
    background: tuple[int, int, int] = (240, 240, 240),
) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    try:
        points = [
            (x, y)
            for y in range(rgb.height)
            for x in range(rgb.width)
            if rgb.getpixel((x, y)) != background
        ]
    finally:
        rgb.close()
    if not points:
        return None
    left = min(x for x, _y in points)
    top = min(y for _x, y in points)
    right = max(x for x, _y in points)
    bottom = max(y for _x, y in points)
    return left, top, right - left + 1, bottom - top + 1


def _pixmap_rgb(pixmap: QtGui.QPixmap, x: int = 0, y: int = 0) -> tuple[int, int, int]:
    color = pixmap.toImage().pixelColor(x, y)
    return color.red(), color.green(), color.blue()


class _RecordingPool:
    """QThreadPool stand-in that never starts a background worker."""

    def __init__(self) -> None:
        self.tasks: list[Any] = []
        self.clear_count = 0

    def start(self, task: Any) -> None:
        self.tasks.append(task)

    def clear(self) -> None:
        self.clear_count += 1
        self.tasks.clear()


class _ThumbnailTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qapplication = get_qapplication()

    def setUp(self) -> None:
        self._models: list[application.ThumbnailListModel] = []
        self._archive_cache_keys: set[str] = set()
        self._saved_mem_zip_bytes = dict(application._MEM_ZIP_BYTES)
        self._saved_mem_zip_meta = {
            key: dict(value) for key, value in application._MEM_ZIP_META.items()
        }
        self._saved_mem_zip_counter = application._MEM_ZIP_COUNTER
        self._saved_image_cache = OrderedDict(application._IMG_CACHE)

        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

    def tearDown(self) -> None:
        self._restore_module_state()
        for model in self._models:
            model.deleteLater()
        self.qapplication.processEvents()
        self.assertEqual(QtCore.QThreadPool.globalInstance().activeThreadCount(), 0)

    def make_model(
        self,
        paths: list[str],
        *,
        thumb_size: tuple[int, int] = (80, 120),
    ) -> tuple[application.ThumbnailListModel, _RecordingPool]:
        model = application.ThumbnailListModel(paths, thumb_size=thumb_size)
        pool = _RecordingPool()
        model._pool = pool
        self._models.append(model)
        return model, pool

    def _write_archive(
        self,
        destination: Path,
        entries: list[tuple[str, bytes]],
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        self._archive_cache_keys.add(str(destination))
        return destination

    @staticmethod
    def _inner_archive_bytes(
        entries: list[tuple[str, bytes]],
    ) -> bytes:
        output = BytesIO()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return output.getvalue()

    def _close_archive_caches(self) -> None:
        keys = self._archive_cache_keys | set(application._MEM_ZIP_BYTES)
        for key in keys:
            try:
                archive = application._open_zip_cached(key)
            except (FileNotFoundError, KeyError):
                continue
            try:
                archive.close()
            except Exception:
                pass
        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()
        self._archive_cache_keys.clear()

    def _restore_module_state(self) -> None:
        self._close_archive_caches()

        for key, item in tuple(application._IMG_CACHE.items()):
            if (
                key not in self._saved_image_cache
                or item is not self._saved_image_cache[key]
            ):
                image = item.get("img") if isinstance(item, dict) else None
                if image is not None:
                    image.close()
        application._IMG_CACHE.clear()
        application._IMG_CACHE.update(self._saved_image_cache)

        application._MEM_ZIP_BYTES.clear()
        application._MEM_ZIP_BYTES.update(self._saved_mem_zip_bytes)
        application._MEM_ZIP_META.clear()
        application._MEM_ZIP_META.update(
            {
                key: dict(value)
                for key, value in self._saved_mem_zip_meta.items()
            }
        )
        application._MEM_ZIP_COUNTER = self._saved_mem_zip_counter

    @contextmanager
    def archive_workspace(self) -> Iterator[Path]:
        with temporary_directory(
            prefix="gazou-kiritori-thumbnails-"
        ) as temporary_root:
            try:
                yield temporary_root
            finally:
                # Cached ZipFile objects otherwise prevent Windows cleanup.
                self._restore_module_state()


class ThumbnailListModelContractTests(_ThumbnailTestCase):
    def test_row_count_contract_for_empty_one_multiple_and_parent(self) -> None:
        model, _pool = self.make_model([])
        self.assertEqual(model.rowCount(), 0)

        model.reset_items(["only.png"])
        self.assertEqual(model.rowCount(), 1)

        model.reset_items(["first.png", "second.png", "third.png"])
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.rowCount(model.createIndex(0, 0)), 0)

    def test_invalid_and_out_of_range_indexes_return_none_for_every_role(
        self,
    ) -> None:
        model, _pool = self.make_model(["only.png"])
        invalid = QtCore.QModelIndex()
        out_of_range = model.createIndex(50, 0)
        roles = (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.DecorationRole,
            QtCore.Qt.ItemDataRole.UserRole,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )

        for index in (invalid, out_of_range):
            for role in roles:
                with self.subTest(valid=index.isValid(), role=role):
                    self.assertIsNone(model.data(index, role))

    def test_display_and_user_roles_preserve_full_identity_and_display_basename(
        self,
    ) -> None:
        first = r"C:\first\same.png"
        second = r"D:\second\same.png"
        spaced = r"C:\images\name with spaces.png"
        japanese = r"C:\images\日本語画像.png"
        zip_image = application.make_zip_uri(
            r"C:\archives\gallery.zip",
            "folder/zip image.png",
        )
        nested_image = application.make_zip_uri(
            "memzip:123",
            "deeper/ネスト画像.png",
        )
        paths = [first, second, spaced, japanese, zip_image, nested_image]
        model, _pool = self.make_model(paths)

        expected_names = (
            "same.png",
            "same.png",
            "name with spaces.png",
            "日本語画像.png",
            "zip image.png",
            "ネスト画像.png",
        )
        for row, (path, expected_name) in enumerate(zip(paths, expected_names)):
            with self.subTest(row=row, path=path):
                index = model.index(row, 0)
                self.assertEqual(
                    model.data(index, QtCore.Qt.ItemDataRole.DisplayRole),
                    expected_name,
                )
                self.assertEqual(
                    model.data(index, QtCore.Qt.ItemDataRole.UserRole),
                    {"path": path, "is_dir": False},
                )

        self.assertNotEqual(
            model.data(model.index(0, 0), QtCore.Qt.ItemDataRole.UserRole)[
                "path"
            ],
            model.data(model.index(1, 0), QtCore.Qt.ItemDataRole.UserRole)[
                "path"
            ],
        )

    def test_decoration_role_returns_placeholder_and_queues_each_row_once(
        self,
    ) -> None:
        model, pool = self.make_model(["first.png", "second.png"])

        first_icon = model.data(
            model.index(0, 0),
            QtCore.Qt.ItemDataRole.DecorationRole,
        )
        repeated_icon = model.data(
            model.index(0, 0),
            QtCore.Qt.ItemDataRole.DecorationRole,
        )
        second_icon = model.data(
            model.index(1, 0),
            QtCore.Qt.ItemDataRole.DecorationRole,
        )

        self.assertIsInstance(first_icon, QtGui.QIcon)
        self.assertIsInstance(repeated_icon, QtGui.QIcon)
        self.assertIsInstance(second_icon, QtGui.QIcon)
        self.assertFalse(first_icon.isNull())
        self.assertEqual([task.row for task in pool.tasks], [0, 1])
        self.assertEqual(model._pending_rows, {0, 1})

    def test_tooltip_records_current_file_metadata_and_failure_fallback(
        self,
    ) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-tooltip-"
        ) as temporary_root:
            valid = create_test_image(
                temporary_root / "日本語 image.png",
                size=(7, 5),
                color=(12, 34, 56),
            )
            missing = temporary_root / "missing.png"
            model, _pool = self.make_model([str(valid), str(missing)])

            valid_tip = model.data(
                model.index(0, 0),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
            missing_tip = model.data(
                model.index(1, 0),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )

            self.assertIn("ファイル名: 日本語 image.png", valid_tip)
            self.assertIn("解像度: 7 x 5", valid_tip)
            self.assertIn("サイズ:", valid_tip)
            self.assertIn("ファイル名: missing.png", missing_tip)
            self.assertIn("解像度: 取得失敗", missing_tip)
            self.assertIn("サイズ: ? KB", missing_tip)
            gc.collect()

    def test_reset_items_uses_model_reset_for_all_list_shape_changes(
        self,
    ) -> None:
        model, pool = self.make_model([])
        resets: list[tuple[str, ...]] = []
        inserted: list[tuple[int, int]] = []
        removed: list[tuple[int, int]] = []
        model.modelReset.connect(
            lambda: resets.append(tuple(model.image_list))
        )
        model.rowsInserted.connect(
            lambda _parent, first, last: inserted.append((first, last))
        )
        model.rowsRemoved.connect(
            lambda _parent, first, last: removed.append((first, last))
        )
        transitions = (
            ["a.png", "b.png"],
            [],
            ["a.png", "b.png"],
            ["c.png", "d.png"],
            ["c.png", "d.png", "e.png"],
            ["c.png", "e.png"],
            ["e.png", "c.png"],
        )

        for paths in transitions:
            model.reset_items(paths)
            self.assertEqual(model.image_list, paths)
            self.assertEqual(model.thumbnails, [None] * len(paths))
            self.assertEqual(model._pending_rows, set())

        self.assertEqual(resets, [tuple(paths) for paths in transitions])
        self.assertEqual(inserted, [])
        self.assertEqual(removed, [])
        self.assertEqual(pool.clear_count, len(transitions))

    def test_single_thumbnail_update_notifies_only_its_current_row(self) -> None:
        paths = ["first.png", "second.png", "third.png"]
        model, _pool = self.make_model(paths)
        first_marker = QtGui.QPixmap(2, 2)
        first_marker.fill(QtGui.QColor(1, 2, 3))
        third_marker = QtGui.QPixmap(2, 2)
        third_marker.fill(QtGui.QColor(7, 8, 9))
        model.thumbnails = [first_marker, None, third_marker]
        events: list[tuple[int, int, tuple[int, ...]]] = []
        model.dataChanged.connect(
            lambda top, bottom, roles: events.append(
                (
                    top.row(),
                    bottom.row(),
                    tuple(int(role) for role in roles),
                )
            )
        )

        model._apply_thumb(
            1,
            paths[1],
            _png_bytes((20, 30, 40)),
            model._gen,
        )

        self.assertIs(model.thumbnails[0], first_marker)
        self.assertIs(model.thumbnails[2], third_marker)
        self.assertEqual(_pixmap_rgb(model.thumbnails[1]), (20, 30, 40))
        self.assertEqual(
            events,
            [
                (
                    1,
                    1,
                    (int(QtCore.Qt.ItemDataRole.DecorationRole),),
                )
            ],
        )


class FixedThumbnailGenerationTests(_ThumbnailTestCase):
    def test_small_images_are_centered_without_upscaling(self) -> None:
        cases = (
            ((12, 6), (34, 57, 12, 6)),
            ((6, 12), (37, 54, 6, 12)),
            ((8, 8), (36, 56, 8, 8)),
            ((1, 1), (39, 59, 1, 1)),
        )
        with temporary_directory(
            prefix="gazou-kiritori-thumb-small-"
        ) as temporary_root:
            for number, (source_size, expected_bounds) in enumerate(cases):
                with self.subTest(source_size=source_size):
                    path = create_test_image(
                        temporary_root / f"small-{number}.png",
                        size=source_size,
                        color=(10, 40, 90),
                    )

                    thumbnail = application.make_fixed_thumbnail_any(str(path))
                    try:
                        self.assertEqual(thumbnail.size, (80, 120))
                        self.assertEqual(thumbnail.mode, "RGB")
                        self.assertEqual(
                            _content_bounds(thumbnail),
                            expected_bounds,
                        )
                    finally:
                        thumbnail.close()

    def test_large_images_shrink_with_aspect_ratio_preserved(self) -> None:
        cases = (
            ((160, 40), (0, 50, 80, 20)),
            ((40, 240), (30, 0, 20, 120)),
            ((200, 200), (0, 20, 80, 80)),
        )
        with temporary_directory(
            prefix="gazou-kiritori-thumb-large-"
        ) as temporary_root:
            for number, (source_size, expected_bounds) in enumerate(cases):
                with self.subTest(source_size=source_size):
                    path = create_test_image(
                        temporary_root / f"large-{number}.png",
                        size=source_size,
                        color=(70, 30, 10),
                    )

                    thumbnail = application.make_fixed_thumbnail_any(str(path))
                    try:
                        self.assertEqual(thumbnail.size, (80, 120))
                        self.assertEqual(
                            _content_bounds(thumbnail),
                            expected_bounds,
                        )
                        source_ratio = source_size[0] / source_size[1]
                        result_ratio = (
                            expected_bounds[2] / expected_bounds[3]
                        )
                        self.assertAlmostEqual(source_ratio, result_ratio)
                    finally:
                        thumbnail.close()

    def test_rgb_and_rgba_inputs_return_rgb_and_discard_alpha_channel(
        self,
    ) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-alpha-"
        ) as temporary_root:
            rgb_path = create_test_image(
                temporary_root / "rgb.png",
                size=(3, 2),
                mode="RGB",
                color=(11, 22, 33),
            )
            rgba_path = create_test_image(
                temporary_root / "rgba.png",
                size=(3, 2),
                mode="RGBA",
                color=(44, 55, 66, 0),
            )

            rgb = application.make_fixed_thumbnail_any(str(rgb_path))
            rgba = application.make_fixed_thumbnail_any(str(rgba_path))
            try:
                self.assertEqual(rgb.mode, "RGB")
                self.assertEqual(rgba.mode, "RGB")
                self.assertEqual(rgb.getpixel((39, 59)), (11, 22, 33))
                # Current behavior pastes RGBA onto RGB without an alpha mask.
                self.assertEqual(rgba.getpixel((39, 59)), (44, 55, 66))
            finally:
                rgb.close()
                rgba.close()

    def test_normal_zip_and_nested_zip_images_share_fixed_canvas_behavior(
        self,
    ) -> None:
        payload = _png_bytes((15, 45, 75), size=(10, 5))
        with self.archive_workspace() as temporary_root:
            normal = temporary_root / "normal.png"
            normal.write_bytes(payload)
            inner_bytes = self._inner_archive_bytes(
                [("nested image.png", payload)]
            )
            outer = self._write_archive(
                temporary_root / "outer.zip",
                [
                    ("zip image.png", payload),
                    ("archives/inner.zip", inner_bytes),
                ],
            )
            zip_image = application.make_zip_uri(
                str(outer),
                "zip image.png",
            )
            inner_entry = application.make_zip_uri(
                str(outer),
                "archives/inner.zip",
            )
            nested_image = application.vfs_listdir(inner_entry)[0]["uri"]

            for identity in (str(normal), zip_image, nested_image):
                with self.subTest(identity=identity):
                    thumbnail = application.make_fixed_thumbnail_any(identity)
                    try:
                        self.assertEqual(thumbnail.size, (80, 120))
                        self.assertEqual(
                            _content_bounds(thumbnail),
                            (35, 57, 10, 5),
                        )
                    finally:
                        thumbnail.close()

    def test_unreadable_image_paths_return_current_placeholders_without_artifacts(
        self,
    ) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-failure-"
        ) as temporary_root:
            missing = temporary_root / "missing.png"
            broken = temporary_root / "broken.png"
            non_image = temporary_root / "payload.txt"
            broken.write_bytes(b"not a Pillow image")
            non_image.write_bytes(b"not an image by extension")
            before = {
                path.relative_to(temporary_root)
                for path in temporary_root.rglob("*")
            }

            missing_thumb = application.make_fixed_thumbnail_any(str(missing))
            broken_thumb = application.make_fixed_thumbnail_any(str(broken))
            non_image_thumb = application.make_fixed_thumbnail_any(str(non_image))
            try:
                self.assertEqual(
                    missing_thumb.getpixel((0, 0)),
                    (90, 90, 90),
                )
                self.assertEqual(
                    broken_thumb.getpixel((0, 0)),
                    (90, 90, 90),
                )
                self.assertEqual(
                    non_image_thumb.getpixel((0, 0)),
                    (200, 200, 200),
                )
                self.assertEqual(
                    {
                        path.relative_to(temporary_root)
                        for path in temporary_root.rglob("*")
                    },
                    before,
                )
            finally:
                missing_thumb.close()
                broken_thumb.close()
                non_image_thumb.close()

    def test_source_file_bytes_are_not_modified(self) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-source-"
        ) as temporary_root:
            source = create_test_image(
                temporary_root / "source.png",
                size=(160, 40),
                color=(90, 40, 20),
            )
            before_bytes = source.read_bytes()
            before_paths = {
                path.relative_to(temporary_root)
                for path in temporary_root.rglob("*")
            }

            thumbnail = application.make_fixed_thumbnail_any(str(source))
            thumbnail.close()

            self.assertEqual(source.read_bytes(), before_bytes)
            self.assertEqual(
                {
                    path.relative_to(temporary_root)
                    for path in temporary_root.rglob("*")
                },
                before_paths,
            )


class ThumbnailCacheCharacterizationTests(_ThumbnailTestCase):
    def test_repeated_generation_and_equivalent_physical_spelling_hit_cache(
        self,
    ) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-cache-"
        ) as temporary_root:
            image_path = create_test_image(
                temporary_root / "same.png",
                size=(20, 10),
                color=(21, 31, 41),
            )
            alternate = (
                str(image_path.parent)
                + os.sep
                + "."
                + os.sep
                + image_path.name
            )
            model, _pool = self.make_model([str(image_path), alternate])

            with mock.patch.object(
                application,
                "make_fixed_thumbnail_any",
                wraps=application.make_fixed_thumbnail_any,
            ) as generate:
                model._generate_thumb(0)
                model._generate_thumb(0)
                model._generate_thumb(1)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(len(model._cache), 1)
            self.assertEqual(
                application.norm_vpath(str(image_path)),
                application.norm_vpath(alternate),
            )
            self.assertEqual(
                _pixmap_rgb(model.thumbnails[0], 39, 59),
                (21, 31, 41),
            )
            self.assertEqual(
                _pixmap_rgb(model.thumbnails[1], 39, 59),
                (21, 31, 41),
            )

    def test_physical_file_and_zip_entry_use_distinct_cache_identities(
        self,
    ) -> None:
        payload = _png_bytes((30, 60, 90), size=(12, 6))
        with self.archive_workspace() as temporary_root:
            physical = temporary_root / "same.png"
            physical.write_bytes(payload)
            archive = self._write_archive(
                temporary_root / "images.zip",
                [("same.png", payload)],
            )
            zip_entry = application.make_zip_uri(
                str(archive),
                "same.png",
            )
            model, _pool = self.make_model([str(physical), zip_entry])

            model._generate_thumb(0)
            model._generate_thumb(1)

            self.assertEqual(
                set(model._cache),
                {
                    application.norm_vpath(str(physical)),
                    application.norm_vpath(zip_entry),
                },
            )
            self.assertEqual(len(model._cache), 2)

    def test_reset_preserves_cache_and_deleted_source_reuses_cached_png(
        self,
    ) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-cache-delete-"
        ) as temporary_root:
            image_path = create_test_image(
                temporary_root / "cached.png",
                size=(10, 5),
                color=(80, 20, 10),
            )
            model, _pool = self.make_model([str(image_path)])
            with mock.patch.object(
                application,
                "make_fixed_thumbnail_any",
                wraps=application.make_fixed_thumbnail_any,
            ) as generate:
                model._generate_thumb(0)
                cache_before = dict(model._cache)
                model.reset_items([str(image_path)])
                self.assertEqual(model._cache, cache_before)

                image_path.unlink()
                model._generate_thumb(0)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(
                _pixmap_rgb(model.thumbnails[0], 39, 59),
                (80, 20, 10),
            )

    def test_source_signature_change_rebuilds_cached_thumbnail(self) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-cache-update-"
        ) as temporary_root:
            image_path = create_test_image(
                temporary_root / "changing.png",
                size=(10, 5),
                color=(10, 20, 30),
            )
            model, _pool = self.make_model([str(image_path)])
            with mock.patch.object(
                application,
                "make_fixed_thumbnail_any",
                wraps=application.make_fixed_thumbnail_any,
            ) as generate:
                model._generate_thumb(0)
                old_signature = model._cache[
                    application.norm_vpath(str(image_path))
                ][0]

                create_test_image(
                    image_path,
                    size=(40, 20),
                    color=(100, 110, 120),
                )
                stat = image_path.stat()
                os.utime(
                    image_path,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
                )
                model._generate_thumb(0)

            new_signature = model._cache[
                application.norm_vpath(str(image_path))
            ][0]
            self.assertEqual(generate.call_count, 2)
            self.assertNotEqual(new_signature, old_signature)
            self.assertEqual(
                _pixmap_rgb(model.thumbnails[0], 39, 59),
                (100, 110, 120),
            )

    def test_cache_is_owned_by_each_model_instance(self) -> None:
        with temporary_directory(
            prefix="gazou-kiritori-thumb-cache-owner-"
        ) as temporary_root:
            image_path = create_test_image(
                temporary_root / "owner.png",
                color=(13, 23, 33),
            )
            first, _first_pool = self.make_model([str(image_path)])
            second, _second_pool = self.make_model([str(image_path)])

            first._generate_thumb(0)

            self.assertIsNot(first._cache, second._cache)
            self.assertEqual(len(first._cache), 1)
            self.assertEqual(second._cache, {})
            self.assertEqual(application.ThumbnailListModel._cache, {})


class ThumbnailResultGenerationTests(_ThumbnailTestCase):
    def test_current_generation_applies_without_pending_or_worker_state(
        self,
    ) -> None:
        path = r"C:\images\current.png"
        model, _pool = self.make_model([path])
        self.assertEqual(model._pending_rows, set())

        model._apply_thumb(
            0,
            path,
            _png_bytes((12, 24, 36)),
            model._gen,
        )

        self.assertEqual(_pixmap_rgb(model.thumbnails[0]), (12, 24, 36))
        self.assertEqual(model._pending_rows, set())

    def test_old_generation_out_of_range_and_deleted_path_are_ignored(
        self,
    ) -> None:
        old_path = r"C:\images\old.png"
        current_path = r"C:\images\current.png"
        model, _pool = self.make_model([old_path])
        old_generation = model._gen
        model.reset_items([current_path])

        model._apply_thumb(
            0,
            old_path,
            _png_bytes((200, 10, 10)),
            old_generation,
        )
        model._apply_thumb(
            99,
            current_path,
            _png_bytes((10, 200, 10)),
            model._gen,
        )
        model._apply_thumb(
            0,
            old_path,
            _png_bytes((10, 10, 200)),
            model._gen,
        )

        self.assertEqual(model.thumbnails, [None])

    def test_current_generation_result_follows_matching_path_after_row_change(
        self,
    ) -> None:
        first = r"C:\one\same.png"
        second = r"C:\two\same.png"
        model, _pool = self.make_model([first, second])
        events: list[int] = []
        model.dataChanged.connect(
            lambda top, _bottom, _roles: events.append(top.row())
        )

        model._apply_thumb(
            0,
            second,
            _png_bytes((9, 19, 29)),
            model._gen,
        )

        self.assertIsNone(model.thumbnails[0])
        self.assertEqual(_pixmap_rgb(model.thumbnails[1]), (9, 19, 29))
        self.assertEqual(events, [1])

    def test_multiple_current_results_apply_in_arrival_order_last_wins(
        self,
    ) -> None:
        path = r"C:\images\same.png"
        model, _pool = self.make_model([path])

        model._apply_thumb(
            0,
            path,
            _png_bytes((10, 20, 30)),
            model._gen,
        )
        model._apply_thumb(
            0,
            path,
            _png_bytes((70, 80, 90)),
            model._gen,
        )

        self.assertEqual(_pixmap_rgb(model.thumbnails[0]), (70, 80, 90))

    def test_running_task_uses_new_generation_and_relocates_after_reorder(
        self,
    ) -> None:
        # Current observed behavior: _generate_thumb captures its path early,
        # but reads self._gen only when emitting. A reset during generation
        # therefore lets old work adopt the new generation and follow its path.
        first = r"C:\images\first.png"
        second = r"C:\images\second.png"
        model, _pool = self.make_model([first, second])
        old_generation = model._gen

        def reset_during_generation(
            _path: str,
            thumb_size: tuple[int, int],
        ) -> Image.Image:
            model.reset_items([second, first])
            return Image.new("RGB", thumb_size, (31, 41, 51))

        with (
            mock.patch.object(application, "_sig_for", return_value=("sig",)),
            mock.patch.object(
                application,
                "make_fixed_thumbnail_any",
                side_effect=reset_during_generation,
            ),
        ):
            model._generate_thumb(0)

        self.assertNotEqual(model._gen, old_generation)
        self.assertIsNone(model.thumbnails[0])
        self.assertEqual(_pixmap_rgb(model.thumbnails[1]), (31, 41, 51))

    def test_running_task_for_removed_path_is_discarded_after_reset(
        self,
    ) -> None:
        old_path = r"C:\images\old.png"
        remaining = r"C:\images\remaining.png"
        model, _pool = self.make_model([old_path, remaining])

        def remove_during_generation(
            _path: str,
            thumb_size: tuple[int, int],
        ) -> Image.Image:
            model.reset_items([remaining])
            return Image.new("RGB", thumb_size, (61, 71, 81))

        with (
            mock.patch.object(application, "_sig_for", return_value=("sig",)),
            mock.patch.object(
                application,
                "make_fixed_thumbnail_any",
                side_effect=remove_during_generation,
            ),
        ):
            model._generate_thumb(0)

        self.assertEqual(model.thumbnails, [None])
        self.assertIn(application.norm_vpath(old_path), model._cache)

    def test_directory_overlay_requires_exact_generation_row_and_path(
        self,
    ) -> None:
        first = r"C:\folders\first"
        second = r"C:\folders\second"
        model, _pool = self.make_model([first, second])
        composed = QtGui.QPixmap(3, 3)
        composed.fill(QtGui.QColor(81, 91, 101))
        model._compose_folder_pm = (  # type: ignore[method-assign]
            lambda _path, _image: composed
        )
        png = _png_bytes((1, 2, 3))

        model._apply_dir_overlay(0, second, png, model._gen)
        model._apply_dir_overlay(0, first, png, model._gen - 1)
        self.assertEqual(model.thumbnails, [None, None])

        model._apply_dir_overlay(0, first, png, model._gen)
        self.assertIs(model.thumbnails[0], composed)
        self.assertIsNone(model.thumbnails[1])

    def test_thumb_task_retains_only_row_and_calls_bound_function_on_run(
        self,
    ) -> None:
        calls: list[int] = []
        task = application._ThumbTask(calls.append, 3)

        task.run()

        self.assertEqual(calls, [3])
        self.assertEqual(task.row, 3)
        self.assertFalse(hasattr(task, "path"))
        self.assertFalse(hasattr(task, "gen"))


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
