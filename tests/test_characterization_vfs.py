"""Characterization tests for the current ZIP-backed VFS behavior."""

from __future__ import annotations

import os
import sys
import unittest
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator

sys.dont_write_bytecode = True

try:
    from tests.helpers import temporary_directory
except ModuleNotFoundError:
    from helpers import temporary_directory  # type: ignore[no-redef]

from PIL import Image

import gazou_kiritori as application


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


def _png_bytes(
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (5, 4),
) -> bytes:
    """Return a complete small PNG without touching the filesystem."""
    output = BytesIO()
    image = Image.new("RGB", size, color)
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    return output.getvalue()


class ZipUriCharacterizationTests(unittest.TestCase):
    def test_make_and_parse_round_trip_root_names_and_subdirectories(self) -> None:
        archive_path = os.path.abspath(
            os.path.join("C:\\", "VFS fixtures", "日本語 archive!.zip")
        )
        inner_paths = (
            "",
            "gallery/",
            "gallery/deeper/photo.png",
            "gallery/file with spaces.png",
            "日本語/画像.png",
            "punctuation/bang!.png",
        )

        for inner_path in inner_paths:
            with self.subTest(inner_path=inner_path):
                uri = application.make_zip_uri(archive_path, inner_path)
                parsed_archive, parsed_inner = application.parse_zip_uri(uri)

                self.assertEqual(parsed_archive, archive_path)
                self.assertEqual(parsed_inner, inner_path)

    def test_windows_drive_and_slash_normalization_are_preserved(self) -> None:
        archive_path = r"C:\VFS fixtures\archive!.zip"
        uri = application.make_zip_uri(
            archive_path,
            r"Sub Directory\日本語 Image.PNG",
        )

        parsed_archive, parsed_inner = application.parse_zip_uri(uri)

        self.assertEqual(parsed_archive, os.path.abspath(archive_path))
        self.assertEqual(parsed_inner, "Sub Directory/日本語 Image.PNG")
        self.assertEqual(
            application.norm_vpath(uri),
            uri.lower().replace("\\", "/"),
        )
        self.assertEqual(
            application.norm_vpath(r"C:\VFS fixtures\folder\..\image.PNG"),
            application.norm_vpath(r"c:/vfs fixtures/image.png"),
        )

    def test_parse_rejects_non_zip_uri(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a zip uri"):
            application.parse_zip_uri(r"C:\images\plain.png")


class _IsolatedVfsTestCase(unittest.TestCase):
    """Keep the module-level archive/image caches from leaking across tests."""

    def setUp(self) -> None:
        self._saved_mem_zip_bytes = dict(application._MEM_ZIP_BYTES)
        self._saved_mem_zip_meta = {
            key: dict(value) for key, value in application._MEM_ZIP_META.items()
        }
        self._saved_mem_zip_counter = application._MEM_ZIP_COUNTER
        self._saved_image_cache = OrderedDict(application._IMG_CACHE)
        self._archive_cache_keys: set[str] = set()

        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

    def tearDown(self) -> None:
        self._restore_module_state()

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

    def _close_archive_caches(self) -> None:
        cache_keys = self._archive_cache_keys | set(application._MEM_ZIP_BYTES)
        for cache_key in cache_keys:
            try:
                archive = application._open_zip_cached(cache_key)
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
    def _archive_workspace(self) -> Iterator[Path]:
        temporary_root: Path
        with temporary_directory(prefix="gazou-kiritori-vfs-") as temporary_root:
            try:
                yield temporary_root
            finally:
                # Windows cannot remove an archive while a cached reader owns it.
                self._restore_module_state()


class ZipListingAndReadingTests(_IsolatedVfsTestCase):
    def test_zip_root_and_subdirectory_listing(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "listing.zip",
                [
                    ("root.png", _png_bytes((10, 20, 30))),
                    ("gallery/photo.png", _png_bytes((40, 50, 60))),
                    ("gallery/deeper/notes.txt", b"notes"),
                    ("documents/readme.txt", b"readme"),
                ],
            )

            root_items = application.vfs_listdir(str(archive_path))
            root_summary = {
                item["name"]: item["is_dir"] for item in root_items
            }
            self.assertEqual(
                root_summary,
                {
                    "documents": True,
                    "gallery": True,
                    "root.png": False,
                },
            )

            gallery_uri = next(
                item["uri"] for item in root_items if item["name"] == "gallery"
            )
            gallery_items = application.vfs_listdir(gallery_uri)
            gallery_summary = {
                item["name"]: item["is_dir"] for item in gallery_items
            }
            self.assertEqual(
                gallery_summary,
                {"deeper": True, "photo.png": False},
            )

    def test_zip_entry_type_checks_are_syntactic(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "types.zip",
                [("gallery/photo.png", _png_bytes((70, 80, 90)))],
            )
            root_uri = application.make_zip_uri(str(archive_path), "")
            directory_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/",
            )
            image_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/photo.png",
            )
            missing_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/missing.png",
            )

            self.assertTrue(application.vfs_is_dir(root_uri))
            self.assertTrue(application.vfs_is_dir(directory_uri))
            self.assertFalse(application.vfs_is_file(directory_uri))
            self.assertTrue(application.vfs_is_file(image_uri))
            # Current behavior checks URI shape, not entry existence.
            self.assertTrue(application.vfs_is_file(missing_uri))

    def test_zip_bytes_and_pillow_image_loading(self) -> None:
        expected_png = _png_bytes((12, 34, 56), size=(7, 6))
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "reading.zip",
                [
                    ("payload/data.bin", b"characterization-bytes"),
                    ("images/sample.png", expected_png),
                ],
            )
            bytes_uri = application.make_zip_uri(
                str(archive_path),
                "payload/data.bin",
            )
            image_uri = application.make_zip_uri(
                str(archive_path),
                "images/sample.png",
            )

            self.assertEqual(
                application.open_bytes_any(bytes_uri),
                b"characterization-bytes",
            )
            image = application.open_image_any(image_uri)
            try:
                self.assertEqual(image.size, (7, 6))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))
            finally:
                image.close()

    def test_parent_navigation_inside_and_outside_zip_root(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "parents.zip",
                [("gallery/deeper/photo.png", _png_bytes((1, 2, 3)))],
            )
            photo_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/deeper/photo.png",
            )
            deeper_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/deeper/",
            )
            gallery_uri = application.make_zip_uri(
                str(archive_path),
                "gallery/",
            )
            root_uri = application.make_zip_uri(str(archive_path), "")

            self.assertEqual(application.vfs_parent(photo_uri), deeper_uri)
            self.assertEqual(application.vfs_parent(deeper_uri), gallery_uri)
            self.assertEqual(application.vfs_parent(gallery_uri), root_uri)
            self.assertEqual(
                application.vfs_parent(root_uri),
                str(temporary_root),
            )

    def test_noise_entries_are_filtered_from_zip_listing(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "noise.zip",
                [
                    ("__MACOSX/", b""),
                    ("__MACOSX/._photo.png", b"metadata"),
                    ("._filename", b"metadata"),
                    ("normal.png", _png_bytes((100, 110, 120))),
                ],
            )

            root_items = application.vfs_listdir(str(archive_path))

            self.assertEqual(
                [(item["name"], item["is_dir"]) for item in root_items],
                [("normal.png", False)],
            )


class ZipCaseResolutionTests(_IsolatedVfsTestCase):
    def test_unique_entry_supports_exact_and_casefolded_image_lookup(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "case.zip",
                [("Mixed/Photo.PNG", _png_bytes((21, 31, 41)))],
            )
            exact_uri = application.make_zip_uri(
                str(archive_path),
                "Mixed/Photo.PNG",
            )
            casefolded_uri = application.make_zip_uri(
                str(archive_path),
                "mixed/photo.png",
            )

            self.assertEqual(
                application.open_bytes_any(exact_uri),
                _png_bytes((21, 31, 41)),
            )
            with self.assertRaises(KeyError):
                application.open_bytes_any(casefolded_uri)

            for uri in (exact_uri, casefolded_uri):
                with self.subTest(uri=uri):
                    image = application.open_image_any(uri)
                    try:
                        self.assertEqual(image.getpixel((0, 0)), (21, 31, 41))
                    finally:
                        image.close()

    def test_casefold_collision_is_last_entry_wins_even_for_exact_name(
        self,
    ) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "collision.zip",
                [
                    ("Case.png", _png_bytes((200, 10, 20))),
                    ("case.png", _png_bytes((20, 200, 10))),
                ],
            )

            self.assertEqual(
                application._zip_resolve_inner(
                    str(archive_path),
                    "Case.png",
                ),
                "case.png",
            )
            exact_first_uri = application.make_zip_uri(
                str(archive_path),
                "Case.png",
            )
            image = application.open_image_any(exact_first_uri)
            try:
                # The lowercase index collapses both entries and retains the last.
                self.assertEqual(image.getpixel((0, 0)), (20, 200, 10))
            finally:
                image.close()


class NestedZipCharacterizationTests(_IsolatedVfsTestCase):
    @staticmethod
    def _inner_archive_bytes() -> bytes:
        output = BytesIO()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("images/inside.png", _png_bytes((9, 19, 29)))
            archive.writestr("readme.txt", b"nested archive")
        return output.getvalue()

    def test_nested_zip_is_listed_opened_and_read_from_memory(self) -> None:
        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "outer.zip",
                [
                    ("nested/inner.zip", self._inner_archive_bytes()),
                    ("cover.png", _png_bytes((3, 4, 5))),
                ],
            )
            nested_directory_uri = application.make_zip_uri(
                str(outer_path),
                "nested/",
            )

            nested_items = application.vfs_listdir(nested_directory_uri)
            inner_item = next(
                item for item in nested_items if item["name"] == "inner.zip"
            )
            self.assertTrue(inner_item["is_dir"])

            inner_root_items = application.vfs_listdir(inner_item["uri"])
            inner_summary = {
                item["name"]: item["is_dir"] for item in inner_root_items
            }
            self.assertEqual(
                inner_summary,
                {"images": True, "readme.txt": False},
            )
            images_uri = next(
                item["uri"]
                for item in inner_root_items
                if item["name"] == "images"
            )
            inner_image_uri = application.vfs_listdir(images_uri)[0]["uri"]
            image = application.open_image_any(inner_image_uri)
            try:
                self.assertEqual(image.getpixel((0, 0)), (9, 19, 29))
            finally:
                image.close()

    def test_nested_zip_parent_and_repeated_open_reuse_registration(self) -> None:
        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "outer-repeat.zip",
                [("inner.zip", self._inner_archive_bytes())],
            )
            inner_entry_uri = application.make_zip_uri(
                str(outer_path),
                "inner.zip",
            )

            first_items = application.vfs_listdir(inner_entry_uri)
            second_items = application.vfs_listdir(inner_entry_uri)

            self.assertEqual(first_items, second_items)
            self.assertEqual(len(application._MEM_ZIP_BYTES), 1)
            self.assertEqual(len(application._MEM_ZIP_META), 1)
            memory_zip_id, _ = application.parse_zip_uri(first_items[0]["uri"])
            self.assertTrue(memory_zip_id.startswith("memzip:"))
            memory_root_uri = application.make_zip_uri(memory_zip_id, "")
            self.assertEqual(
                application.vfs_parent(memory_root_uri),
                inner_entry_uri,
            )


class ZipCacheCharacterizationTests(_IsolatedVfsTestCase):
    def test_archive_cache_reuses_exact_key_but_not_path_spelling(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "cache.zip",
                [("sample.png", _png_bytes((60, 70, 80)))],
            )
            exact_key = str(archive_path)
            alternate_key = str(archive_path.parent) + "\\.\\" + archive_path.name
            self._archive_cache_keys.add(alternate_key)

            first = application._open_zip_cached(exact_key)
            repeated = application._open_zip_cached(exact_key)
            alternate = application._open_zip_cached(alternate_key)

            self.assertIs(first, repeated)
            self.assertIsNot(first, alternate)
            self.assertEqual(application._open_zip_cached.cache_info().currsize, 2)
            self.assertEqual(
                application.norm_vpath(exact_key),
                application.norm_vpath(alternate_key),
            )

    def test_image_cache_uses_normalized_vfs_key_and_reuses_image(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "image-cache.zip",
                [("Folder/Image.PNG", _png_bytes((90, 100, 110)))],
            )
            image_uri = application.make_zip_uri(
                str(archive_path),
                "Folder/Image.PNG",
            )
            image = application.open_image_any(image_uri)
            application._cache_put(image_uri, image)

            first_hit = application._cache_get(image_uri)
            second_hit = application._cache_get(image_uri)

            self.assertIs(first_hit, image)
            self.assertIs(second_hit, image)
            self.assertEqual(
                tuple(application._IMG_CACHE),
                (application.norm_vpath(image_uri),),
            )

    def test_temporary_archives_and_global_state_are_restored(self) -> None:
        original_bytes = dict(application._MEM_ZIP_BYTES)
        original_meta = {
            key: dict(value) for key, value in application._MEM_ZIP_META.items()
        }
        original_counter = application._MEM_ZIP_COUNTER

        temporary_root: Path
        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "lifecycle.zip",
                [("inner.zip", NestedZipCharacterizationTests._inner_archive_bytes())],
            )
            inner_uri = application.make_zip_uri(
                str(outer_path),
                "inner.zip",
            )
            application.vfs_listdir(inner_uri)
            self.assertTrue(application._MEM_ZIP_BYTES)
            self.assertGreater(application._open_zip_cached.cache_info().currsize, 0)

        self.assertFalse(temporary_root.exists())
        self.assertEqual(application._MEM_ZIP_BYTES, original_bytes)
        self.assertEqual(application._MEM_ZIP_META, original_meta)
        self.assertEqual(application._MEM_ZIP_COUNTER, original_counter)
        self.assertEqual(application._open_zip_cached.cache_info().currsize, 0)
        self.assertEqual(application._zip_index_lower.cache_info().currsize, 0)


def tearDownModule() -> None:
    """Remove bytecode produced while unittest discovery imported this module."""
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
