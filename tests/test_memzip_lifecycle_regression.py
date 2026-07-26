"""Regression tests for nested in-memory ZIP registration lifecycles."""

from __future__ import annotations

import sys
import unittest
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator


sys.dont_write_bytecode = True

try:
    from tests.helpers import temporary_directory
except ModuleNotFoundError:
    from helpers import temporary_directory  # type: ignore[no-redef]

import gazou_kiritori as application


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


class MemZipLifecycleRegressionTests(unittest.TestCase):
    """Keep every reader, cache, and memzip mutation local to one test."""

    def setUp(self) -> None:
        self._saved_mem_zip_bytes = dict(application._MEM_ZIP_BYTES)
        self._saved_mem_zip_meta = {
            key: dict(value) for key, value in application._MEM_ZIP_META.items()
        }
        self._saved_mem_zip_counter = application._MEM_ZIP_COUNTER
        self._saved_open_zip_cache_state = (
            application._open_zip_cached.cache_info()
        )
        self._saved_zip_index_cache_state = (
            application._zip_index_lower.cache_info()
        )
        self._readers: list[zipfile.ZipFile] = []

        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()
        application._MEM_ZIP_BYTES.clear()
        application._MEM_ZIP_META.clear()

    def tearDown(self) -> None:
        self._restore_module_state()

    @staticmethod
    def _archive_bytes(entries: list[tuple[str, bytes]]) -> bytes:
        output = BytesIO()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return output.getvalue()

    @staticmethod
    def _write_archive(
        destination: Path,
        entries: list[tuple[str, bytes]],
    ) -> Path:
        with zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return destination

    def _track_reader(self, archive_id: str) -> zipfile.ZipFile:
        reader = application._open_zip_cached(archive_id)
        if all(reader is not existing for existing in self._readers):
            self._readers.append(reader)
        return reader

    @staticmethod
    def _memzip_id(items: list[dict]) -> str:
        memzip_id, _ = application.parse_zip_uri(items[0]["uri"])
        return memzip_id

    def _restore_module_state(self) -> None:
        for reader in reversed(self._readers):
            try:
                reader.close()
            except Exception:
                pass
        self._readers.clear()

        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

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
        with temporary_directory(
            prefix="gazou-kiritori-memzip-lifecycle-",
        ) as temporary_root:
            try:
                yield temporary_root
            finally:
                # Release Windows archive handles before removing the ZIP files.
                self._restore_module_state()

    def test_outer_archive_replacement_refreshes_nested_memzip_contents(
        self,
    ) -> None:
        version_one = self._archive_bytes([("old.txt", b"old")])
        version_two = self._archive_bytes(
            [("new.txt", b"new payload with a different size")]
        )
        starting_counter = application._MEM_ZIP_COUNTER

        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "outer.zip",
                [("inner.zip", version_one)],
            )
            inner_uri = application.make_zip_uri(
                str(outer_path),
                "inner.zip",
            )

            first_items = application.vfs_listdir(inner_uri)
            self.assertEqual(
                {item["name"] for item in first_items},
                {"old.txt"},
            )
            first_memzip_id = self._memzip_id(first_items)
            self.assertIn(first_memzip_id, application._MEM_ZIP_BYTES)
            self.assertIn(first_memzip_id, application._MEM_ZIP_META)
            first_reader = self._track_reader(first_memzip_id)
            first_bytes = application._MEM_ZIP_BYTES[first_memzip_id]
            first_metadata = dict(application._MEM_ZIP_META[first_memzip_id])
            first_signature = application._archive_cache_signature(
                first_memzip_id
            )
            self.assertEqual(first_reader.namelist(), ["old.txt"])
            self.assertEqual(first_reader.read("old.txt"), b"old")

            self._write_archive(
                outer_path,
                [("inner.zip", version_two)],
            )

            second_items = application.vfs_listdir(inner_uri)
            self.assertEqual(
                {item["name"] for item in second_items},
                {"new.txt"},
            )
            second_memzip_id = self._memzip_id(second_items)
            second_reader = self._track_reader(second_memzip_id)

            self.assertNotEqual(second_memzip_id, first_memzip_id)
            self.assertIsNotNone(first_reader.fp)
            self.assertEqual(first_reader.namelist(), ["old.txt"])
            self.assertEqual(first_reader.read("old.txt"), b"old")
            self.assertEqual(second_reader.namelist(), ["new.txt"])
            self.assertEqual(
                second_reader.read("new.txt"),
                b"new payload with a different size",
            )
            self.assertIs(
                application._MEM_ZIP_BYTES[first_memzip_id],
                first_bytes,
            )
            self.assertEqual(
                application._MEM_ZIP_META[first_memzip_id],
                first_metadata,
            )
            self.assertEqual(
                application._archive_cache_signature(first_memzip_id),
                first_signature,
            )
            self.assertEqual(
                set(application._MEM_ZIP_BYTES),
                set(application._MEM_ZIP_META),
            )
            self.assertEqual(
                application._MEM_ZIP_COUNTER,
                starting_counter + 2,
            )

    def test_distinct_inner_archives_have_distinct_consistent_registrations(
        self,
    ) -> None:
        first_bytes = self._archive_bytes(
            [("first-only.txt", b"first payload")]
        )
        second_bytes = self._archive_bytes(
            [("second-only.txt", b"second payload")]
        )
        starting_counter = application._MEM_ZIP_COUNTER

        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "two-inner-archives.zip",
                [
                    ("first.zip", first_bytes),
                    ("second.zip", second_bytes),
                ],
            )
            outer_key = str(outer_path)
            first_items = application.vfs_listdir(
                application.make_zip_uri(outer_key, "first.zip")
            )
            second_items = application.vfs_listdir(
                application.make_zip_uri(outer_key, "second.zip")
            )
            first_id = self._memzip_id(first_items)
            second_id = self._memzip_id(second_items)

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(
                set(application._MEM_ZIP_BYTES),
                set(application._MEM_ZIP_META),
            )
            self.assertEqual(
                set(application._MEM_ZIP_BYTES),
                {first_id, second_id},
            )
            self.assertEqual(
                application._MEM_ZIP_META[first_id],
                {"outer": outer_key, "inner": "first.zip"},
            )
            self.assertEqual(
                application._MEM_ZIP_META[second_id],
                {"outer": outer_key, "inner": "second.zip"},
            )
            self.assertEqual(application._MEM_ZIP_BYTES[first_id], first_bytes)
            self.assertEqual(
                application._MEM_ZIP_BYTES[second_id],
                second_bytes,
            )

            with zipfile.ZipFile(
                BytesIO(application._MEM_ZIP_BYTES[first_id]),
                mode="r",
            ) as first_archive:
                self.assertEqual(
                    first_archive.read("first-only.txt"),
                    b"first payload",
                )
            with zipfile.ZipFile(
                BytesIO(application._MEM_ZIP_BYTES[second_id]),
                mode="r",
            ) as second_archive:
                self.assertEqual(
                    second_archive.read("second-only.txt"),
                    b"second payload",
                )

            self.assertEqual(
                application._MEM_ZIP_COUNTER,
                starting_counter + 2,
            )

    def test_missing_memzip_registration_rejects_and_closes_cached_reader(
        self,
    ) -> None:
        inner_bytes = self._archive_bytes([("inside.txt", b"payload")])

        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "missing-registration.zip",
                [("inner.zip", inner_bytes)],
            )
            inner_items = application.vfs_listdir(
                application.make_zip_uri(
                    str(outer_path),
                    "inner.zip",
                )
            )
            memzip_id = self._memzip_id(inner_items)
            reader = self._track_reader(memzip_id)
            self.assertIsNotNone(reader.fp)

            del application._MEM_ZIP_BYTES[memzip_id]
            del application._MEM_ZIP_META[memzip_id]

            with self.assertRaises(FileNotFoundError):
                application._open_zip_cached(memzip_id)

            self.assertIsNone(reader.fp)


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
