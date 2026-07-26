"""Regression tests for archive-reader and case-fold-index lifecycles."""

from __future__ import annotations

import sys
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


sys.dont_write_bytecode = True

try:
    from tests.helpers import temporary_directory
except ModuleNotFoundError:
    from helpers import temporary_directory  # type: ignore[no-redef]

import gazou_kiritori as application


class ArchiveLifecycleRegressionTests(unittest.TestCase):
    """Isolate every cached reader and registry mutation made by these tests."""

    def setUp(self) -> None:
        self._saved_mem_zip_bytes = dict(application._MEM_ZIP_BYTES)
        self._saved_mem_zip_meta = {
            key: dict(value) for key, value in application._MEM_ZIP_META.items()
        }
        self._saved_mem_zip_counter = application._MEM_ZIP_COUNTER
        self._readers: list[zipfile.ZipFile] = []

        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

    def tearDown(self) -> None:
        self._restore_module_state()

    def _write_archive(
        self,
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

    def _open_archive(self, path: Path) -> zipfile.ZipFile:
        reader = application._open_zip_cached(str(path))
        if all(reader is not existing for existing in self._readers):
            self._readers.append(reader)
        return reader

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
            prefix="gazou-kiritori-archive-lifecycle-",
        ) as temporary_root:
            try:
                yield temporary_root
            finally:
                # Release Windows archive handles before TemporaryDirectory exits.
                self._restore_module_state()

    def test_closed_cached_reader_is_reopened_after_archive_replacement(
        self,
    ) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "replace.zip",
                [("version.txt", b"version 1")],
            )
            first_reader = self._open_archive(archive_path)
            self.assertEqual(first_reader.read("version.txt"), b"version 1")

            first_reader.close()
            self._write_archive(
                archive_path,
                [("version.txt", b"version 2")],
            )

            replacement_reader = self._open_archive(archive_path)
            self.assertEqual(
                replacement_reader.read("version.txt"),
                b"version 2",
            )

    def test_casefold_index_is_rebuilt_after_archive_replacement(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "casefold.zip",
                [("Old/Photo.PNG", b"old")],
            )
            reader = self._open_archive(archive_path)
            self.assertEqual(
                application._zip_resolve_inner(
                    str(archive_path),
                    "old/photo.png",
                ),
                "Old/Photo.PNG",
            )

            reader.close()
            self._write_archive(
                archive_path,
                [("New/Image.PNG", b"new")],
            )

            self.assertEqual(
                application._zip_resolve_inner(
                    str(archive_path),
                    "new/image.png",
                ),
                "New/Image.PNG",
            )
            self.assertEqual(
                application._zip_resolve_inner(
                    str(archive_path),
                    "old/photo.png",
                ),
                "old/photo.png",
            )

    def test_casefold_index_cache_reuses_unchanged_archive(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "casefold-cache.zip",
                [("Folder/Image.PNG", b"payload")],
            )

            first_index = application._zip_index_lower(str(archive_path))
            second_index = application._zip_index_lower(str(archive_path))

            self.assertEqual(second_index, first_index)
            cache_info = application._zip_index_lower.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.maxsize, 32)
            self.assertEqual(cache_info.currsize, 1)

    def test_casefold_index_cache_clear_resets_statistics(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "casefold-cache-clear.zip",
                [("Folder/Image.PNG", b"payload")],
            )
            application._zip_index_lower(str(archive_path))

            application._zip_index_lower.cache_clear()

            cache_info = application._zip_index_lower.cache_info()
            self.assertEqual(cache_info.hits, 0)
            self.assertEqual(cache_info.misses, 0)
            self.assertEqual(cache_info.maxsize, 32)
            self.assertEqual(cache_info.currsize, 0)

    def test_lru_eviction_closes_the_evicted_zip_reader(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_paths = [
                self._write_archive(
                    temporary_root / f"archive-{index}.zip",
                    [(f"entry-{index}.txt", str(index).encode("ascii"))],
                )
                for index in range(9)
            ]

            first_reader = self._open_archive(archive_paths[0])
            for archive_path in archive_paths[1:]:
                self._open_archive(archive_path)

            cache_after_ninth_open = application._open_zip_cached.cache_info()
            self.assertEqual(cache_after_ninth_open.maxsize, 8)
            self.assertEqual(cache_after_ninth_open.currsize, 8)

            reopened_first = self._open_archive(archive_paths[0])
            cache_after_reopen = application._open_zip_cached.cache_info()
            self.assertEqual(
                cache_after_reopen.misses,
                cache_after_ninth_open.misses + 1,
            )
            self.assertIsNot(reopened_first, first_reader)

            self.assertIsNone(first_reader.fp)

    def test_cache_clear_closes_all_cached_readers(self) -> None:
        with self._archive_workspace() as temporary_root:
            archive_paths = [
                self._write_archive(
                    temporary_root / f"clear-{index}.zip",
                    [(f"entry-{index}.txt", str(index).encode("ascii"))],
                )
                for index in range(3)
            ]
            readers = [
                self._open_archive(archive_path)
                for archive_path in archive_paths
            ]
            self.assertTrue(all(reader.fp is not None for reader in readers))

            application._open_zip_cached.cache_clear()

            self.assertTrue(all(reader.fp is None for reader in readers))
            cache_info = application._open_zip_cached.cache_info()
            self.assertEqual(cache_info.currsize, 0)
            self.assertEqual(cache_info.hits, 0)
            self.assertEqual(cache_info.misses, 0)

    def test_workspace_cleanup_releases_handles_and_restores_module_state(
        self,
    ) -> None:
        temporary_root: Path
        reader: zipfile.ZipFile
        with self._archive_workspace() as temporary_root:
            archive_path = self._write_archive(
                temporary_root / "release.zip",
                [("Folder/Image.PNG", b"payload")],
            )
            reader = self._open_archive(archive_path)
            self.assertEqual(
                application._zip_resolve_inner(
                    str(archive_path),
                    "folder/image.png",
                ),
                "Folder/Image.PNG",
            )

        self.assertFalse(temporary_root.exists())
        self.assertIsNone(reader.fp)
        self.assertEqual(application._open_zip_cached.cache_info().currsize, 0)
        self.assertEqual(application._zip_index_lower.cache_info().currsize, 0)
        self.assertEqual(
            application._MEM_ZIP_BYTES,
            self._saved_mem_zip_bytes,
        )
        self.assertEqual(
            application._MEM_ZIP_META,
            self._saved_mem_zip_meta,
        )
        self.assertEqual(
            application._MEM_ZIP_COUNTER,
            self._saved_mem_zip_counter,
        )


if __name__ == "__main__":
    unittest.main()
