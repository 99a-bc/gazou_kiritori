"""Characterize memzip retention before application cleanup is implemented."""

from __future__ import annotations

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

import gazou_kiritori as application
from vfs_memzip import MemZipRegistry


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


class MemZipRegistryCleanupRegressionTests(unittest.TestCase):
    def test_signature_changes_retain_every_immutable_registration(self) -> None:
        registry = MemZipRegistry()
        payloads = [
            bytes([version]) * (version + 2)
            for version in range(1, 4)
        ]
        memzip_ids: list[str] = []
        counter_values: list[int] = []

        for version, payload in enumerate(payloads, start=1):
            memzip_ids.append(
                registry.register(
                    "outer.zip",
                    "folder/inner.zip",
                    ("physical", version),
                    lambda version=version, payload=payload: (
                        f"Folder/Inner-{version}.zip",
                        payload,
                    ),
                )
            )
            counter_values.append(registry.counter)

        self.assertEqual(len(set(memzip_ids)), len(payloads))
        self.assertEqual(counter_values, [1, 2, 3])
        self.assertEqual(len(registry.bytes_by_id), len(payloads))
        self.assertEqual(len(registry.metadata_by_id), len(payloads))
        self.assertEqual(len(registry._registrations), len(payloads))

        for version, (memzip_id, payload) in enumerate(
            zip(memzip_ids, payloads),
            start=1,
        ):
            self.assertIs(registry.get_bytes(memzip_id), payload)
            self.assertEqual(
                registry.metadata_by_id[memzip_id],
                {
                    "outer": "outer.zip",
                    "inner": f"Folder/Inner-{version}.zip",
                },
            )

    def test_same_registration_reuse_does_not_grow_registry(self) -> None:
        registry = MemZipRegistry()
        loader_calls = 0

        def loader() -> tuple[str, bytes]:
            nonlocal loader_calls
            loader_calls += 1
            return "Folder/Inner.zip", b"only payload"

        memzip_ids = [
            registry.register(
                "outer.zip",
                "folder/inner.zip",
                ("physical", 1),
                loader,
            )
            for _ in range(3)
        ]

        self.assertEqual(memzip_ids, [memzip_ids[0]] * 3)
        self.assertEqual(loader_calls, 1)
        self.assertEqual(len(registry.bytes_by_id), 1)
        self.assertEqual(len(registry.metadata_by_id), 1)
        self.assertEqual(len(registry._registrations), 1)
        self.assertEqual(registry.counter, 1)

    def test_clear_discards_registrations_without_reusing_ids(self) -> None:
        registry = MemZipRegistry()
        old_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 1),
            lambda: ("inner.zip", b"old"),
        )
        counter_before_clear = registry.counter

        registry.clear()

        self.assertEqual(registry.bytes_by_id, {})
        self.assertEqual(registry.metadata_by_id, {})
        self.assertEqual(registry._registrations, {})
        self.assertEqual(registry.counter, counter_before_clear)
        with self.assertRaises(FileNotFoundError):
            registry.get_bytes(old_id)

        new_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 2),
            lambda: ("inner.zip", b"new"),
        )
        self.assertNotEqual(new_id, old_id)
        self.assertGreater(
            int(new_id.removeprefix("memzip:")),
            int(old_id.removeprefix("memzip:")),
        )
        self.assertEqual(registry.counter, counter_before_clear + 1)


class ApplicationMemZipRetentionRegressionTests(unittest.TestCase):
    """Isolate application caches and every mutable memzip registry field."""

    def setUp(self) -> None:
        registry = application._MEM_ZIP_REGISTRY
        with application._MEM_ZIP_COMPAT_LOCK:
            with registry._lock:
                self._saved_memzip_bytes = dict(registry.bytes_by_id)
                self._saved_memzip_metadata = {
                    key: dict(value)
                    for key, value in registry.metadata_by_id.items()
                }
                self._saved_registrations = dict(registry._registrations)
                self._saved_registry_counter = registry.counter
                self._saved_compat_counter = application._MEM_ZIP_COUNTER

                registry.bytes_by_id.clear()
                registry.metadata_by_id.clear()
                registry._registrations.clear()
                registry.counter = 0
                application._MEM_ZIP_COUNTER = 0

        self._saved_image_cache = OrderedDict(application._IMG_CACHE)
        self._clear_archive_caches()

    def tearDown(self) -> None:
        self._restore_application_state()

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

    @staticmethod
    def _clear_archive_caches() -> None:
        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

    def _restore_application_state(self) -> None:
        self._clear_archive_caches()

        for key, item in tuple(application._IMG_CACHE.items()):
            if (
                key not in self._saved_image_cache
                or item is not self._saved_image_cache[key]
            ):
                image = item.get("img") if isinstance(item, dict) else None
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
        application._IMG_CACHE.clear()
        application._IMG_CACHE.update(self._saved_image_cache)

        registry = application._MEM_ZIP_REGISTRY
        with application._MEM_ZIP_COMPAT_LOCK:
            with registry._lock:
                registry.bytes_by_id.clear()
                registry.bytes_by_id.update(self._saved_memzip_bytes)
                registry.metadata_by_id.clear()
                registry.metadata_by_id.update(
                    {
                        key: dict(value)
                        for key, value in self._saved_memzip_metadata.items()
                    }
                )
                registry._registrations.clear()
                registry._registrations.update(self._saved_registrations)
                registry.counter = self._saved_registry_counter
                application._MEM_ZIP_COUNTER = self._saved_compat_counter

    @contextmanager
    def _archive_workspace(self) -> Iterator[Path]:
        with temporary_directory(
            prefix="gazou-kiritori-memzip-cleanup-",
        ) as temporary_root:
            try:
                yield temporary_root
            finally:
                # Release Windows archive handles before deleting the ZIPs.
                self._clear_archive_caches()

    def test_cache_clear_retains_memzip_registration(self) -> None:
        inner_bytes = self._archive_bytes([("inside.txt", b"retained")])

        with self._archive_workspace() as temporary_root:
            outer_path = self._write_archive(
                temporary_root / "outer.zip",
                [("inner.zip", inner_bytes)],
            )
            inner_items = application.vfs_listdir(
                application.make_zip_uri(str(outer_path), "inner.zip")
            )
            self.assertEqual(
                application.open_bytes_any(inner_items[0]["uri"]),
                b"retained",
            )

            memzip_id, _ = application.parse_zip_uri(inner_items[0]["uri"])
            registered_bytes = application._MEM_ZIP_BYTES[memzip_id]
            registered_metadata = dict(application._MEM_ZIP_META[memzip_id])

            self.assertGreater(
                application._open_zip_cached.cache_info().currsize,
                0,
            )
            self.assertGreater(
                application._zip_index_lower.cache_info().currsize,
                0,
            )

            application._zip_index_lower.cache_clear()
            application._open_zip_cached.cache_clear()

            self.assertEqual(
                application._open_zip_cached.cache_info().currsize,
                0,
            )
            self.assertEqual(
                application._zip_index_lower.cache_info().currsize,
                0,
            )
            self.assertIs(
                application._MEM_ZIP_BYTES[memzip_id],
                registered_bytes,
            )
            self.assertEqual(
                application._MEM_ZIP_META[memzip_id],
                registered_metadata,
            )
            self.assertIs(
                application._MEM_ZIP_REGISTRY.get_bytes(memzip_id),
                registered_bytes,
            )


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
