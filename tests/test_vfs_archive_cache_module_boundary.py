"""Boundary checks for the owning archive-reader cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import unittest
from pathlib import Path


sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBE_PREFIX = "VFS_ARCHIVE_CACHE_IMPORT_PROBE="
IMPORT_PROBE = textwrap.dedent(
    f"""
    import json
    import sys

    import vfs_archive_cache

    forbidden_roots = ("PyQt6", "PIL", "torch", "gazou_kiritori")
    loaded = {{
        root: any(name == root or name.startswith(root + ".") for name in sys.modules)
        for root in forbidden_roots
    }}
    print({PROBE_PREFIX!r} + json.dumps(loaded, sort_keys=True))
    """
)


class _TrackingReader:
    def __init__(self, key: str, signature: int) -> None:
        self.key = key
        self.signature = signature
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _ReaderHarness:
    def __init__(self) -> None:
        self.signatures: dict[str, int] = {}
        self.opened: list[_TrackingReader] = []

    def signature(self, key: str) -> int:
        return self.signatures.get(key, 0)

    def open(self, key: str) -> _TrackingReader:
        reader = _TrackingReader(key, self.signature(key))
        self.opened.append(reader)
        return reader


class VfsArchiveCacheModuleBoundaryTests(unittest.TestCase):
    def test_standalone_import_does_not_load_application_dependencies(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", IMPORT_PROBE],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"standalone import failed:\n{completed.stdout}\n{completed.stderr}",
        )
        probe_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(PROBE_PREFIX)
        ]
        self.assertEqual(
            len(probe_lines),
            1,
            msg=f"missing import probe result:\n{completed.stdout}\n{completed.stderr}",
        )
        self.assertEqual(
            json.loads(probe_lines[0][len(PROBE_PREFIX) :]),
            {
                "PIL": False,
                "PyQt6": False,
                "gazou_kiritori": False,
                "torch": False,
            },
        )

    def test_initial_cache_info(self) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache(maxsize=8)
        cache_info = cache.cache_info()

        self.assertEqual(cache_info.hits, 0)
        self.assertEqual(cache_info.misses, 0)
        self.assertEqual(cache_info.maxsize, 8)
        self.assertEqual(cache_info.currsize, 0)

    def test_eviction_defers_close_while_reader_is_leased(self) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=1)
        harness = _ReaderHarness()
        self.addCleanup(cache.clear)

        with cache.lease("a", harness.open, harness.signature) as reader_a:
            reader_b = cache.get("b", harness.open, harness.signature)

            self.assertFalse(reader_a.closed)
            self.assertFalse(reader_b.closed)
            self.assertEqual(cache.cache_info().currsize, 1)

        self.assertTrue(reader_a.closed)
        self.assertEqual(reader_a.close_calls, 1)
        self.assertFalse(reader_b.closed)
        cache.clear()
        self.assertTrue(reader_b.closed)
        self.assertEqual(reader_b.close_calls, 1)

    def test_normal_lease_release_keeps_reader_cached_and_reusable(
        self,
    ) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=2)
        harness = _ReaderHarness()
        self.addCleanup(cache.clear)

        with cache.lease("a", harness.open, harness.signature) as leased:
            self.assertFalse(leased.closed)

        self.assertFalse(leased.closed)
        self.assertIs(
            cache.get("a", harness.open, harness.signature),
            leased,
        )
        self.assertEqual(cache.cache_info().currsize, 1)

        cache.clear()
        self.assertTrue(leased.closed)
        self.assertEqual(leased.close_calls, 1)

    def test_clear_defers_close_and_resets_statistics_while_leased(self) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=2)
        harness = _ReaderHarness()
        self.addCleanup(cache.clear)

        with cache.lease("a", harness.open, harness.signature) as reader:
            self.assertIs(
                cache.get("a", harness.open, harness.signature),
                reader,
            )

            cache.clear()

            cache_info = cache.cache_info()
            self.assertEqual(cache_info.hits, 0)
            self.assertEqual(cache_info.misses, 0)
            self.assertEqual(cache_info.maxsize, 2)
            self.assertEqual(cache_info.currsize, 0)
            self.assertFalse(reader.closed)

        self.assertTrue(reader.closed)
        self.assertEqual(reader.close_calls, 1)

    def test_lease_releases_deferred_reader_when_context_raises(self) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=1)
        harness = _ReaderHarness()
        self.addCleanup(cache.clear)
        reader: _TrackingReader | None = None

        with self.assertRaisesRegex(RuntimeError, "intentional lease failure"):
            with cache.lease("a", harness.open, harness.signature) as leased:
                reader = leased
                cache.clear()
                self.assertFalse(reader.closed)
                raise RuntimeError("intentional lease failure")

        self.assertIsNotNone(reader)
        self.assertTrue(reader.closed)
        self.assertEqual(reader.close_calls, 1)
        self.assertEqual(cache.cache_info().currsize, 0)

    def test_stale_replacement_defers_old_reader_close_until_release(
        self,
    ) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=2)
        harness = _ReaderHarness()
        harness.signatures["a"] = 1
        self.addCleanup(cache.clear)

        with cache.lease("a", harness.open, harness.signature) as old_reader:
            harness.signatures["a"] = 2
            replacement = cache.get("a", harness.open, harness.signature)

            self.assertIsNot(replacement, old_reader)
            self.assertFalse(old_reader.closed)
            self.assertFalse(replacement.closed)
            self.assertEqual(cache.cache_info().currsize, 1)

        self.assertTrue(old_reader.closed)
        self.assertEqual(old_reader.close_calls, 1)
        self.assertFalse(replacement.closed)
        self.assertIs(
            cache.get("a", harness.open, harness.signature),
            replacement,
        )

    def test_concurrent_leases_survive_eviction_until_last_release(
        self,
    ) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache[str, _TrackingReader](maxsize=2)
        harness = _ReaderHarness()
        self.addCleanup(cache.clear)
        release_leases = threading.Event()
        all_leased = threading.Event()
        state_lock = threading.Lock()
        leased_readers: list[_TrackingReader] = []
        worker_errors: list[BaseException] = []
        worker_count = 4

        def worker() -> None:
            try:
                with cache.lease(
                    "shared",
                    harness.open,
                    harness.signature,
                ) as reader:
                    with state_lock:
                        leased_readers.append(reader)
                        if len(leased_readers) == worker_count:
                            all_leased.set()
                    if not release_leases.wait(5.0):
                        raise TimeoutError("timed out waiting to release leases")
            except BaseException as error:
                with state_lock:
                    worker_errors.append(error)

        threads = [
            threading.Thread(
                target=worker,
                name=f"archive-reader-lease-{index}",
            )
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        entered_in_time = False
        closed_during_eviction = True
        cache_size_during_eviction = None
        try:
            entered_in_time = all_leased.wait(5.0)
            if entered_in_time:
                cache.get("other-1", harness.open, harness.signature)
                cache.get("other-2", harness.open, harness.signature)
                closed_during_eviction = leased_readers[0].closed
                cache_size_during_eviction = cache.cache_info().currsize
        finally:
            release_leases.set()
            for thread in threads:
                thread.join(5.0)

        self.assertTrue(entered_in_time)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(worker_errors, [])
        self.assertEqual(len(leased_readers), worker_count)
        self.assertEqual(len({id(reader) for reader in leased_readers}), 1)
        self.assertFalse(closed_during_eviction)
        self.assertEqual(cache_size_during_eviction, 2)
        self.assertTrue(leased_readers[0].closed)
        self.assertEqual(leased_readers[0].close_calls, 1)


if __name__ == "__main__":
    unittest.main()
