"""Tests for the centralized application shutdown cleanup."""

from __future__ import annotations

import ast
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock


sys.dont_write_bytecode = True

import gazou_kiritori as application


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"


class _FakeThreadPool:
    def __init__(
        self,
        *,
        wait_result: bool | None = True,
        events: list[str] | None = None,
    ) -> None:
        self.wait_result = wait_result
        self.events = events
        self.clear_calls = 0
        self.wait_timeouts: list[int] = []

    def clear(self) -> None:
        self.clear_calls += 1
        if self.events is not None:
            self.events.append("pool.clear")

    def waitForDone(self, timeout_ms: int) -> bool | None:
        self.wait_timeouts.append(timeout_ms)
        if self.events is not None:
            self.events.append("pool.waitForDone")
        return self.wait_result


class _FakeImage:
    def __init__(
        self,
        *,
        label: str = "image.close",
        events: list[str] | None = None,
        raise_on_close: bool = False,
    ) -> None:
        self.label = label
        self.events = events
        self.raise_on_close = raise_on_close
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.events is not None:
            self.events.append(self.label)
        if self.raise_on_close:
            raise RuntimeError(f"{self.label} failed")


class _FakeSignal:
    def __init__(self) -> None:
        self.connections: list[object] = []

    def connect(self, callback) -> None:
        self.connections.append(callback)


class _FakeApplication:
    def __init__(self) -> None:
        self.aboutToQuit = _FakeSignal()


class _IsolatedApplicationCleanupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_image_cache = OrderedDict(application._IMG_CACHE)
        self._saved_index_cache_info = (
            application._zip_index_lower.cache_info()
        )
        self._saved_archive_cache_info = (
            application._open_zip_cached.cache_info()
        )

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

    def tearDown(self) -> None:
        try:
            self.assertEqual(
                application._zip_index_lower.cache_info(),
                self._saved_index_cache_info,
            )
            self.assertEqual(
                application._open_zip_cached.cache_info(),
                self._saved_archive_cache_info,
            )
        finally:
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
                    registry._registrations.update(
                        self._saved_registrations
                    )
                    registry.counter = self._saved_registry_counter
                    application._MEM_ZIP_COUNTER = self._saved_compat_counter


class ApplicationResourceCleanupTests(_IsolatedApplicationCleanupTestCase):
    def test_cleanup_waits_for_workers_then_releases_resources_in_order(
        self,
    ) -> None:
        events: list[str] = []
        pool = _FakeThreadPool(events=events)
        image = _FakeImage(events=events)
        application._IMG_CACHE.clear()
        application._IMG_CACHE["image"] = {"sig": "test", "img": image}

        with (
            mock.patch.object(
                application._zip_index_lower,
                "cache_clear",
                new=lambda: events.append("index.clear"),
            ),
            mock.patch.object(
                application._open_zip_cached,
                "cache_clear",
                new=lambda: events.append("archive.clear"),
            ),
            mock.patch.object(
                application._MEM_ZIP_REGISTRY,
                "clear",
                new=lambda: events.append("memzip.clear"),
            ),
        ):
            result = application._cleanup_application_resources(pool)

        self.assertTrue(result)
        self.assertEqual(
            events,
            [
                "pool.clear",
                "pool.waitForDone",
                "index.clear",
                "archive.clear",
                "memzip.clear",
                "image.close",
            ],
        )
        self.assertEqual(
            pool.wait_timeouts,
            [application._SHUTDOWN_WORKER_WAIT_MS],
        )
        self.assertEqual(application._IMG_CACHE, {})

    def test_worker_timeout_retains_every_resource_cache(self) -> None:
        pool = _FakeThreadPool(wait_result=False)
        image = _FakeImage()
        application._IMG_CACHE.clear()
        application._IMG_CACHE["image"] = {"sig": "test", "img": image}
        index_clear = mock.Mock()
        archive_clear = mock.Mock()
        memzip_clear = mock.Mock()

        with (
            mock.patch.object(
                application._zip_index_lower,
                "cache_clear",
                new=index_clear,
            ),
            mock.patch.object(
                application._open_zip_cached,
                "cache_clear",
                new=archive_clear,
            ),
            mock.patch.object(
                application._MEM_ZIP_REGISTRY,
                "clear",
                new=memzip_clear,
            ),
        ):
            result = application._cleanup_application_resources(pool)

        self.assertFalse(result)
        self.assertEqual(pool.clear_calls, 1)
        self.assertEqual(
            pool.wait_timeouts,
            [application._SHUTDOWN_WORKER_WAIT_MS],
        )
        index_clear.assert_not_called()
        archive_clear.assert_not_called()
        memzip_clear.assert_not_called()
        self.assertEqual(image.close_calls, 0)
        self.assertEqual(
            tuple(application._IMG_CACHE),
            ("image",),
        )

    def test_image_cleanup_deduplicates_continues_and_is_idempotent(
        self,
    ) -> None:
        pool = _FakeThreadPool()
        shared_image = _FakeImage(label="shared")
        failing_image = _FakeImage(
            label="failing",
            raise_on_close=True,
        )
        remaining_image = _FakeImage(label="remaining")
        application._IMG_CACHE.clear()
        application._IMG_CACHE.update(
            {
                "shared-one": {"sig": 1, "img": shared_image},
                "shared-two": {"sig": 2, "img": shared_image},
                "failing": {"sig": 3, "img": failing_image},
                "unexpected": object(),
                "remaining": {"sig": 4, "img": remaining_image},
            }
        )
        registry_counter = application._MEM_ZIP_REGISTRY.counter
        compatibility_counter = application._MEM_ZIP_COUNTER

        with (
            mock.patch.object(
                application._zip_index_lower,
                "cache_clear",
                new=mock.Mock(),
            ),
            mock.patch.object(
                application._open_zip_cached,
                "cache_clear",
                new=mock.Mock(),
            ),
        ):
            first_result = application._cleanup_application_resources(pool)
            second_result = application._cleanup_application_resources(pool)

        self.assertTrue(first_result)
        self.assertTrue(second_result)
        self.assertEqual(shared_image.close_calls, 1)
        self.assertEqual(failing_image.close_calls, 1)
        self.assertEqual(remaining_image.close_calls, 1)
        self.assertEqual(application._IMG_CACHE, {})
        self.assertEqual(
            application._MEM_ZIP_REGISTRY.counter,
            registry_counter,
        )
        self.assertEqual(
            application._MEM_ZIP_COUNTER,
            compatibility_counter,
        )


class ApplicationCleanupHookTests(unittest.TestCase):
    def test_install_connects_cleanup_once_to_about_to_quit(self) -> None:
        app = _FakeApplication()

        application._install_application_cleanup(app)

        self.assertEqual(
            app.aboutToQuit.connections,
            [application._cleanup_application_resources],
        )

    def test_main_startup_installs_cleanup_after_qapplication_creation(
        self,
    ) -> None:
        source = Path(application.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        main_guards = [
            node
            for node in module.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        self.assertEqual(len(main_guards), 1)

        calls = [
            node
            for node in ast.walk(main_guards[0])
            if isinstance(node, ast.Call)
        ]
        qapplication_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "QtWidgets"
            and node.func.attr == "QApplication"
        ]
        install_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "_install_application_cleanup"
        ]
        event_loop_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "app"
            and node.func.attr == "exec"
        ]

        self.assertEqual(len(qapplication_calls), 1)
        self.assertEqual(len(install_calls), 1)
        self.assertEqual(len(event_loop_calls), 1)
        self.assertEqual(len(install_calls[0].args), 1)
        self.assertIsInstance(install_calls[0].args[0], ast.Name)
        self.assertEqual(install_calls[0].args[0].id, "app")
        self.assertLess(qapplication_calls[0].lineno, install_calls[0].lineno)
        self.assertLess(install_calls[0].lineno, event_loop_calls[0].lineno)


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
