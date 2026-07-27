"""Boundary and concurrency tests for the standalone memzip registry."""

from __future__ import annotations

import subprocess
import sys
import threading
import unittest
from pathlib import Path

from vfs_memzip import MemZipRegistry


sys.dont_write_bytecode = True

TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREAD_TIMEOUT = 3.0


class MemZipRegistryBoundaryTests(unittest.TestCase):
    def test_standalone_import_does_not_load_application_dependencies(
        self,
    ) -> None:
        script = """
import importlib.util
import pathlib
import sys

module_path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("standalone_vfs_memzip", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

forbidden = ("PyQt6", "PIL", "torch", "gazou_kiritori")
loaded = [
    name for name in sys.modules
    if any(name == item or name.startswith(item + ".") for item in forbidden)
]
if loaded:
    raise SystemExit("unexpected imports: " + repr(loaded))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                script,
                str(PROJECT_ROOT / "vfs_memzip.py"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=THREAD_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_initial_state_is_empty(self) -> None:
        registry = MemZipRegistry()

        self.assertEqual(registry.bytes_by_id, {})
        self.assertEqual(registry.metadata_by_id, {})
        self.assertEqual(registry.counter, 0)

    def test_same_registration_reuses_id_without_calling_loader_again(
        self,
    ) -> None:
        registry = MemZipRegistry()
        loader_calls = 0

        def loader() -> tuple[str, bytes]:
            nonlocal loader_calls
            loader_calls += 1
            return "Folder/Inner.zip", b"first"

        first_id = registry.register(
            r"relative\Outer.zip",
            r"folder\inner.zip",
            ("physical", 1),
            loader,
        )
        second_id = registry.register(
            r"relative\Outer.zip",
            "folder/inner.zip",
            ("physical", 1),
            loader,
        )

        self.assertEqual(first_id, "memzip:0")
        self.assertEqual(second_id, first_id)
        self.assertEqual(loader_calls, 1)
        self.assertEqual(registry.counter, 1)
        self.assertEqual(registry.bytes_by_id[first_id], b"first")
        self.assertEqual(
            registry.metadata_by_id[first_id],
            {
                "outer": r"relative\Outer.zip",
                "inner": "Folder/Inner.zip",
            },
        )
        self.assertEqual(
            registry.signature(first_id),
            ("memzip", id(registry.bytes_by_id[first_id]), 5),
        )

    def test_signature_change_creates_new_immutable_registration(self) -> None:
        registry = MemZipRegistry()
        old_data = b"old"
        new_data = b"new contents"

        old_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 1),
            lambda: ("Inner.zip", old_data),
        )
        old_signature = registry.signature(old_id)
        old_metadata = dict(registry.metadata_by_id[old_id])

        new_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 2),
            lambda: ("INNER.ZIP", new_data),
        )

        self.assertNotEqual(new_id, old_id)
        self.assertIs(registry.bytes_by_id[old_id], old_data)
        self.assertEqual(registry.metadata_by_id[old_id], old_metadata)
        self.assertEqual(registry.signature(old_id), old_signature)
        self.assertIs(registry.bytes_by_id[new_id], new_data)
        self.assertEqual(
            registry.metadata_by_id[new_id],
            {"outer": "outer.zip", "inner": "INNER.ZIP"},
        )
        self.assertEqual(registry.counter, 2)
        self.assertEqual(
            set(registry.bytes_by_id),
            set(registry.metadata_by_id),
        )

    def test_clear_empties_registrations_without_reusing_ids(self) -> None:
        registry = MemZipRegistry()
        old_ids = [
            registry.register(
                "outer.zip",
                "inner.zip",
                ("physical", version),
                lambda version=version: (
                    "inner.zip",
                    bytes([version]),
                ),
            )
            for version in range(2)
        ]
        counter_before_clear = registry.counter

        registry.clear()

        self.assertEqual(registry.bytes_by_id, {})
        self.assertEqual(registry.metadata_by_id, {})
        self.assertEqual(registry._registrations, {})
        self.assertEqual(registry.counter, counter_before_clear)
        for old_id in old_ids:
            with self.subTest(old_id=old_id):
                with self.assertRaises(FileNotFoundError):
                    registry.get_bytes(old_id)

        new_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 2),
            lambda: ("inner.zip", b"new"),
        )
        self.assertNotIn(new_id, old_ids)
        self.assertGreater(
            int(new_id.removeprefix("memzip:")),
            max(
                int(old_id.removeprefix("memzip:"))
                for old_id in old_ids
            ),
        )

    def test_clear_preserves_dictionary_identity_and_is_idempotent(self) -> None:
        registry = MemZipRegistry()
        bytes_reference = registry.bytes_by_id
        metadata_reference = registry.metadata_by_id
        registrations_reference = registry._registrations
        registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 1),
            lambda: ("inner.zip", b"payload"),
        )
        counter_before_clear = registry.counter

        registry.clear()
        registry.clear()

        self.assertIs(registry.bytes_by_id, bytes_reference)
        self.assertIs(registry.metadata_by_id, metadata_reference)
        self.assertIs(registry._registrations, registrations_reference)
        self.assertEqual(bytes_reference, {})
        self.assertEqual(metadata_reference, {})
        self.assertEqual(registrations_reference, {})
        self.assertEqual(registry.counter, counter_before_clear)

    def test_missing_id_raises_file_not_found(self) -> None:
        registry = MemZipRegistry()

        with self.assertRaisesRegex(
            FileNotFoundError,
            "memzip not registered: memzip:missing",
        ):
            registry.get_bytes("memzip:missing")
        with self.assertRaises(FileNotFoundError):
            registry.signature("memzip:missing")

    def test_loader_runs_outside_registry_lock(self) -> None:
        registry = MemZipRegistry()
        result: list[str] = []
        errors: list[BaseException] = []

        def outer_loader() -> tuple[str, bytes]:
            nested_id = registry.register(
                "nested.zip",
                "nested-inner.zip",
                ("physical", "nested"),
                lambda: ("nested-inner.zip", b"nested"),
            )
            result.append(nested_id)
            return "outer-inner.zip", b"outer"

        def register_outer() -> None:
            try:
                result.append(
                    registry.register(
                        "outer.zip",
                        "outer-inner.zip",
                        ("physical", "outer"),
                        outer_loader,
                    )
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=register_outer, daemon=True)
        thread.start()
        thread.join(THREAD_TIMEOUT)

        self.assertFalse(thread.is_alive(), "registration deadlocked")
        self.assertEqual(errors, [])
        self.assertEqual(set(result), {"memzip:0", "memzip:1"})

    def test_concurrent_same_registration_is_committed_once(self) -> None:
        registry = MemZipRegistry()
        worker_count = 8
        loader_barrier = threading.Barrier(worker_count)
        results: list[str] = []
        errors: list[BaseException] = []
        result_lock = threading.Lock()

        def loader() -> tuple[str, bytes]:
            loader_barrier.wait(THREAD_TIMEOUT)
            return "inner.zip", b"shared"

        def worker() -> None:
            try:
                memzip_id = registry.register(
                    "outer.zip",
                    "inner.zip",
                    ("physical", 1),
                    loader,
                )
                with result_lock:
                    results.append(memzip_id)
            except BaseException as error:
                with result_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(THREAD_TIMEOUT)

        self.assertFalse(
            any(thread.is_alive() for thread in threads),
            "concurrent registration timed out",
        )
        self.assertEqual(errors, [])
        self.assertEqual(results, ["memzip:0"] * worker_count)
        self.assertEqual(registry.counter, 1)
        self.assertEqual(set(registry.bytes_by_id), {"memzip:0"})
        self.assertEqual(
            set(registry.bytes_by_id),
            set(registry.metadata_by_id),
        )

    def test_concurrent_different_signatures_get_unique_ids(self) -> None:
        registry = MemZipRegistry()
        worker_count = 8
        loader_barrier = threading.Barrier(worker_count)
        results: list[str] = []
        errors: list[BaseException] = []
        result_lock = threading.Lock()

        def worker(signature_number: int) -> None:
            def loader() -> tuple[str, bytes]:
                loader_barrier.wait(THREAD_TIMEOUT)
                return "inner.zip", bytes([signature_number])

            try:
                memzip_id = registry.register(
                    "outer.zip",
                    "inner.zip",
                    ("physical", signature_number),
                    loader,
                )
                with result_lock:
                    results.append(memzip_id)
            except BaseException as error:
                with result_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(number,), daemon=True)
            for number in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(THREAD_TIMEOUT)

        self.assertFalse(
            any(thread.is_alive() for thread in threads),
            "concurrent registration timed out",
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), worker_count)
        self.assertEqual(registry.counter, worker_count)
        self.assertEqual(len(registry.bytes_by_id), worker_count)
        self.assertEqual(
            set(registry.bytes_by_id),
            set(registry.metadata_by_id),
        )

    def test_corrupted_compatibility_entry_is_not_reused(self) -> None:
        registry = MemZipRegistry()
        first_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 1),
            lambda: ("inner.zip", b"first"),
        )
        registry.bytes_by_id[first_id] = b"externally replaced"

        second_id = registry.register(
            "outer.zip",
            "inner.zip",
            ("physical", 1),
            lambda: ("inner.zip", b"second"),
        )

        self.assertNotEqual(second_id, first_id)
        self.assertEqual(registry.bytes_by_id[second_id], b"second")

        for missing_dictionary in ("bytes", "metadata"):
            with self.subTest(missing_dictionary=missing_dictionary):
                registry = MemZipRegistry()
                first_id = registry.register(
                    "outer.zip",
                    "inner.zip",
                    ("physical", 1),
                    lambda: ("inner.zip", b"first"),
                )
                if missing_dictionary == "bytes":
                    del registry.bytes_by_id[first_id]
                else:
                    del registry.metadata_by_id[first_id]

                second_id = registry.register(
                    "outer.zip",
                    "inner.zip",
                    ("physical", 1),
                    lambda: ("inner.zip", b"second"),
                )

                self.assertNotEqual(second_id, first_id)
                self.assertEqual(
                    set(registry.bytes_by_id),
                    set(registry.metadata_by_id),
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
