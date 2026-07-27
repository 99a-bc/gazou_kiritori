"""Regression coverage for archive cover-overlay generation and reader eviction."""

from __future__ import annotations

import sys
import threading
import unittest
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator
from unittest import mock


sys.dont_write_bytecode = True

try:
    from tests.helpers import create_test_zip, temporary_directory
except ModuleNotFoundError:
    from helpers import (  # type: ignore[no-redef]
        create_test_zip,
        temporary_directory,
    )

import gazou_kiritori as application
from PIL import Image


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"
EVENT_TIMEOUT_SECONDS = 5.0


def _png_bytes(
    color: tuple[int, int, int] = (24, 96, 168),
    *,
    size: tuple[int, int] = (12, 8),
) -> bytes:
    output = BytesIO()
    image = Image.new("RGB", size, color)
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    return output.getvalue()


class _SignalRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[int, str, bytes, int]] = []

    def emit(self, row: int, path: str, png: bytes, generation: int) -> None:
        self.events.append((row, path, png, generation))


class _OverlayModel:
    def __init__(self, generation: int) -> None:
        self._gen = generation
        self.thumb_size = (80, 120)
        self.dirOverlayReady = _SignalRecorder()


@dataclass(frozen=True)
class _EvictionOutcome:
    acquired_in_time: bool
    worker_alive: bool
    worker_errors: tuple[BaseException, ...]
    held_reader_count: int
    cache_maxsize: int | None
    cache_currsize: int | None
    target_reader_closed_during_eviction: bool
    target_reader_closed_after_release: bool
    target_open_count: int
    events: tuple[tuple[int, str, bytes, int], ...]
    target_path: str
    row: int
    generation: int


class ArchiveCoverThumbnailRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._clear_archive_state()

    def tearDown(self) -> None:
        self._clear_archive_state()
        self.assertEqual(application._open_zip_cached.cache_info().currsize, 0)
        self.assertEqual(application._zip_index_lower.cache_info().currsize, 0)

    @staticmethod
    def _clear_archive_state() -> None:
        application._zip_index_lower.cache_clear()
        application._open_zip_cached.cache_clear()

    @contextmanager
    def _archive_workspace(self) -> Iterator[Path]:
        with temporary_directory(
            prefix="gazou-kiritori-archive-cover-",
        ) as temporary_root:
            try:
                yield temporary_root
            finally:
                # Release cached readers before Windows removes the ZIP files.
                self._clear_archive_state()

    def test_archive_overlay_emits_decodable_png_with_expected_identity(
        self,
    ) -> None:
        row = 7
        generation = 4107
        with self._archive_workspace() as temporary_root:
            archive_path = create_test_zip(
                temporary_root / "cover.zip",
                archive_name="Images/Cover.PNG",
                content=_png_bytes(),
            )
            path = str(archive_path)
            model = _OverlayModel(generation)

            application._DirOverlayTask(
                model,
                row,
                path,
                generation,
            ).run()

            self.assertEqual(len(model.dirOverlayReady.events), 1)
            emitted_row, emitted_path, png, emitted_generation = (
                model.dirOverlayReady.events[0]
            )
            self.assertEqual(emitted_row, row)
            self.assertEqual(emitted_path, path)
            self.assertEqual(emitted_generation, generation)
            self.assertTrue(png)

            with Image.open(BytesIO(png)) as generated:
                generated.load()
                self.assertEqual(generated.format, "PNG")
                self.assertGreater(generated.width, 0)
                self.assertGreater(generated.height, 0)

    def _exercise_active_reader_eviction(self) -> _EvictionOutcome:
        row = 3
        generation = 9203
        reader_acquired = threading.Event()
        release_reader = threading.Event()
        held_readers: list[zipfile.ZipFile] = []
        worker_errors: list[BaseException] = []
        target_open_count = 0

        with self._archive_workspace() as temporary_root:
            archive_paths = [
                create_test_zip(
                    temporary_root / f"archive-{index}.zip",
                    archive_name=(
                        "Cover.PNG" if index == 0 else f"entry-{index}.txt"
                    ),
                    content=(
                        _png_bytes((160, 48, 24))
                        if index == 0
                        else str(index).encode("ascii")
                    ),
                )
                for index in range(9)
            ]
            target_path = str(archive_paths[0])
            model = _OverlayModel(generation)
            task = application._DirOverlayTask(
                model,
                row,
                target_path,
                generation,
            )
            original_open = application._open_zip_cached
            original_lease = application._lease_zip_cached

            @contextmanager
            def controlled_lease(path: str):
                nonlocal target_open_count
                with original_lease(path) as reader:
                    if path == target_path:
                        target_open_count += 1
                        # First lease lists the archive. The second protects the
                        # reader retained immediately before image open.
                        if target_open_count == 2:
                            held_readers.append(reader)
                            reader_acquired.set()
                            if not release_reader.wait(EVENT_TIMEOUT_SECONDS):
                                raise TimeoutError(
                                    "timed out waiting to resume overlay reader"
                                )
                    yield reader

            def run_task() -> None:
                try:
                    task.run()
                except BaseException as error:
                    worker_errors.append(error)

            worker = threading.Thread(
                target=run_task,
                name="archive-cover-overlay-race",
            )
            acquired_in_time = False
            cache_during_eviction = None
            target_reader_closed_during_eviction = True

            with mock.patch.object(
                application,
                "_lease_zip_cached",
                side_effect=controlled_lease,
            ):
                worker.start()
                try:
                    acquired_in_time = reader_acquired.wait(
                        EVENT_TIMEOUT_SECONDS
                    )
                    if acquired_in_time:
                        for archive_path in archive_paths[1:]:
                            original_open(str(archive_path))
                        cache_during_eviction = original_open.cache_info()
                        target_reader_closed_during_eviction = (
                            held_readers[0].fp is None
                        )
                finally:
                    release_reader.set()
                    worker.join(EVENT_TIMEOUT_SECONDS)

            target_reader_closed_after_release = (
                len(held_readers) == 1 and held_readers[0].fp is None
            )
            return _EvictionOutcome(
                acquired_in_time=acquired_in_time,
                worker_alive=worker.is_alive(),
                worker_errors=tuple(worker_errors),
                held_reader_count=len(held_readers),
                cache_maxsize=(
                    cache_during_eviction.maxsize
                    if cache_during_eviction is not None
                    else None
                ),
                cache_currsize=(
                    cache_during_eviction.currsize
                    if cache_during_eviction is not None
                    else None
                ),
                target_reader_closed_during_eviction=(
                    target_reader_closed_during_eviction
                ),
                target_reader_closed_after_release=(
                    target_reader_closed_after_release
                ),
                target_open_count=target_open_count,
                events=tuple(model.dirOverlayReady.events),
                target_path=target_path,
                row=row,
                generation=generation,
            )

    def test_lru_eviction_defers_close_and_overlay_still_emits(
        self,
    ) -> None:
        outcome = self._exercise_active_reader_eviction()

        self.assertTrue(outcome.acquired_in_time)
        self.assertFalse(outcome.worker_alive)
        self.assertEqual(outcome.worker_errors, ())
        self.assertEqual(outcome.held_reader_count, 1)
        self.assertEqual(outcome.cache_maxsize, 8)
        self.assertEqual(outcome.cache_currsize, 8)
        self.assertFalse(outcome.target_reader_closed_during_eviction)
        self.assertTrue(outcome.target_reader_closed_after_release)
        self.assertGreaterEqual(outcome.target_open_count, 3)
        self.assertEqual(len(outcome.events), 1)

    def test_active_overlay_generation_survives_unrelated_lru_eviction(
        self,
    ) -> None:
        outcome = self._exercise_active_reader_eviction()

        self.assertTrue(outcome.acquired_in_time)
        self.assertFalse(outcome.worker_alive)
        self.assertEqual(outcome.worker_errors, ())
        self.assertFalse(outcome.target_reader_closed_during_eviction)
        self.assertTrue(outcome.target_reader_closed_after_release)
        self.assertEqual(len(outcome.events), 1)
        emitted_row, emitted_path, png, emitted_generation = outcome.events[0]
        self.assertEqual(emitted_row, outcome.row)
        self.assertEqual(emitted_path, outcome.target_path)
        self.assertEqual(emitted_generation, outcome.generation)
        self.assertTrue(png)
        with Image.open(BytesIO(png)) as generated:
            generated.load()
            self.assertEqual(generated.format, "PNG")


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
