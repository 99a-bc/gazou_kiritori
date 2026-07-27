"""Small owning LRU cache for archive readers."""

from __future__ import annotations

import os
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, Hashable, Iterator, NamedTuple, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
ReaderT = TypeVar("ReaderT")


class ArchiveCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int


@dataclass
class _CacheEntry(Generic[ReaderT]):
    signature: object
    reader: ReaderT
    lease_count: int = 0
    close_when_released: bool = False
    closed: bool = False


def physical_archive_signature(path: os.PathLike[str] | str) -> tuple[int, int, int, int]:
    """Return a stable-enough signature for detecting archive replacement."""
    stat_result = os.stat(path)
    return (
        stat_result.st_mtime_ns,
        stat_result.st_size,
        stat_result.st_dev,
        stat_result.st_ino,
    )


class ArchiveReaderCache(Generic[KeyT, ReaderT]):
    """Own archive readers and close them when their cache entries expire."""

    def __init__(self, maxsize: int = 8) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be greater than zero")
        self._maxsize = maxsize
        self._entries: OrderedDict[KeyT, _CacheEntry[ReaderT]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    @staticmethod
    def _reader_is_closed(reader: ReaderT) -> bool:
        if isinstance(reader, zipfile.ZipFile):
            return reader.fp is None

        try:
            closed = getattr(reader, "closed", None)
        except Exception:
            return False
        return closed is True

    @staticmethod
    def _close_reader(reader: ReaderT) -> None:
        try:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def _schedule_close_locked(
        self,
        entry: _CacheEntry[ReaderT],
        readers_to_close: list[ReaderT],
    ) -> None:
        if entry.closed:
            return
        entry.closed = True
        if not self._reader_is_closed(entry.reader):
            readers_to_close.append(entry.reader)

    def _retire_entry_locked(
        self,
        entry: _CacheEntry[ReaderT],
        readers_to_close: list[ReaderT],
    ) -> None:
        entry.close_when_released = True
        if entry.lease_count == 0:
            self._schedule_close_locked(entry, readers_to_close)

    def _discard_entry_locked(
        self,
        key: KeyT,
        entry: _CacheEntry[ReaderT],
        readers_to_close: list[ReaderT],
    ) -> None:
        if self._entries.get(key) is entry:
            del self._entries[key]
        self._retire_entry_locked(entry, readers_to_close)

    def _get_or_open_entry_locked(
        self,
        key: KeyT,
        opener: Callable[[KeyT], ReaderT],
        signature_factory: Callable[[KeyT], object],
        readers_to_close: list[ReaderT],
    ) -> _CacheEntry[ReaderT]:
        entry = self._entries.get(key)
        try:
            signature = signature_factory(key)
        except Exception:
            if entry is not None:
                self._discard_entry_locked(key, entry, readers_to_close)
            self._misses += 1
            raise

        if entry is not None:
            if (
                entry.signature == signature
                and not self._reader_is_closed(entry.reader)
            ):
                self._entries.move_to_end(key)
                self._hits += 1
                return entry
            self._discard_entry_locked(key, entry, readers_to_close)

        self._misses += 1
        reader = opener(key)
        entry = _CacheEntry(signature=signature, reader=reader)
        self._entries[key] = entry
        self._entries.move_to_end(key)

        while len(self._entries) > self._maxsize:
            _, evicted = self._entries.popitem(last=False)
            self._retire_entry_locked(evicted, readers_to_close)

        return entry

    @classmethod
    def _close_readers(cls, readers: list[ReaderT]) -> None:
        reader_ids: set[int] = set()
        for reader in readers:
            reader_id = id(reader)
            if reader_id in reader_ids:
                continue
            reader_ids.add(reader_id)
            cls._close_reader(reader)

    def _acquire_entry(
        self,
        key: KeyT,
        opener: Callable[[KeyT], ReaderT],
        signature_factory: Callable[[KeyT], object],
        *,
        lease: bool,
    ) -> _CacheEntry[ReaderT]:
        readers_to_close: list[ReaderT] = []
        try:
            with self._lock:
                entry = self._get_or_open_entry_locked(
                    key,
                    opener,
                    signature_factory,
                    readers_to_close,
                )
                if lease:
                    entry.lease_count += 1
                return entry
        finally:
            self._close_readers(readers_to_close)

    def _release_entry(self, entry: _CacheEntry[ReaderT]) -> None:
        readers_to_close: list[ReaderT] = []
        with self._lock:
            if entry.lease_count <= 0:
                raise RuntimeError("archive reader lease released more than once")
            entry.lease_count -= 1
            if entry.lease_count == 0 and entry.close_when_released:
                self._schedule_close_locked(entry, readers_to_close)
        self._close_readers(readers_to_close)

    def get(
        self,
        key: KeyT,
        opener: Callable[[KeyT], ReaderT],
        signature_factory: Callable[[KeyT], object],
    ) -> ReaderT:
        """Return a current reader for the exact key, opening one on a miss."""
        return self._acquire_entry(
            key,
            opener,
            signature_factory,
            lease=False,
        ).reader

    @contextmanager
    def lease(
        self,
        key: KeyT,
        opener: Callable[[KeyT], ReaderT],
        signature_factory: Callable[[KeyT], object],
    ) -> Iterator[ReaderT]:
        """Yield a reader that cannot be closed by this cache until release."""
        entry = self._acquire_entry(
            key,
            opener,
            signature_factory,
            lease=True,
        )
        try:
            yield entry.reader
        finally:
            self._release_entry(entry)

    def clear(self) -> None:
        """Close all owned readers, empty the cache, and reset statistics."""
        readers_to_close: list[ReaderT] = []
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            for entry in entries:
                self._retire_entry_locked(entry, readers_to_close)
        self._close_readers(readers_to_close)

    def cache_info(self) -> ArchiveCacheInfo:
        with self._lock:
            return ArchiveCacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._maxsize,
                currsize=len(self._entries),
            )
