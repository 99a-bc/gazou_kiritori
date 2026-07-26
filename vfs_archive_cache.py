"""Small owning LRU cache for archive readers."""

from __future__ import annotations

import os
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, Hashable, NamedTuple, TypeVar


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

    def _discard_stale(self, key: KeyT, entry: _CacheEntry[ReaderT]) -> None:
        self._close_reader(entry.reader)
        if self._entries.get(key) is entry:
            del self._entries[key]

    def get(
        self,
        key: KeyT,
        opener: Callable[[KeyT], ReaderT],
        signature_factory: Callable[[KeyT], object],
    ) -> ReaderT:
        """Return a current reader for the exact key, opening one on a miss."""
        with self._lock:
            entry = self._entries.get(key)
            try:
                signature = signature_factory(key)
            except Exception:
                if entry is not None:
                    self._discard_stale(key, entry)
                self._misses += 1
                raise

            if entry is not None:
                if (
                    entry.signature == signature
                    and not self._reader_is_closed(entry.reader)
                ):
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return entry.reader
                self._discard_stale(key, entry)

            self._misses += 1
            reader = opener(key)
            self._entries[key] = _CacheEntry(signature=signature, reader=reader)
            self._entries.move_to_end(key)

            while len(self._entries) > self._maxsize:
                _, evicted = self._entries.popitem(last=False)
                self._close_reader(evicted.reader)

            return reader

    def clear(self) -> None:
        """Close all owned readers, empty the cache, and reset statistics."""
        with self._lock:
            readers: list[ReaderT] = []
            reader_ids: set[int] = set()
            for entry in self._entries.values():
                reader_id = id(entry.reader)
                if reader_id not in reader_ids:
                    reader_ids.add(reader_id)
                    readers.append(entry.reader)

            self._entries.clear()
            self._hits = 0
            self._misses = 0

            for reader in readers:
                self._close_reader(reader)

    def cache_info(self) -> ArchiveCacheInfo:
        with self._lock:
            return ArchiveCacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._maxsize,
                currsize=len(self._entries),
            )
