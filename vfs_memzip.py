"""Thread-safe immutable registration for ZIP archives held in memory."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Hashable


MemZipLoader = Callable[[], tuple[str, bytes]]
RegistrationKey = tuple[str, str, Hashable]


@dataclass(frozen=True)
class _Registration:
    memzip_id: str
    data: bytes
    resolved_inner: str


class MemZipRegistry:
    """Own immutable memzip bytes and their compatibility metadata."""

    def __init__(self) -> None:
        self.bytes_by_id: dict[str, bytes] = {}
        self.metadata_by_id: dict[str, dict[str, str]] = {}
        self.counter = 0
        self._registrations: dict[RegistrationKey, _Registration] = {}
        self._lock = Lock()

    @staticmethod
    def _registration_key(
        outer_path: str,
        requested_inner: str,
        outer_signature: Hashable,
    ) -> RegistrationKey:
        return (
            outer_path,
            (requested_inner or "").replace("\\", "/"),
            outer_signature,
        )

    def _find_valid_registration(
        self,
        key: RegistrationKey,
    ) -> str | None:
        registration = self._registrations.get(key)
        if registration is None:
            return None

        outer_path, _requested_inner, _outer_signature = key
        data = self.bytes_by_id.get(registration.memzip_id)
        metadata = self.metadata_by_id.get(registration.memzip_id)
        if (
            data is registration.data
            and isinstance(metadata, dict)
            and metadata.get("outer") == outer_path
            and metadata.get("inner") == registration.resolved_inner
        ):
            return registration.memzip_id

        del self._registrations[key]
        if data is None or metadata is None:
            self.bytes_by_id.pop(registration.memzip_id, None)
            self.metadata_by_id.pop(registration.memzip_id, None)
        return None

    def register(
        self,
        outer_path: str,
        requested_inner: str,
        outer_signature: Hashable,
        loader: MemZipLoader,
    ) -> str:
        """Return the immutable registration for one outer/inner/signature."""
        key = self._registration_key(
            outer_path,
            requested_inner,
            outer_signature,
        )

        with self._lock:
            existing_id = self._find_valid_registration(key)
            if existing_id is not None:
                return existing_id

        resolved_inner, data = loader()
        if not isinstance(data, bytes):
            raise TypeError("memzip loader must return bytes")

        with self._lock:
            existing_id = self._find_valid_registration(key)
            if existing_id is not None:
                return existing_id

            while True:
                memzip_id = f"memzip:{self.counter}"
                self.counter += 1
                if (
                    memzip_id not in self.bytes_by_id
                    and memzip_id not in self.metadata_by_id
                ):
                    break

            metadata = {
                "outer": outer_path,
                "inner": resolved_inner,
            }
            self.bytes_by_id[memzip_id] = data
            self.metadata_by_id[memzip_id] = metadata
            self._registrations[key] = _Registration(
                memzip_id=memzip_id,
                data=data,
                resolved_inner=resolved_inner,
            )
            return memzip_id

    def get_bytes(self, memzip_id: str) -> bytes:
        """Return registered bytes or raise the legacy missing-ID error."""
        with self._lock:
            try:
                return self.bytes_by_id[memzip_id]
            except KeyError:
                raise FileNotFoundError(
                    f"memzip not registered: {memzip_id}"
                ) from None

    def signature(self, memzip_id: str) -> tuple[str, int, int]:
        """Return the cache signature for immutable registered bytes."""
        data = self.get_bytes(memzip_id)
        return ("memzip", id(data), len(data))
