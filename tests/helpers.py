"""Small, side-effect-conscious helpers for characterization tests."""

from __future__ import annotations

import atexit
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from PIL import Image


_qt_settings_directory: tempfile.TemporaryDirectory[str] | None = None
_qapplication: Any = None


def _cleanup_qt_settings_directory() -> None:
    global _qt_settings_directory
    if _qt_settings_directory is not None:
        _qt_settings_directory.cleanup()
        _qt_settings_directory = None


atexit.register(_cleanup_qt_settings_directory)


def get_qapplication() -> Any:
    """Return a reusable offscreen QApplication without starting its event loop."""
    global _qapplication, _qt_settings_directory

    previous_platform = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    try:
        # Import Qt only when a test explicitly asks for a QApplication. This
        # preserves the application's required Torch-before-PyQt6 import order.
        from PyQt6 import QtCore, QtWidgets

        if _qt_settings_directory is None:
            _qt_settings_directory = tempfile.TemporaryDirectory(
                prefix="gazou-kiritori-qt-settings-",
            )

        settings_path = _qt_settings_directory.name
        QtCore.QStandardPaths.setTestModeEnabled(True)
        QtCore.QCoreApplication.setOrganizationName("gazou-kiritori-tests")
        QtCore.QCoreApplication.setApplicationName("characterization-tests")
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        for scope in (
            QtCore.QSettings.Scope.UserScope,
            QtCore.QSettings.Scope.SystemScope,
        ):
            QtCore.QSettings.setPath(
                QtCore.QSettings.Format.IniFormat,
                scope,
                settings_path,
            )

        existing = QtWidgets.QApplication.instance()
        if existing is not None:
            _qapplication = existing
            return existing

        _qapplication = QtWidgets.QApplication(["gazou-kiritori-tests"])
        _qapplication.setQuitOnLastWindowClosed(False)
        return _qapplication
    finally:
        if previous_platform is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = previous_platform


@contextmanager
def temporary_directory(
    prefix: str = "gazou-kiritori-test-",
) -> Iterator[Path]:
    """Yield a pathlib path that is removed when the context exits."""
    with tempfile.TemporaryDirectory(prefix=prefix) as directory:
        yield Path(directory)


def create_test_image(
    destination: str | os.PathLike[str],
    *,
    size: tuple[int, int] = (8, 6),
    image_format: str = "PNG",
    color: str | int | tuple[int, ...] = (32, 96, 160),
    mode: str = "RGB",
) -> Path:
    """Create a small Pillow image with caller-selected basic properties."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, size, color)
    try:
        image.save(path, format=image_format)
    finally:
        image.close()
    return path


def create_test_zip(
    destination: str | os.PathLike[str],
    *,
    archive_name: str,
    content: bytes | bytearray | memoryview | str | os.PathLike[str],
) -> Path:
    """Create a ZIP containing bytes or the contents of one source file."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_name = str(PurePosixPath(archive_name.replace("\\", "/")))
    if (
        not normalized_name
        or normalized_name == "."
        or PurePosixPath(normalized_name).is_absolute()
        or normalized_name == ".."
        or normalized_name.startswith("../")
    ):
        raise ValueError("archive_name must stay inside the ZIP root")

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if isinstance(content, (bytes, bytearray, memoryview)):
            archive.writestr(normalized_name, bytes(content))
        else:
            archive.write(Path(content), arcname=normalized_name)
    return path
