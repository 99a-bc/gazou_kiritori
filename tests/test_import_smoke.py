"""Initial import smoke tests and checks for the shared test helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

try:
    from tests.helpers import (
        create_test_image,
        create_test_zip,
        get_qapplication,
        temporary_directory,
    )
except ModuleNotFoundError:
    from helpers import (  # type: ignore[no-redef]
        create_test_image,
        create_test_zip,
        get_qapplication,
        temporary_directory,
    )

from PIL import Image
from PyQt6 import QtWidgets


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"
IMPORT_PROBE_PREFIX = "GAZOU_KIRITORI_IMPORT_PROBE="
IMPORT_PROBE = textwrap.dedent(
    f"""
    import contextlib
    import io
    import json
    import traceback

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    imported_module = None
    import_error = None

    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        try:
            import gazou_kiritori as imported_module
        except BaseException:
            import_error = traceback.format_exc()

    qapplication_created = False
    visible_windows = []
    symbols = {{}}
    if imported_module is not None:
        from PyQt6 import QtWidgets

        application = QtWidgets.QApplication.instance()
        qapplication_created = application is not None
        if application is not None:
            visible_windows = [
                type(widget).__name__
                for widget in application.topLevelWidgets()
                if widget.isVisible()
            ]
        symbols = {{
            name: hasattr(imported_module, name)
            for name in ("CropperApp", "CropLabel", "ThumbnailListModel")
        }}

    result = {{
        "import_error": import_error,
        "qapplication_created": qapplication_created,
        "visible_windows": visible_windows,
        "symbols": symbols,
        "stdout": captured_stdout.getvalue(),
        "stderr": captured_stderr.getvalue(),
    }}
    print({IMPORT_PROBE_PREFIX!r} + json.dumps(result, ensure_ascii=False))
    """
)


class ImportSmokeTests(unittest.TestCase):
    def test_import_completes_without_starting_qt_or_showing_a_window(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-import-") as sandbox:
            environment = os.environ.copy()
            environment.update(
                {
                    "QT_QPA_PLATFORM": "offscreen",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "HF_HOME": str(sandbox / "hf-home"),
                    "HF_HUB_CACHE": str(sandbox / "hf-home" / "hub"),
                    "HUGGINGFACE_HUB_CACHE": str(sandbox / "hf-home" / "hub"),
                    "TRANSFORMERS_CACHE": str(sandbox / "transformers-cache"),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "GAZOU_BG_DEBUG": "0",
                    "GAZOU_KIRITORI_BG_DEBUG": "0",
                }
            )

            completed = subprocess.run(
                [sys.executable, "-B", "-c", IMPORT_PROBE],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"import probe failed:\n{completed.stdout}\n{completed.stderr}",
        )
        probe_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(IMPORT_PROBE_PREFIX)
        ]
        self.assertEqual(
            len(probe_lines),
            1,
            msg=f"missing import probe result:\n{completed.stdout}\n{completed.stderr}",
        )
        result = json.loads(probe_lines[0][len(IMPORT_PROBE_PREFIX) :])

        self.assertIsNone(result["import_error"], msg=result["import_error"])
        self.assertEqual(
            result["symbols"],
            {
                "CropperApp": True,
                "CropLabel": True,
                "ThumbnailListModel": True,
            },
        )
        self.assertFalse(result["qapplication_created"])
        self.assertEqual(result["visible_windows"], [])
        self.assertRegex(
            result["stdout"],
            r"(?m)^\[preload\] torch (?:OK:|preload failed:)",
        )


class QApplicationHelperTests(unittest.TestCase):
    def test_qapplication_is_offscreen_reused_and_does_not_start_exec(self) -> None:
        self.assertEqual(os.environ["QT_QPA_PLATFORM"], "offscreen")

        with mock.patch.object(QtWidgets.QApplication, "exec") as event_loop:
            application = get_qapplication()
            reused_application = get_qapplication()

        self.assertIs(application, reused_application)
        self.assertEqual(application.platformName().lower(), "offscreen")
        event_loop.assert_not_called()
        self.assertFalse(
            any(widget.isVisible() for widget in application.topLevelWidgets())
        )


class TemporaryDataHelperTests(unittest.TestCase):
    def test_image_is_created_and_reopened_only_inside_temporary_directory(
        self,
    ) -> None:
        temporary_root: Path
        with temporary_directory() as temporary_root:
            image_path = create_test_image(
                temporary_root / "images" / "sample.png",
                size=(7, 5),
                image_format="PNG",
                color=(12, 34, 56),
            )

            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as reopened:
                self.assertEqual(reopened.size, (7, 5))
                self.assertEqual(reopened.format, "PNG")
                self.assertEqual(reopened.getpixel((0, 0)), (12, 34, 56))

        self.assertFalse(temporary_root.exists())

    def test_zip_accepts_bytes_and_a_source_file_without_leaving_artifacts(
        self,
    ) -> None:
        temporary_root: Path
        with temporary_directory() as temporary_root:
            bytes_zip = create_test_zip(
                temporary_root / "archives" / "bytes.zip",
                archive_name="nested/payload.bin",
                content=b"characterization-bytes",
            )
            with zipfile.ZipFile(bytes_zip) as archive:
                self.assertEqual(
                    archive.read("nested/payload.bin"),
                    b"characterization-bytes",
                )

            source_path = temporary_root / "source.bin"
            source_path.write_bytes(b"characterization-file")
            file_zip = create_test_zip(
                temporary_root / "archives" / "file.zip",
                archive_name="copied/source.bin",
                content=source_path,
            )
            with zipfile.ZipFile(file_zip) as archive:
                self.assertEqual(
                    archive.read("copied/source.bin"),
                    b"characterization-file",
                )

        self.assertFalse(temporary_root.exists())


def tearDownModule() -> None:
    """Remove bytecode produced before this test module disabled cache writes."""
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
