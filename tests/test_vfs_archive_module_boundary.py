"""Boundary checks for physical archive adapters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path


sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBE_PREFIX = "VFS_ARCHIVE_IMPORT_PROBE="
IMPORT_PROBE = textwrap.dedent(
    f"""
    import json
    import sys

    import vfs_archives

    forbidden_roots = ("PyQt6", "PIL", "torch", "gazou_kiritori")
    loaded = {{
        root: any(name == root or name.startswith(root + ".") for name in sys.modules)
        for root in forbidden_roots
    }}
    print({PROBE_PREFIX!r} + json.dumps(loaded, sort_keys=True))
    """
)


def _create_zip(directory: Path) -> Path:
    archive_path = directory / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/payload.bin", b"archive-boundary")
    return archive_path


class VfsArchiveModuleBoundaryTests(unittest.TestCase):
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

    def test_application_reexports_archive_compatibility_types(self) -> None:
        import gazou_kiritori
        import vfs_archives

        for name in (
            "PasswordProtectedArchiveError",
            "_SevenZipInfoCompat",
            "SevenZipCompat",
        ):
            with self.subTest(name=name):
                self.assertIs(
                    getattr(gazou_kiritori, name),
                    getattr(vfs_archives, name),
                )

    def test_open_physical_archive_reads_zip_contents(self) -> None:
        import vfs_archives

        with tempfile.TemporaryDirectory(
            prefix="gazou-kiritori-archive-boundary-",
        ) as directory:
            archive_path = _create_zip(Path(directory))
            archive = vfs_archives.open_physical_archive(str(archive_path))
            try:
                self.assertEqual(archive.namelist(), ["nested/payload.bin"])
                self.assertEqual(
                    archive.read("nested/payload.bin"),
                    b"archive-boundary",
                )
            finally:
                archive.close()

    def test_open_physical_archive_does_not_cache_objects(self) -> None:
        import vfs_archives

        self.assertFalse(hasattr(vfs_archives.open_physical_archive, "cache_info"))
        with tempfile.TemporaryDirectory(
            prefix="gazou-kiritori-archive-no-cache-",
        ) as directory:
            archive_path = _create_zip(Path(directory))
            first = vfs_archives.open_physical_archive(str(archive_path))
            second = vfs_archives.open_physical_archive(str(archive_path))
            try:
                self.assertIsNot(first, second)
            finally:
                first.close()
                second.close()

    def test_application_archive_cache_still_reuses_same_object(self) -> None:
        import gazou_kiritori

        gazou_kiritori._open_zip_cached.cache_clear()
        with tempfile.TemporaryDirectory(
            prefix="gazou-kiritori-archive-cache-",
        ) as directory:
            archive_path = _create_zip(Path(directory))
            first = gazou_kiritori._open_zip_cached(str(archive_path))
            repeated = gazou_kiritori._open_zip_cached(str(archive_path))
            try:
                self.assertIs(first, repeated)
                self.assertEqual(
                    gazou_kiritori._open_zip_cached.cache_info().maxsize,
                    8,
                )
            finally:
                first.close()
                gazou_kiritori._open_zip_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
