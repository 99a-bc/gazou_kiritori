"""Boundary checks for the pure VFS path and URI module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBE_PREFIX = "VFS_PATH_IMPORT_PROBE="
IMPORT_PROBE = textwrap.dedent(
    f"""
    import json
    import sys

    import vfs_paths

    forbidden_roots = ("PyQt6", "PIL", "torch", "gazou_kiritori")
    loaded = {{
        root: any(name == root or name.startswith(root + ".") for name in sys.modules)
        for root in forbidden_roots
    }}
    print({PROBE_PREFIX!r} + json.dumps(loaded, sort_keys=True))
    """
)


class VfsPathModuleBoundaryTests(unittest.TestCase):
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

    def test_application_reexports_the_moved_implementations(self) -> None:
        import gazou_kiritori
        import vfs_paths

        moved_functions = (
            "_ext",
            "_is_zip_like_name",
            "is_archive_file",
            "is_archive_name",
            "is_zip_uri",
            "make_zip_uri",
            "norm_vpath",
            "parse_zip_uri",
            "vfs_display_name",
        )
        for name in moved_functions:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(gazou_kiritori, name),
                    getattr(vfs_paths, name),
                )

        self.assertIs(
            gazou_kiritori.ARCHIVE_FILE_EXTS,
            vfs_paths.ARCHIVE_FILE_EXTS,
        )
        self.assertIs(
            gazou_kiritori.ARCHIVE_EMBED_EXTS,
            vfs_paths.ARCHIVE_EMBED_EXTS,
        )


if __name__ == "__main__":
    unittest.main()
