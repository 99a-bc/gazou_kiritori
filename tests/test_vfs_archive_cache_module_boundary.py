"""Boundary checks for the owning archive-reader cache."""

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
PROBE_PREFIX = "VFS_ARCHIVE_CACHE_IMPORT_PROBE="
IMPORT_PROBE = textwrap.dedent(
    f"""
    import json
    import sys

    import vfs_archive_cache

    forbidden_roots = ("PyQt6", "PIL", "torch", "gazou_kiritori")
    loaded = {{
        root: any(name == root or name.startswith(root + ".") for name in sys.modules)
        for root in forbidden_roots
    }}
    print({PROBE_PREFIX!r} + json.dumps(loaded, sort_keys=True))
    """
)


class VfsArchiveCacheModuleBoundaryTests(unittest.TestCase):
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

    def test_initial_cache_info(self) -> None:
        from vfs_archive_cache import ArchiveReaderCache

        cache = ArchiveReaderCache(maxsize=8)
        cache_info = cache.cache_info()

        self.assertEqual(cache_info.hits, 0)
        self.assertEqual(cache_info.misses, 0)
        self.assertEqual(cache_info.maxsize, 8)
        self.assertEqual(cache_info.currsize, 0)


if __name__ == "__main__":
    unittest.main()
