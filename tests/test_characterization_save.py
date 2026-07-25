"""Characterization tests for the current save naming and image output paths."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

sys.dont_write_bytecode = True

# The application deliberately preloads Torch before importing PyQt6. Import it
# before QtCore so discovery keeps the production import order.
import gazou_kiritori as application

try:
    from tests.helpers import temporary_directory
except ModuleNotFoundError:
    from helpers import temporary_directory  # type: ignore[no-redef]

from PIL import Image, PngImagePlugin
from PyQt6 import QtCore


TEST_BYTECODE_CACHE = Path(__file__).resolve().parent / "__pycache__"
SAVE_METHODS = (
    "_ensure_jpeg_compatible",
    "_get_image_source_dir",
    "_output_name_from_image_path",
    "_effective_save_folder",
    "_resolve_batch_save_root",
    "_build_batch_output_path",
    "save_cropped",
)


def _pattern_image(
    size: tuple[int, int],
    *,
    mode: str = "RGB",
) -> Image.Image:
    """Return a small image whose coordinates can be recovered from its pixels."""
    image = Image.new(mode, size)
    pixels: list[tuple[int, ...]] = []
    for y in range(size[1]):
        for x in range(size[0]):
            rgb = (
                (x * 31 + y * 7) % 256,
                (x * 11 + y * 47) % 256,
                (x * 29 + y * 19) % 256,
            )
            pixels.append(rgb if mode == "RGB" else (*rgb, 40 + (x + y) % 216))
    image.putdata(pixels)
    return image


def _bind_save_methods(subject: SimpleNamespace) -> SimpleNamespace:
    for method_name in SAVE_METHODS:
        method = getattr(application.CropperApp, method_name)
        setattr(subject, method_name, MethodType(method, subject))
    return subject


def _save_subject(
    image: Image.Image,
    image_path: str | os.PathLike[str],
    save_folder: str | os.PathLike[str] | None,
    rect: tuple[int, int, int, int] | None,
    *,
    overwrite: bool = False,
    alpha_output_format: str = "png",
) -> SimpleNamespace:
    label = SimpleNamespace(
        fixed_crop_mode=False,
        fixed_crop_rect_img=None,
        fixed_crop_rect_img_base=None,
        label_to_image_coords=lambda x, y: (int(x), int(y)),
    )
    subject = SimpleNamespace(
        image=image,
        image_path=str(image_path),
        save_folder="" if save_folder is None else str(save_folder),
        save_dest_mode="same",
        overwrite_mode=overwrite,
        alpha_output_format=alpha_output_format,
        folder="",
        label=label,
        _crop_rect=None,
        _crop_rect_img=None if rect is None else QtCore.QRect(*rect),
        _suppress_save_dialog_paths=set(),
        _preserve_ui_on_next_load=None,
        model=None,
        _batch_transform_ops=[],
        _update_last_saved_size_from_path=lambda _path: None,
        _snapshot_adjust_state=lambda: {},
        open_image_from_path=lambda _path: None,
    )
    return _bind_save_methods(subject)


def _assert_same_pixels(
    testcase: unittest.TestCase,
    actual_path: str | os.PathLike[str],
    expected: Image.Image,
    *,
    expected_format: str,
) -> None:
    with Image.open(actual_path) as actual:
        actual.load()
        testcase.assertEqual(actual.format, expected_format)
        testcase.assertEqual(actual.mode, expected.mode)
        testcase.assertEqual(actual.size, expected.size)
        testcase.assertEqual(actual.tobytes(), expected.tobytes())


class OutputNamingCharacterizationTests(unittest.TestCase):
    def test_single_output_name_preserves_physical_basename_edge_cases(self) -> None:
        cases = (
            ("photo.png", "photo.png"),
            ("file with spaces.jpg", "file with spaces.jpg"),
            ("日本語 画像.webp", "日本語 画像.webp"),
            ("many.dots.in.name.PNG", "many.dots.in.name.PNG"),
            ("UPPER.JPEG", "UPPER.JPEG"),
            ("extensionless", "extensionless"),
        )
        with temporary_directory(prefix="gazou-kiritori-save-name-") as root:
            subject = _bind_save_methods(SimpleNamespace(image_path=""))
            for filename, expected in cases:
                with self.subTest(filename=filename):
                    subject.image_path = str(root / filename)
                    self.assertEqual(subject._output_name_from_image_path(), expected)

            subject.image_path = ""
            self.assertEqual(subject._output_name_from_image_path(), "cropped")

    def test_single_output_name_uses_archive_entry_basename(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-name-") as root:
            subject = _bind_save_methods(SimpleNamespace(image_path=""))
            archive_path = root / "outer archive!.zip"
            cases = (
                (
                    application.make_zip_uri(
                        str(archive_path),
                        "sub folder/日本語.image.PNG",
                    ),
                    "日本語.image.PNG",
                ),
                (
                    application.make_zip_uri("memzip:save-name", "deep/photo.JPG"),
                    "photo.JPG",
                ),
                (
                    application.make_zip_uri(str(archive_path), ""),
                    archive_path.name,
                ),
            )
            for image_path, expected in cases:
                with self.subTest(image_path=image_path):
                    subject.image_path = image_path
                    self.assertEqual(subject._output_name_from_image_path(), expected)

    def test_single_sequence_starts_at_001_and_skips_existing_numbers(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-sequence-") as root:
            source = root / "source" / "日本 語.photo.final.PNG"
            output = root / "output"
            output.mkdir()
            image = _pattern_image((4, 3))
            try:
                source.parent.mkdir()
                image.save(source, format="PNG")
                first = output / "日本 語.photo.final_cropped_001.png"
                second = output / "日本 語.photo.final_cropped_002.png"
                first.write_bytes(b"first sentinel")
                second.write_bytes(b"second sentinel")

                subject = _save_subject(image, source, output, (0, 0, 2, 2))
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved).name, "日本 語.photo.final_cropped_003.png")
                self.assertEqual(first.read_bytes(), b"first sentinel")
                self.assertEqual(second.read_bytes(), b"second sentinel")

                ok, saved_again = subject.save_cropped(None)
                self.assertTrue(ok, saved_again)
                self.assertEqual(
                    Path(saved_again).name,
                    "日本 語.photo.final_cropped_004.png",
                )
            finally:
                image.close()

    def test_batch_sequence_starts_unnumbered_then_uses_002_and_flattens_zip(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-batch-name-") as root:
            subject = _bind_save_methods(SimpleNamespace(overwrite_mode=False))
            archive = root / "archive.zip"
            first_uri = application.make_zip_uri(str(archive), "one/photo.png")
            second_uri = application.make_zip_uri(str(archive), "two/photo.png")

            first = subject._build_batch_output_path(root, first_uri)
            self.assertEqual(first.name, "photo_cropped.png")
            first.write_bytes(b"first")
            second = subject._build_batch_output_path(root, second_uri)
            self.assertEqual(second.name, "photo_cropped_002.png")
            second.write_bytes(b"second")
            third = subject._build_batch_output_path(root, first_uri)
            self.assertEqual(third.name, "photo_cropped_003.png")

    def test_batch_names_normalize_extensions_and_overwrite_has_no_suffix(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-batch-name-") as root:
            subject = _bind_save_methods(SimpleNamespace(overwrite_mode=True))
            cases = (
                ("file with spaces.JPG", "file with spaces.jpg"),
                ("日本語.multi.part.PNG", "日本語.multi.part.png"),
                ("extensionless", "extensionless.png"),
                ("unsupported.xyz", "unsupported.png"),
            )
            for source_name, expected in cases:
                with self.subTest(source_name=source_name):
                    candidate = subject._build_batch_output_path(
                        root,
                        str(root / source_name),
                    )
                    self.assertEqual(candidate.name, expected)

    @unittest.skipUnless(os.name == "nt", "records Windows filename semantics")
    def test_batch_case_only_collision_is_seen_as_existing_on_windows(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-batch-case-") as root:
            subject = _bind_save_methods(SimpleNamespace(overwrite_mode=False))
            (root / "Photo_cropped.PNG").write_bytes(b"case-only collision")

            candidate = subject._build_batch_output_path(
                root,
                str(root / "photo.png"),
            )

            self.assertEqual(candidate.name, "photo_cropped_002.png")


class SaveDestinationCharacterizationTests(unittest.TestCase):
    def test_source_directory_resolves_physical_zip_and_nested_zip(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-dest-") as root:
            subject = _bind_save_methods(SimpleNamespace())
            physical = root / "plain" / "photo.png"
            archive = root / "archives" / "outer.zip"
            zip_uri = application.make_zip_uri(
                str(archive),
                "subdir/photo.png",
            )
            inner_id = "memzip:save-destination-inner"
            outer_id = "memzip:save-destination-outer"
            nested_uri = application.make_zip_uri(inner_id, "photo.png")
            saved_meta = {
                key: dict(value) for key, value in application._MEM_ZIP_META.items()
            }
            try:
                application._MEM_ZIP_META[inner_id] = {
                    "outer": outer_id,
                    "inner": "inner-most.zip",
                }
                application._MEM_ZIP_META[outer_id] = {
                    "outer": str(archive),
                    "inner": "middle.zip",
                }

                self.assertEqual(
                    Path(subject._get_image_source_dir(str(physical))),
                    physical.parent,
                )
                self.assertEqual(
                    Path(subject._get_image_source_dir(zip_uri)),
                    archive.parent,
                )
                self.assertEqual(
                    Path(subject._get_image_source_dir(nested_uri)),
                    archive.parent,
                )
            finally:
                application._MEM_ZIP_META.clear()
                application._MEM_ZIP_META.update(saved_meta)

    def test_effective_save_folder_prefers_explicit_then_source_then_browser(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-effective-dest-") as root:
            source = root / "source" / "photo.png"
            explicit = root / "explicit" / ".." / "explicit"
            subject = _bind_save_methods(
                SimpleNamespace(
                    save_folder=str(explicit),
                    image_path=str(source),
                    folder=str(root / "browser"),
                )
            )
            self.assertEqual(
                subject._effective_save_folder(),
                os.path.normpath(str(explicit)),
            )

            subject.save_folder = ""
            self.assertEqual(
                Path(subject._effective_save_folder()),
                source.parent,
            )

            subject.image_path = ""
            self.assertEqual(
                Path(subject._effective_save_folder()),
                root / "browser",
            )

    def test_batch_save_root_resolves_custom_physical_and_zip_sources(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-batch-dest-") as root:
            custom = root / "custom"
            archive = root / "archives" / "source.zip"
            physical = root / "physical" / "photo.png"
            zip_uri = application.make_zip_uri(
                str(archive),
                "subdir/photo.png",
            )
            subject = _bind_save_methods(
                SimpleNamespace(
                    save_folder=str(custom),
                    save_dest_mode="custom",
                )
            )
            self.assertEqual(subject._resolve_batch_save_root(str(physical)), custom)
            self.assertEqual(subject._resolve_batch_save_root(zip_uri), custom)

            subject.save_folder = ""
            subject.save_dest_mode = "same"
            self.assertEqual(
                subject._resolve_batch_save_root(str(physical)),
                physical.parent,
            )
            self.assertEqual(
                subject._resolve_batch_save_root(zip_uri),
                archive.parent,
            )

    def test_nested_zip_batch_root_currently_resolves_to_working_directory(
        self,
    ) -> None:
        """Record the current nested-batch destination issue without fixing it."""
        with temporary_directory(prefix="gazou-kiritori-nested-batch-") as root:
            subject = _bind_save_methods(
                SimpleNamespace(save_folder="", save_dest_mode="same")
            )
            nested_uri = application.make_zip_uri(
                "memzip:save-batch-current-behavior",
                "deep/photo.png",
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                save_root = subject._resolve_batch_save_root(nested_uri)
                self.assertEqual(save_root, Path("."))
                self.assertEqual(save_root.resolve(), root.resolve())
            finally:
                os.chdir(previous_cwd)

    def test_single_save_defaults_to_physical_zip_and_outer_zip_directories(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-single-dest-") as root:
            image = _pattern_image((4, 3))
            saved_meta = {
                key: dict(value) for key, value in application._MEM_ZIP_META.items()
            }
            try:
                physical_source = root / "physical" / "plain.png"
                physical_source.parent.mkdir()
                image.save(physical_source, format="PNG")

                archive = root / "archive-source" / "outer.zip"
                archive.parent.mkdir()
                zip_uri = application.make_zip_uri(
                    str(archive),
                    "subdir/inside.png",
                )

                mem_id = "memzip:single-save-current-behavior"
                application._MEM_ZIP_META[mem_id] = {
                    "outer": str(archive),
                    "inner": "nested/inner.zip",
                }
                nested_uri = application.make_zip_uri(
                    mem_id,
                    "deep/nested.png",
                )

                cases = (
                    (physical_source, physical_source.parent, "plain"),
                    (zip_uri, archive.parent, "inside"),
                    (nested_uri, archive.parent, "nested"),
                )
                for image_path, expected_parent, base in cases:
                    with self.subTest(image_path=str(image_path)):
                        subject = _save_subject(
                            image,
                            image_path,
                            None,
                            (0, 0, 2, 2),
                        )
                        ok, saved = subject.save_cropped(None)
                        self.assertTrue(ok, saved)
                        self.assertEqual(Path(saved).parent, expected_parent)
                        self.assertEqual(Path(saved).name, f"{base}_cropped_001.png")
            finally:
                image.close()
                application._MEM_ZIP_META.clear()
                application._MEM_ZIP_META.update(saved_meta)

    def test_explicit_relative_missing_destination_is_created_under_cwd(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-relative-dest-") as root:
            source = root / "source" / "photo.png"
            source.parent.mkdir()
            image = _pattern_image((4, 3))
            previous_cwd = Path.cwd()
            try:
                image.save(source, format="PNG")
                os.chdir(root)
                subject = _save_subject(
                    image,
                    source,
                    Path("relative") / "new-output",
                    (0, 0, 2, 2),
                )
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)
                self.assertEqual(
                    Path(saved).resolve(),
                    (root / "relative" / "new-output" / "photo_cropped_001.png"),
                )
                self.assertTrue(Path(saved).is_file())
            finally:
                os.chdir(previous_cwd)
                image.close()


class FullImageSaveCharacterizationTests(unittest.TestCase):
    def test_unmodified_full_jpeg_is_copied_byte_for_byte_and_reopens(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-full-copy-") as root:
            source = root / "source" / "photo.jpg"
            output = root / "output"
            source.parent.mkdir()
            image = _pattern_image((6, 4))
            try:
                image.save(source, format="JPEG", quality=83)
            finally:
                image.close()

            with Image.open(source) as loaded:
                loaded.load()
                current = loaded.copy()
            try:
                subject = _save_subject(current, source, output, (0, 0, 6, 4))
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved).read_bytes(), source.read_bytes())
                with Image.open(saved) as reopened:
                    reopened.load()
                    self.assertEqual(reopened.format, "JPEG")
                    self.assertEqual(reopened.size, (6, 4))
                    self.assertEqual(reopened.tobytes(), current.tobytes())
            finally:
                current.close()

    def test_transformed_full_jpeg_currently_copies_original_file(self) -> None:
        """Record that current transformed pixels are ignored by full-copy save."""
        with temporary_directory(prefix="gazou-kiritori-full-transformed-") as root:
            source = root / "source" / "rotated.jpg"
            output = root / "output"
            source.parent.mkdir()
            original = _pattern_image((5, 3))
            try:
                original.save(source, format="JPEG", quality=87)
            finally:
                original.close()

            with Image.open(source) as loaded:
                loaded.load()
                current = loaded.transpose(Image.Transpose.ROTATE_90)
            try:
                subject = _save_subject(current, source, output, (0, 0, 3, 5))
                subject._batch_transform_ops = ["rot_left_90", "flip_h"]
                subject.rotation_angle = 90
                subject.horizontal_flipped = True

                ok, saved = subject.save_cropped(None)

                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved).read_bytes(), source.read_bytes())
                with Image.open(saved) as reopened:
                    reopened.load()
                    self.assertEqual(reopened.size, (5, 3))
                    self.assertNotEqual(reopened.size, current.size)
            finally:
                current.close()

    def test_transformed_full_overwrite_same_source_reports_success_unchanged(
        self,
    ) -> None:
        """Record the no-write branch when full-copy output is the source path."""
        with temporary_directory(prefix="gazou-kiritori-full-overwrite-") as root:
            source = root / "photo.jpg"
            original = _pattern_image((5, 3))
            try:
                original.save(source, format="JPEG", quality=89)
            finally:
                original.close()
            original_bytes = source.read_bytes()

            with Image.open(source) as loaded:
                loaded.load()
                current = loaded.transpose(Image.Transpose.ROTATE_90)
            try:
                reopened_paths: list[str] = []
                subject = _save_subject(
                    current,
                    source,
                    source.parent,
                    (0, 0, 3, 5),
                    overwrite=True,
                )
                subject.open_image_from_path = reopened_paths.append

                ok, saved = subject.save_cropped(None)

                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved), source)
                self.assertEqual(source.read_bytes(), original_bytes)
                self.assertEqual(reopened_paths, [str(source)])
            finally:
                current.close()

    def test_full_png_size_matrix_preserves_pixels(self) -> None:
        sizes = ((1, 1), (7, 2), (2, 7), (5, 3), (4, 2))
        with temporary_directory(prefix="gazou-kiritori-full-matrix-") as root:
            for index, size in enumerate(sizes):
                with self.subTest(size=size):
                    case_root = root / str(index)
                    source = case_root / "source.png"
                    output = case_root / "output"
                    source.parent.mkdir(parents=True)
                    image = _pattern_image(size)
                    try:
                        image.save(source, format="PNG")
                        subject = _save_subject(
                            image,
                            source,
                            output,
                            (0, 0, size[0], size[1]),
                        )
                        ok, saved = subject.save_cropped(None)
                        self.assertTrue(ok, saved)
                        _assert_same_pixels(
                            self,
                            saved,
                            image,
                            expected_format="PNG",
                        )
                    finally:
                        image.close()

    def test_one_pixel_smaller_and_edge_touching_jpeg_are_reencoded(self) -> None:
        cases = (
            ("one-smaller", (0, 0, 4, 3), (4, 3)),
            ("right-edge", (1, 0, 4, 4), (4, 4)),
            ("bottom-edge", (0, 1, 5, 3), (5, 3)),
        )
        with temporary_directory(prefix="gazou-kiritori-partial-jpeg-") as root:
            for name, rect, expected_size in cases:
                with self.subTest(name=name):
                    case_root = root / name
                    source = case_root / "source.jpg"
                    output = case_root / "output"
                    source.parent.mkdir(parents=True)
                    image = _pattern_image((5, 4))
                    try:
                        image.save(source, format="JPEG", quality=82)
                    finally:
                        image.close()

                    with Image.open(source) as loaded:
                        loaded.load()
                        current = loaded.copy()
                    try:
                        subject = _save_subject(current, source, output, rect)
                        ok, saved = subject.save_cropped(None)
                        self.assertTrue(ok, saved)
                        self.assertNotEqual(
                            Path(saved).read_bytes(),
                            source.read_bytes(),
                        )
                        with Image.open(saved) as reopened:
                            self.assertEqual(reopened.format, "JPEG")
                            self.assertEqual(reopened.size, expected_size)
                    finally:
                        current.close()


class CropSaveCharacterizationTests(unittest.TestCase):
    def test_png_crop_matrix_preserves_half_open_sizes_and_pixels(self) -> None:
        cases = (
            ("center", (2, 1, 3, 3)),
            ("top-left", (0, 0, 2, 2)),
            ("right-bottom", (4, 3, 2, 2)),
            ("width-one", (3, 1, 1, 3)),
            ("height-one", (1, 4, 4, 1)),
        )
        with temporary_directory(prefix="gazou-kiritori-crop-matrix-") as root:
            for name, rect in cases:
                with self.subTest(name=name):
                    case_root = root / name
                    source = case_root / "source.png"
                    output = case_root / "output"
                    source.parent.mkdir(parents=True)
                    image = _pattern_image((6, 5))
                    expected = image.crop(
                        (
                            rect[0],
                            rect[1],
                            rect[0] + rect[2],
                            rect[1] + rect[3],
                        )
                    )
                    try:
                        image.save(source, format="PNG")
                        subject = _save_subject(image, source, output, rect)
                        ok, saved = subject.save_cropped(None)
                        self.assertTrue(ok, saved)
                        _assert_same_pixels(
                            self,
                            saved,
                            expected,
                            expected_format="PNG",
                        )
                    finally:
                        expected.close()
                        image.close()

    def test_label_qrect_fallback_maps_to_half_open_pillow_box(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-qrect-save-") as root:
            source = root / "source.png"
            output = root / "output"
            image = _pattern_image((7, 6))
            expected = image.crop((1, 2, 4, 4))
            try:
                image.save(source, format="PNG")
                subject = _save_subject(image, source, output, None)
                subject._crop_rect_img = None
                label_rect = QtCore.QRect(1, 2, 3, 2)

                ok, saved = subject.save_cropped(label_rect)

                self.assertTrue(ok, saved)
                _assert_same_pixels(
                    self,
                    saved,
                    expected,
                    expected_format="PNG",
                )
            finally:
                expected.close()
                image.close()


class SaveFormatAndMetadataCharacterizationTests(unittest.TestCase):
    def test_output_format_and_mode_follow_supported_source_extension(self) -> None:
        cases = (
            ("photo.PNG", "PNG", "RGB", ".png"),
            ("photo.JPG", "JPEG", "RGB", ".jpg"),
            ("photo.unsupported", "PNG", "RGB", ".png"),
            ("extensionless", "PNG", "RGB", ".png"),
        )
        with temporary_directory(prefix="gazou-kiritori-save-format-") as root:
            for index, (name, expected_format, expected_mode, suffix) in enumerate(
                cases
            ):
                with self.subTest(name=name):
                    case_root = root / str(index)
                    source = case_root / name
                    output = case_root / "output"
                    source.parent.mkdir(parents=True)
                    image = _pattern_image((4, 3))
                    try:
                        subject = _save_subject(
                            image,
                            source,
                            output,
                            (0, 0, 3, 2),
                        )
                        ok, saved = subject.save_cropped(None)
                        self.assertTrue(ok, saved)
                        self.assertEqual(Path(saved).suffix, suffix)
                        with Image.open(saved) as reopened:
                            reopened.load()
                            self.assertEqual(reopened.format, expected_format)
                            self.assertEqual(reopened.mode, expected_mode)
                            self.assertEqual(reopened.size, (3, 2))
                    finally:
                        image.close()

    def test_rgba_with_jpeg_source_name_currently_forces_png_output(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-alpha-format-") as root:
            source = root / "alpha.jpg"
            output = root / "output"
            source_rgb = _pattern_image((3, 2))
            rgba = _pattern_image((3, 2), mode="RGBA")
            try:
                source_rgb.save(source, format="JPEG")
                subject = _save_subject(
                    rgba,
                    source,
                    output,
                    (0, 0, 3, 2),
                    alpha_output_format="png",
                )
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved).name, "alpha_cropped_001.png")
                _assert_same_pixels(
                    self,
                    saved,
                    rgba,
                    expected_format="PNG",
                )
            finally:
                rgba.close()
                source_rgb.close()

    def test_jpeg_compatibility_flattens_rgba_onto_white(self) -> None:
        rgba = Image.new("RGBA", (2, 1))
        rgba.putdata(((10, 20, 30, 0), (40, 50, 60, 255)))
        subject = _bind_save_methods(SimpleNamespace())
        try:
            converted = subject._ensure_jpeg_compatible(rgba, "jpg")
            try:
                self.assertEqual(converted.mode, "RGB")
                self.assertEqual(
                    [converted.getpixel((0, 0)), converted.getpixel((1, 0))],
                    [(255, 255, 255), (40, 50, 60)],
                )
            finally:
                if converted is not rgba:
                    converted.close()
        finally:
            rgba.close()

    def test_partial_jpeg_save_preserves_exif_currently_passed_by_save(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-exif-") as root:
            source = root / "source.jpg"
            output = root / "output"
            description = "gazou-kiritori characterization"
            image = _pattern_image((5, 4))
            exif = Image.Exif()
            exif[0x010E] = description
            try:
                image.save(source, format="JPEG", quality=90, exif=exif)
            finally:
                image.close()

            with Image.open(source) as loaded:
                loaded.load()
                subject = _save_subject(loaded, source, output, (0, 0, 4, 3))
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)

            with Image.open(saved) as reopened:
                self.assertEqual(reopened.format, "JPEG")
                self.assertEqual(reopened.getexif().get(0x010E), description)

    def test_arbitrary_png_text_metadata_is_currently_dropped(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-pnginfo-") as root:
            source = root / "source.png"
            output = root / "output"
            image = _pattern_image((5, 4))
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("characterization-note", "kept only in source")
            try:
                image.save(source, format="PNG", pnginfo=pnginfo)
            finally:
                image.close()

            with Image.open(source) as loaded:
                loaded.load()
                self.assertEqual(
                    loaded.info.get("characterization-note"),
                    "kept only in source",
                )
                subject = _save_subject(loaded, source, output, (0, 0, 4, 3))
                ok, saved = subject.save_cropped(None)
                self.assertTrue(ok, saved)

            with Image.open(saved) as reopened:
                self.assertNotIn("characterization-note", reopened.info)


class SaveCollisionAndFailureCharacterizationTests(unittest.TestCase):
    def test_overwrite_mode_replaces_existing_output_inside_temporary_directory(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-save-overwrite-") as root:
            source = root / "source" / "photo.png"
            output = root / "output"
            source.parent.mkdir()
            output.mkdir()
            old = Image.new("RGB", (3, 2), (1, 2, 3))
            current = Image.new("RGB", (3, 2), (200, 150, 100))
            destination = output / "photo.png"
            try:
                old.save(source, format="PNG")
                old.save(destination, format="PNG")
                old_bytes = destination.read_bytes()
                subject = _save_subject(
                    current,
                    source,
                    output,
                    (0, 0, 3, 2),
                    overwrite=True,
                )

                ok, saved = subject.save_cropped(None)

                self.assertTrue(ok, saved)
                self.assertEqual(Path(saved), destination)
                self.assertNotEqual(destination.read_bytes(), old_bytes)
                _assert_same_pixels(
                    self,
                    destination,
                    current,
                    expected_format="PNG",
                )
            finally:
                current.close()
                old.close()

    def test_missing_image_and_non_intersecting_rect_return_false(self) -> None:
        no_image = _bind_save_methods(SimpleNamespace(image=None))
        ok, message = no_image.save_cropped(None)
        self.assertFalse(ok)
        self.assertEqual(message, "画像が読み込まれていません")

        image = _pattern_image((4, 3))
        try:
            outside = _save_subject(
                image,
                "unused.png",
                "unused-output",
                (10, 10, 2, 2),
            )
            ok, message = outside.save_cropped(None)
            self.assertFalse(ok)
            self.assertEqual(message, "切り出し範囲が画像外です")
        finally:
            image.close()

    def test_missing_full_jpeg_source_falls_back_to_encoded_current_image(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-missing-source-") as root:
            source = root / "missing.jpg"
            output = root / "output"
            image = _pattern_image((4, 3))
            try:
                subject = _save_subject(image, source, output, (0, 0, 4, 3))

                ok, saved = subject.save_cropped(None)

                self.assertTrue(ok, saved)
                self.assertFalse(source.exists())
                with Image.open(saved) as reopened:
                    reopened.load()
                    self.assertEqual(reopened.format, "JPEG")
                    self.assertEqual(reopened.size, (4, 3))
            finally:
                image.close()

    def test_pillow_write_exception_is_returned_as_save_failure(self) -> None:
        with temporary_directory(prefix="gazou-kiritori-write-failure-") as root:
            image = _pattern_image((4, 3))
            try:
                subject = _save_subject(
                    image,
                    root / "source.png",
                    root / "output",
                    (0, 0, 2, 2),
                )
                with mock.patch.object(
                    Image.Image,
                    "save",
                    side_effect=OSError("forced characterization write failure"),
                ):
                    ok, message = subject.save_cropped(None)

                self.assertFalse(ok)
                self.assertIn("forced characterization write failure", message)
                self.assertEqual(list((root / "output").iterdir()), [])
            finally:
                image.close()

    def test_save_root_that_is_a_file_returns_failure_without_overwriting_it(
        self,
    ) -> None:
        with temporary_directory(prefix="gazou-kiritori-invalid-root-") as root:
            invalid_root = root / "not-a-directory"
            invalid_root.write_bytes(b"destination sentinel")
            image = _pattern_image((4, 3))
            try:
                subject = _save_subject(
                    image,
                    root / "source.png",
                    invalid_root,
                    (0, 0, 2, 2),
                )

                ok, message = subject.save_cropped(None)

                self.assertFalse(ok)
                self.assertTrue(message)
                self.assertEqual(
                    invalid_root.read_bytes(),
                    b"destination sentinel",
                )
            finally:
                image.close()


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
