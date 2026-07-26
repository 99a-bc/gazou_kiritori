"""Physical archive adapters used by the virtual filesystem."""

from __future__ import annotations

import os
import zipfile

from vfs_paths import _ext


# RAR 対応（rarfile が無ければ None のまま）
try:
    import rarfile  # type: ignore
except Exception:
    rarfile = None

# 7z 対応（py7zr が無ければ None のまま）
try:
    import py7zr  # type: ignore
except Exception:
    py7zr = None


class PasswordProtectedArchiveError(Exception):
    """パスワード付きアーカイブだったときに投げる専用の例外。"""
    pass


class _SevenZipInfoCompat:
    """ZipInfo っぽい情報だけを持つダミー."""
    __slots__ = ("CRC", "file_size", "date_time")

    def __init__(self, *, crc=None, file_size=0, date_time=None):
        self.CRC = crc
        self.file_size = file_size
        self.date_time = date_time


class SevenZipCompat:
    """
    py7zr.SevenZipFile を zipfile.ZipFile 互換っぽく見せる薄いラッパー。
    - namelist()
    - getinfo(name) -> .CRC / .file_size / .date_time を持つオブジェクト
    - open(name) -> バイナリ file-like (BytesIO)
    """

    def __init__(self, path: str):
        if py7zr is None:
            raise RuntimeError(
                "7zファイルを開くには 'py7zr' モジュールが必要です。\n"
                "    pip install py7zr"
            )

        self._path = path

        # まず通常通り SevenZipFile を開く
        zf = py7zr.SevenZipFile(path, mode="r")

        # py7zr が持っているフラグからパスワード保護を判定
        # （新しい py7zr では password_protected プロパティがある）
        if getattr(zf, "password_protected", False):
            zf.close()
            # パスワード付きアーカイブはサポートしない
            raise PasswordProtectedArchiveError(
                f"パスワード付きアーカイブはサポートしていません: {path}"
            )

        self._zf = zf

        # 古いバージョン（0.22 以前）は read() がある、新しい 1.0 以降は無い
        self._has_read = hasattr(self._zf, "read")
        self._build_index()

    # ------- メタ情報のインデックス -------

    def _build_index(self) -> None:
        files: dict[str, object] = {}
        file_list = getattr(self._zf, "files", [])
        for af in file_list:
            name = getattr(af, "filename", None)
            if not name:
                continue
            files[name] = af
        self._files = files

    # ------- ZipFile 互換メソッド -------

    def namelist(self):
        return list(self._files.keys())

    def getinfo(self, name: str) -> _SevenZipInfoCompat:
        af = self._files[name]

        # CRC 相当
        crc = getattr(af, "CRC", None)
        if crc is None:
            crc = getattr(af, "crc32", None)

        # サイズ
        size = getattr(af, "file_size", None)
        if size is None:
            uncompressed = getattr(af, "uncompressed", None)
            if isinstance(uncompressed, (list, tuple)):
                size = uncompressed[-1] if uncompressed else 0
            elif isinstance(uncompressed, int):
                size = uncompressed
            else:
                size = 0

        # 日付（署名用なのでざっくりでOK）
        dt_tuple = None
        ts = getattr(af, "lastwritetime", None)
        if ts is not None:
            to_dt = getattr(ts, "to_datetime", None)
            if callable(to_dt):
                d = to_dt()
                dt_tuple = (d.year, d.month, d.day, d.hour, d.minute, d.second)

        return _SevenZipInfoCompat(crc=crc, file_size=size, date_time=dt_tuple)

    def open(self, name: str, mode: str = "r", *args, **kwargs):
        """
        ZipFile.open と同じノリで、バイナリ file-like を返す。
        """
        target = name.replace("\\", "/")

        # --- 古い py7zr (read/readall がある) 向け ---
        if self._has_read:
            self._zf.reset()
            # read() のシグネチャは read(targets=None) なので list で渡す
            mapping = self._zf.read([target])  # type: ignore[attr-defined]
            if isinstance(mapping, dict) and mapping:
                bio = mapping.get(target)
                if bio is None:
                    # キーが微妙に違う場合があるので、とりあえず先頭を使う
                    bio = next(iter(mapping.values()))
                bio.seek(0)
                return bio
            raise FileNotFoundError(target)

        # --- 新しい py7zr v1.0〜 向け: factory + extract ---
        from py7zr import Py7zIO, WriterFactory  # type: ignore
        from io import BytesIO

        class _MemIO(Py7zIO):  # type: ignore[misc]
            def __init__(self):
                self._buf = BytesIO()
            def write(self, b):
                self._buf.write(b)
            def read(self, size=None):
                if size is None:
                    return self._buf.getvalue()
                return self._buf.getvalue()[:size]
            def seek(self, offset, whence=0):
                return self._buf.seek(offset, whence)
            def flush(self):
                pass
            def size(self):
                return len(self._buf.getvalue())

        class _Factory(WriterFactory):  # type: ignore[misc]
            def __init__(self, want: str):
                self.want = want
                self.io: _MemIO | None = None
            def create(self, fname: str) -> Py7zIO:  # type: ignore[override]
                fname_norm = fname.replace("\\", "/")
                io = _MemIO()
                if fname_norm == self.want:
                    self.io = io
                return io

        factory = _Factory(target)
        self._zf.reset()
        # 指定ファイルだけ展開（全展開よりマシ）
        self._zf.extract(targets=[target], factory=factory)
        if factory.io is None:
            raise FileNotFoundError(target)
        factory.io._buf.seek(0)
        return factory.io._buf

    def close(self):
        self._zf.close()


def open_physical_archive(zip_path: str, log_debug=None):
    """
    zip_path に応じて ZipFile / RarFile / 7z を返す。
    戻り値は ZipFile / RarFile / SevenZipCompat 互換を想定。
    パスワード付きアーカイブの場合は PasswordProtectedArchiveError を投げる。
    """
    if log_debug is None:
        def log_debug(*args, **kwargs):
            pass

    # ★ ここで通常パスを正規化する
    if isinstance(zip_path, str):
        zip_path = os.path.normpath(zip_path)

    ext = _ext(zip_path)

    # -------- RAR / CBR --------
    if ext in (".rar", ".cbr"):
        if rarfile is None:
            raise RuntimeError(
                "RARファイルを開くには 'rarfile' モジュールのインストールが必要です。\n"
                "    pip install rarfile\n"
                "加えて unrar / unar / bsdtar などの展開コマンドが PATH 上にある必要があります。"
            )

        log_debug(f"[rar_pw] open {zip_path}")
        rf = rarfile.RarFile(zip_path)  # type: ignore[attr-defined]

        # --- まず各エントリのフラグ状況をログに出す（判定には使わない） ---
        infos: list = []
        try:
            infos = rf.infolist()
            log_debug(f"[rar_pw]  infolist len={len(infos)}")
        except Exception as e:
            log_debug(f"[rar_pw]  infolist error: {e!r}")

        test_info = None

        for inf in infos:
            # needs_password が属性かメソッドか両対応で見る（ログ用）
            enc_attr = getattr(inf, "needs_password", False)
            try:
                enc = bool(enc_attr()) if callable(enc_attr) else bool(enc_attr)
            except Exception:
                enc = bool(getattr(inf, "needs_password", False))

            name = getattr(inf, "filename", None)

            # ディレクトリ判定
            is_dir = False
            try:
                if hasattr(inf, "is_dir"):
                    is_dir = inf.is_dir()
                elif hasattr(inf, "isdir"):
                    is_dir = inf.isdir()
            except Exception:
                is_dir = False

            log_debug(
                f"[rar_pw]  entry: name={name!r}, "
                f"needs_password={enc}, is_dir={is_dir}"
            )

            # テストに使うのは最初の「非ディレクトリ」エントリ
            if not is_dir and test_info is None:
                test_info = inf

        PasswordRequired = getattr(rarfile, "PasswordRequired", None)
        ErrorBase = getattr(rarfile, "Error", Exception)

        needs_pwd = False

        # --- 1ファイルだけ実際に開いて 1バイト読んでみる ---
        if test_info is not None:
            name = getattr(test_info, "filename", None)
            log_debug(f"[rar_pw]  test entry: {name!r}")
            try:
                with rf.open(test_info) as f:  # パスワード指定なし
                    chunk = f.read(1)
                log_debug(
                    f"[rar_pw]  test_open ok: read={len(chunk)} byte "
                    f"from {name!r}"
                )
                needs_pwd = False
            except Exception as e:
                log_debug(
                    f"[rar_pw]  test_open error: {type(e).__name__}: {e!r}"
                )
                # rarfile.PasswordRequired だけを「パスワード必須」とみなす
                if PasswordRequired is not None and isinstance(e, PasswordRequired):
                    log_debug("[rar_pw]  -> PasswordRequired exception")
                    needs_pwd = True
                elif isinstance(e, ErrorBase) and "password" in str(e).lower():
                    # メッセージに password が入っている場合もパス付きとみなす
                    log_debug("[rar_pw]  -> Error mentions 'password'")
                    needs_pwd = True
                else:
                    # それ以外のエラーは「パスワード判定」とは切り離しておく
                    needs_pwd = False
        else:
            # テストに使えるファイルが無いときだけ、最後の手段として needs_password() を見る
            log_debug("[rar_pw]  no file entry to test, fallback to RarFile.needs_password()")
            if hasattr(rf, "needs_password"):
                try:
                    np = rf.needs_password()  # type: ignore[call-arg]
                    log_debug(f"[rar_pw]  fallback rf.needs_password() -> {np}")
                    needs_pwd = bool(np)
                except Exception as e:
                    log_debug(f"[rar_pw]  fallback rf.needs_password() error: {e!r}")
                    needs_pwd = False
            else:
                needs_pwd = False

        # --- 判定結果 ---
        if needs_pwd:
            log_debug("[rar_pw]  => treat as password-protected archive (raise)")
            rf.close()
            raise PasswordProtectedArchiveError(
                f"パスワード付きRAR/CBRアーカイブはサポートしていません: {zip_path}"
            )

        log_debug("[rar_pw]  => archive allowed (no password protection)")
        return rf

    # -------- 7z / 7zip --------
    if ext in (".7z", ".7zip"):
        # SevenZipCompat.__init__ 側で password_protected を見て
        # PasswordProtectedArchiveError を投げるようにしてある想定
        return SevenZipCompat(zip_path)

    # -------- それ以外は zip 系（zip / cbz / zipx など） --------
    zf = zipfile.ZipFile(zip_path, "r")

    try:
        # PKWARE 汎用ビットフラグの bit0 が立っているエントリは暗号化されている
        for zinfo in zf.infolist():
            # flag_bits が無いことはまず無いが、念のため getattr で保護
            flag_bits = getattr(zinfo, "flag_bits", 0)
            if flag_bits & 0x1:
                zf.close()
                raise PasswordProtectedArchiveError(
                    f"パスワード付きZIPアーカイブはサポートしていません: {zip_path}"
                )
    except Exception:
        # infolist() 自体が失敗した場合などはちゃんと閉じてから再送出
        zf.close()
        raise

    return zf
