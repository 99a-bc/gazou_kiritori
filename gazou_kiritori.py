# ===== stdlib =====
import sys
import os
import re
import math
import traceback
import time
from io import BytesIO
import os, tempfile
import subprocess
from pathlib import Path


# ===== third-party =====
from PIL.ImageQt import ImageQt
from PIL import Image, ImageOps

# ===== PyQt6 =====
from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtGui import QAction, QActionGroup, QShortcut, QKeySequence
from PyQt6.QtCore import QPoint, QSignalBlocker
from PyQt6.QtWidgets import QSizePolicy, QMessageBox
from PyQt6.QtWidgets import QDialogButtonBox

# ===== VFS (zip / rar / 7z) helpers: メモリ展開のみ／tempは使わない =====
import zipfile
from functools import lru_cache
from io import BytesIO

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

APP_NAME = "画像切り取りツール"
APP_VERSION = "1.1.1" 

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# 物理フォルダで「フォルダ扱い」したい拡張子
# ※ .cbr も実質 rar ベースなので一緒に対応しておく
ARCHIVE_FILE_EXTS = {".zip", ".cbz", ".rar", ".cbr", ".7z", ".7zip"}

# zip 内 zip (memzip) として特別扱いする拡張子（従来どおり zip/cbz のみ）
ARCHIVE_EMBED_EXTS = {".zip", ".cbz"}

from collections import OrderedDict
_IMG_CACHE = OrderedDict()
_IMG_CACHE_MAX = 8

# --- zip内zip 用：メモリ上に展開した zip 本体 ---
_MEM_ZIP_BYTES: dict[str, bytes] = {}
_MEM_ZIP_META: dict[str, dict] = {}
_MEM_ZIP_COUNTER = 0

from html import escape

# ===== debug flags =====
DEBUG_VIEW_RECT = False

class PasswordProtectedArchiveError(Exception):
    """パスワード付きアーカイブだったときに投げる専用の例外。"""
    pass

def _dbg_time(label, start=None):
    """
    start=None で呼ぶと開始時刻を返す。
    start にその値を渡すと経過時間を表示する。
    """
    if start is None:
        return time.perf_counter()
    else:
        dt = time.perf_counter() - start
        # ★ ログONのときだけ出力
        if LOG_ENABLED:
            print(f"[perf] {label}: {dt*1000:.1f} ms")
        return None

# ----------------------------------------
# ログ出力制御
# ----------------------------------------
LOG_ENABLED = False  # デフォルトではログ出さない

def log_debug(*args, **kwargs):
    """
    デバッグ用ログ出力。
    LOG_ENABLED が True のときだけ print する。
    """
    if LOG_ENABLED:
        print(*args, **kwargs)

def _sig_for(path_or_uri: str):
    import os
    if is_zip_uri(path_or_uri):
        zp, inner = parse_zip_uri(path_or_uri)
        zf = _open_zip_cached(zp)
        # ★ まずケース非依存で実在名に解決
        inner2 = _zip_resolve_inner(zp, inner)
        try:
            info = zf.getinfo(inner2)
            return ("zip", norm_vpath(zp), inner2, info.CRC, info.file_size, info.date_time)
        except KeyError:
            # 署名が取れないときは ZIP + inner（正規化）の簡易署名で代用（キャッシュ精度は下がるが動作はする）
            return ("zip", norm_vpath(zp), inner.lower())
    else:
        st = os.stat(path_or_uri)
        return ("disk", norm_vpath(path_or_uri), st.st_mtime_ns, st.st_size)

def _cache_get(path_or_uri):
    key = norm_vpath(path_or_uri)
    try:
        sig = _sig_for(path_or_uri)  # ← ここで getinfo が落ちることがある
    except Exception as e:
        log_debug(f"[img-cache] MISS key={key} (sig error: {e})")
        return None

    item = _IMG_CACHE.get(key)
    if item and item['sig'] == sig:
        log_debug(f"[img-cache] HIT  key={key}")
        _IMG_CACHE.move_to_end(key)
        return item['img']

    log_debug(f"[img-cache] MISS key={key}")
    return None

def _cache_put(path_or_uri, img):
    key = norm_vpath(path_or_uri)
    sig = _sig_for(path_or_uri)
    _IMG_CACHE[key] = {"sig": sig, "img": img}
    _IMG_CACHE.move_to_end(key)
    # ★ 追加：格納ログ & 現在サイズ
    log_debug(f"[img-cache] PUT  key={key}  size={len(_IMG_CACHE)}/{_IMG_CACHE_MAX}")
    while len(_IMG_CACHE) > _IMG_CACHE_MAX:
        evk, _ = _IMG_CACHE.popitem(last=False)
        log_debug(f"[img-cache] EVICT {evk}")

def _ext(s: str) -> str:
    s = (s or "").lower()
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if s.endswith(ext):
            return ext
    import os
    return os.path.splitext(s)[1]

def is_image_name(name: str) -> bool:
    return _ext(name) in IMAGE_EXTS

def is_archive_file(path: str) -> bool:
    """物理ファイルが zip / cbz / rar / cbr かどうか"""
    return _ext(path) in ARCHIVE_FILE_EXTS


def is_archive_name(name: str) -> bool:
    """パス/URI/zip内エントリ名を拡張子だけでアーカイブ判定"""
    return _ext(name) in ARCHIVE_FILE_EXTS


def _is_zip_like_name(name: str) -> bool:
    """
    zip 内 zip(memzip) として特別扱いする拡張子だけ判定。
    → zip / cbz のみ（rar はここでは扱わない）
    """
    return _ext(name) in ARCHIVE_EMBED_EXTS

def _register_mem_zip(parent_zip_path: str, inner_name: str) -> str:
    """
    zipファイル内に含まれる zip/CBZ を、メモリ上の仮想zipとして登録し、
    そのID (memzip:*) を返す。既に登録済みなら同じIDを再利用する。
    """
    global _MEM_ZIP_COUNTER

    inner_norm = (inner_name or "").replace("\\", "/")

    # 既に登録済みかチェック（親zip + エントリ名の組み合わせでユニーク）
    for mid, meta in _MEM_ZIP_META.items():
        if meta.get("outer") == parent_zip_path and meta.get("inner") == inner_norm:
            return mid

    zf = _open_zip_cached(parent_zip_path)
    # ケース非依存で実在名に解決
    inner_real = _zip_resolve_inner(parent_zip_path, inner_norm)
    data = zf.read(inner_real)

    mem_id = f"memzip:{_MEM_ZIP_COUNTER}"
    _MEM_ZIP_COUNTER += 1

    _MEM_ZIP_BYTES[mem_id] = data
    _MEM_ZIP_META[mem_id] = {"outer": parent_zip_path, "inner": inner_real}
    return mem_id

def make_zip_uri(zip_path: str, inner: str) -> str:
    import os
    inner = (inner or "").replace("\\", "/")
    if inner and not inner.startswith("/"):
        inner = "/" + inner

    # memzip:* のときだけは abspath を噛ませず、そのまま使う
    if isinstance(zip_path, str) and zip_path.startswith("memzip:"):
        base = zip_path
    else:
        base = os.path.abspath(zip_path)

    return f"zip://{base}!{inner}"

def is_zip_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("zip://") and "!" in uri

def parse_zip_uri(uri: str):
    """zip://... 形式の URI を「実ファイルパス」と「zip 内パス」に分解する。

    ★ ポイント:
        * zip 側のファイル名に '!' が含まれていても壊れないようにする
        * 区切りの '!' は「直後が '/' か、URI の終端」のものを使う
          例:
            zip://C:/dir/aa!.zip!/inner/file.png
                -> zip_path = C:/dir/aa!.zip
                   inner    = inner/file.png
            zip://C:/dir/aa!.zip!
                -> zip_path = C:/dir/aa!.zip
                   inner    = ""
    """
    if not is_zip_uri(uri):
        raise ValueError(f"not a zip uri: {uri!r}")

    body = uri[len("zip://"):]

    sep = -1
    # 「直後が '/' か終端」の '!' を区切りとして採用
    for i, ch in enumerate(body):
        if ch == "!" and (i + 1 == len(body) or body[i + 1] == "/"):
            sep = i
            break

    if sep == -1:
        # 念のためのフォールバック（従来互換）。基本的には来ない想定。
        sep = body.find("!")
        if sep == -1:
            raise ValueError(f"invalid zip uri (no '!'): {uri!r}")

    zip_path = body[:sep]
    inner = body[sep + 1 :]

    if inner.startswith("/"):
        inner = inner[1:]

    return zip_path, inner

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

@lru_cache(maxsize=8)
def _open_zip_cached(zip_path: str):
    """
    zip_path に応じて ZipFile / RarFile / 7z / memzip を返す。
    戻り値は ZipFile / RarFile / SevenZipCompat 互換を想定。
    パスワード付きアーカイブの場合は PasswordProtectedArchiveError を投げる。
    """
    # memzip: メモリ上に展開済みの zip（ここは従来どおり ZipFile 固定）
    if isinstance(zip_path, str) and zip_path.startswith("memzip:"):
        data = _MEM_ZIP_BYTES.get(zip_path)
        if data is None:
            raise FileNotFoundError(f"memzip not registered: {zip_path}")
        return zipfile.ZipFile(BytesIO(data), "r")

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

@lru_cache(maxsize=32)
def _zip_index_lower(zip_path: str) -> dict[str, str]:
    """
    zip内のエントリを小文字キーで引ける dict を返す。
    { lower(name) : name(オリジナルケース) }
    """
    zf = _open_zip_cached(zip_path)
    # getinfoはケース完全一致なので、事前に名寄せテーブルを作る
    return { name.lower(): name for name in zf.namelist() }

def _zip_resolve_inner(zip_path: str, inner: str) -> str:
    """
    inner（大小文字が混ざっているかも）を、実在するオリジナルケースに解決。
    見つからなければ inner をそのまま返す（上位で KeyError を拾う）。
    """
    if not inner:
        return inner
    idx = _zip_index_lower(zip_path)
    key = inner.replace("\\", "/").lower()
    return idx.get(key, inner)

def _is_noise_entry(name: str) -> bool:
    """__MACOSX フォルダと ._* を除外するためのフィルタ。"""
    n = (name or "").replace("\\", "/").strip("/")
    return n == "__MACOSX" or n.startswith("._")

def _zip_list_children(zip_path: str, inner_dir: str):
    """zip内の inner_dir 直下の子を一段だけ列挙。"""
    zf = _open_zip_cached(zip_path)
    inner = (inner_dir or "").strip("/")
    prefix = (inner + "/") if inner else ""
    seen_dirs = set()
    children = []
    for n in zf.namelist():
        if not n.startswith(prefix):
            continue
        rest = n[len(prefix):]
        if not rest:
            continue
        part = rest.split("/", 1)
        head = part[0]
        if _is_noise_entry(head):
            continue
        if len(part) == 1:
            # ファイル
            if _is_zip_like_name(head):
                # zip内の zip/CBZ は“フォルダ扱い”
                children.append({"name": head, "is_dir": True,
                                 "uri": make_zip_uri(zip_path, prefix + head)})
            else:
                children.append({"name": head, "is_dir": False,
                                 "uri": make_zip_uri(zip_path, prefix + head)})
        else:
            if head not in seen_dirs:
                seen_dirs.add(head)
                children.append({"name": head, "is_dir": True,
                                 "uri": make_zip_uri(zip_path, prefix + head + "/")})
    try:
        children.sort(key=lambda x: natural_key(x["name"]))  # 既存の natural_key を利用
    except Exception:
        children.sort(key=lambda x: x["name"].lower())
    return children

def vfs_is_dir(path_or_uri: str) -> bool:
    import os
    if is_zip_uri(path_or_uri):
        _, inner = parse_zip_uri(path_or_uri)
        # zipルート or zip内フォルダ
        if inner == "" or inner.endswith("/"):
            return True
        # zip内の zip/CBZ はフォルダ扱い
        if _is_zip_like_name(inner):
            return True
        return False

    if os.path.isdir(path_or_uri):
        return True
    if is_archive_file(path_or_uri):  # zip/rar/cbz/cbr 自体を“フォルダ扱い”
        return True
    return False


def vfs_is_file(path_or_uri: str) -> bool:
    import os
    if is_zip_uri(path_or_uri):
        _, inner = parse_zip_uri(path_or_uri)
        if not inner or inner.endswith("/"):
            return False
        # zip内の zip/CBZ はフォルダ扱いなのでファイルとはみなさない
        if _is_zip_like_name(inner):
            return False
        return True
    return os.path.isfile(path_or_uri)

def vfs_listdir(dir_path_or_uri: str):
    """
    戻り値: list[{ 'name': str, 'is_dir': bool, 'uri': str }]
      - 物理フォルダ：通常列挙、zip/CBZは“フォルダ扱い”で uri=zip://. として返す
      - zip://.    ：仮想フォルダを1階層列挙
      - zipファイルパス：zipの最上位(= zip://<path>!/ )を列挙
    """
    import os
    if is_zip_uri(dir_path_or_uri):
        zp, inner = parse_zip_uri(dir_path_or_uri)

        # zip内の「xxx.zip / xxx.cbz」を開こうとしている場合だけ memzip 展開
        if inner and not inner.endswith("/") and _is_zip_like_name(inner):
            mem_id = _register_mem_zip(zp, inner)
            return _zip_list_children(mem_id, "")

        # それ以外（zipルート or zip内フォルダ）
        return _zip_list_children(zp, inner)

    if os.path.isdir(dir_path_or_uri):
        items = []
        try:
            names = sorted(os.listdir(dir_path_or_uri), key=natural_key)
        except Exception:
            # ★ ソートキーで失敗したら、キー無しの通常ソートにフォールバック
            try:
                names = sorted(os.listdir(dir_path_or_uri))
            except Exception:
                names = []
        for name in names:
            p = os.path.join(dir_path_or_uri, name)
            if os.path.isdir(p):
                items.append({"name": name, "is_dir": True, "uri": os.path.abspath(p)})
            elif is_archive_file(p):
                items.append({"name": name, "is_dir": True, "uri": make_zip_uri(p, "")})
            else:
                items.append({"name": name, "is_dir": False, "uri": os.path.abspath(p)})
        return items

    if is_archive_file(dir_path_or_uri):
        import os
        return _zip_list_children(os.path.normpath(dir_path_or_uri), "")

def vfs_parent(path_or_uri: str) -> str | None:
    import os
    if is_zip_uri(path_or_uri):
        zp, inner = parse_zip_uri(path_or_uri)

        # memzip のルート → 外側 zip 内の zip エントリへ戻る
        if isinstance(zp, str) and zp.startswith("memzip:"):
            meta = _MEM_ZIP_META.get(zp)
            if inner == "" or inner == "/":
                if not meta:
                    return None
                outer = meta["outer"]
                entry = meta["inner"]
                return make_zip_uri(outer, entry)

            inner2 = inner.strip("/")
            parent_inner = "/".join(inner2.split("/")[:-1])
            return make_zip_uri(zp, (parent_inner + "/") if parent_inner else "")

        # 通常の zip://
        if inner == "":
            return os.path.dirname(zp) or None
        inner2 = inner.strip("/")
        parent_inner = "/".join(inner2.split("/")[:-1])
        return make_zip_uri(zp, (parent_inner + "/") if parent_inner else "")
    return os.path.dirname(path_or_uri) or None

def open_bytes_any(path_or_uri: str) -> bytes:
    if is_zip_uri(path_or_uri):
        zp, inner = parse_zip_uri(path_or_uri)
        zf = _open_zip_cached(zp)
        return zf.read(inner)  # メモリに読み出し（temp展開なし）
    with open(path_or_uri, "rb") as f:
        return f.read()

def open_image_any(path_or_uri):
    from PIL import Image, ImageOps
    if is_zip_uri(path_or_uri):
        zp, inner = parse_zip_uri(path_or_uri)
        zf = _open_zip_cached(zp)
        # ★ ケース非依存に解決（ヒットすれば正しいオリジナル名が戻る）
        inner2 = _zip_resolve_inner(zp, inner)
        try:
            with zf.open(inner2, "r") as f:   # ストリーミングで読む（中間コピー削減）
                im = Image.open(f)
                im.load()                     # ここでデコードを完了させてクローズ可に
        except KeyError:
            # どうしても見つからない場合は元の名前で再挑戦（念のため）
            with zf.open(inner, "r") as f:
                im = Image.open(f); im.load()
        return ImageOps.exif_transpose(im)
    else:
        im = Image.open(path_or_uri)
        return ImageOps.exif_transpose(im)

def make_fixed_thumbnail_any(path_or_uri, thumb_size=(80, 120)):
    # ディレクトリ（物理 or zip仮想）は単色グレー（フォルダは別処理で描くため）
    if vfs_is_dir(path_or_uri):
        from PIL import Image
        return Image.new("RGB", thumb_size, (235, 235, 235))

    # 画像のみサムネ化（非画像はグレー返し）
    name = path_or_uri
    if is_zip_uri(path_or_uri):
        _, inner = parse_zip_uri(path_or_uri)
        name = inner
    if not is_image_name(name):
        from PIL import Image
        return Image.new("RGB", thumb_size, (200, 200, 200))

    try:
        from PIL import Image, ImageOps

        # まずは「軽い読み方」で開く（フル解像度の EXIF 回転は行わない）
        if is_zip_uri(path_or_uri):
            zp, inner = parse_zip_uri(path_or_uri)
            zf = _open_zip_cached(zp)
            # ケース非依存に解決（見つかればオリジナル名に）
            inner2 = _zip_resolve_inner(zp, inner)
            try:
                f = zf.open(inner2, "r")
            except KeyError:
                # 念のため元名でもう一度
                f = zf.open(inner, "r")
            with f:
                im = Image.open(f)
                # JPEG の場合は draft で読み取り負荷を軽減（効くフォーマットだけ）
                if getattr(im, "format", None) == "JPEG":
                    try:
                        im.draft("RGB", thumb_size)
                    except Exception:
                        pass
                im.load()
        else:
            im = Image.open(path_or_uri)
            if getattr(im, "format", None) == "JPEG":
                try:
                    im.draft("RGB", thumb_size)
                except Exception:
                    pass
            im.load()

        # カラーモードを統一（必要なら）
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        # ★ 先に縮小してから EXIF 回転（小さい画像を回転するので軽い）
        im.thumbnail(thumb_size, Image.BILINEAR)
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            # EXIF が壊れている／無いなど、失敗してもサムネ生成は続行
            pass

        # 中央に配置するキャンバスを作成
        canvas = Image.new("RGB", thumb_size, (240, 240, 240))
        off = ((thumb_size[0] - im.width) // 2, (thumb_size[1] - im.height) // 2)
        canvas.paste(im, off)
        return canvas
    except Exception as e:
        log_debug(f"[thumb] error (lite): {path_or_uri}: {e}")
        from PIL import Image
        return Image.new("RGB", thumb_size, (90, 90, 90))

def norm_vpath(p: str) -> str:
    """ファイルパス/zip URI 両対応の比較キー"""
    import os
    if is_zip_uri(p):
        # 大文字小文字だけ吸収してそのまま
        return p.lower().replace("\\", "/")
    return os.path.normcase(os.path.abspath(p or ""))

def vfs_display_name(p: str, is_dir: bool) -> str:
    """表示名（zip:// のときは内側のベース名／rootはzip名）"""
    import os
    if is_zip_uri(p):
        zp, inner = parse_zip_uri(p)
        if inner == "":
            return f"{os.path.basename(zp)}"  # zipのファイル名
        inner = inner.rstrip("/")
        return os.path.basename(inner) or os.path.basename(zp)
    try:
        return os.path.basename(str(p).rstrip(os.sep))
    except Exception:
        return str(p)

def _enable_dark_titlebar(win):
    """Windowsのタイトルバーをダークにする。失敗しても無視（他環境対応）"""
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = int(win.winId())  # ハンドル確保
        value = ctypes.c_int(1)

        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                ctypes.c_uint(attr),
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
            if res == 0:
                break
    except Exception:
        pass

def _qobject_alive(obj) -> bool:
    """PyQtのQObjectがまだ生きているか（破棄済み参照でないか）を安全に判定"""
    if obj is None:
        return False
    # sip があれば “C++側が削除済み” を検出
    try:
        from sip import isdeleted
        if isdeleted(obj):
            return False
    except Exception:
        pass
    # 参照アクセステスト（RuntimeError が出たら死んでいる）
    try:
        _ = obj.objectName()
    except RuntimeError:
        return False
    return True

from PyQt6 import QtCore

def _safe_qrect(val, *, fmt="xywh"):
    """
    val: None / QRect / [x,y,w,h] / (l,t,r,b)
    fmt: "xywh" または "ltrb"
    不正なら空の QRect() を返す（例外にしない）
    """
    if isinstance(val, QtCore.QRect):
        return QtCore.QRect(val)
    if not isinstance(val, (list, tuple)):
        return QtCore.QRect()
    if any(v is None for v in val):
        return QtCore.QRect()

    nums = list(map(int, list(val)[:4]))  # 4つ以上来ても先頭4つだけ
    if len(nums) != 4:
        return QtCore.QRect()

    if fmt == "ltrb":
        l, t, r, b = nums
        return QtCore.QRect(l, t, max(0, r - l), max(0, b - t))

    x, y, w, h = nums
    if w <= 0 or h <= 0:
        return QtCore.QRect()
    return QtCore.QRect(x, y, w, h)

def natural_key(s: str):
    """
    自然順ソート用キー（グローバル版）:
    - 連続する10進数字(0-9 / 全角)だけ数値として扱う
    - それ以外（丸数字「②」などの isdigit だが isdecimal でないもの）は文字列として扱う
    """
    import os, re

    b = os.path.basename(s) if isinstance(s, str) else str(s)
    parts = re.split(r"(\d+)", b)

    key = []
    for t in parts:
        if not t:
            continue
        # ★ isdigit() ではなく isdecimal() を使う
        if t.isdecimal():
            try:
                key.append(int(t))
            except Exception:
                # 万一 int 変換できなかったときの保険
                key.append(t.lower())
        else:
            key.append(t.lower())
    return key

class CustomListView(QtWidgets.QListView):
    def __init__(self, mainwin=None):
        super().__init__(mainwin)
        self.mainwin = mainwin
        # …既存の初期化…

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key  = event.key()
        mods = event.modifiers()

        # --- まず Ctrl + ←/→ を自前で処理して終了 ---
        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == QtCore.Qt.Key.Key_Left:
                if self.mainwin and hasattr(self.mainwin, "_move_thumb_focus"):
                    if hasattr(self.mainwin, "_prepare_preserve_for_nav"):
                        self.mainwin._prepare_preserve_for_nav()  # ★ 追加
                    self.mainwin._move_thumb_focus(-1)
                event.accept(); return
            if key == QtCore.Qt.Key.Key_Right:
                if self.mainwin and hasattr(self.mainwin, "_move_thumb_focus"):
                    if hasattr(self.mainwin, "_prepare_preserve_for_nav"):
                        self.mainwin._prepare_preserve_for_nav()  # ★ 追加
                    self.mainwin._move_thumb_focus(+1)
                event.accept(); return
    
        # --- 素の ←/→ は従来通り：リスト既定の移動を無効化 ---
        if key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            event.ignore()
            return

        # その他は通常処理
        super().keyPressEvent(event)

# --- ズーム倍率表示ラベル ---
class ZoomLabel(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        # ★ 自分専用の名前をつける
        self.setObjectName("ZoomLabel")

        # ★ このクラスの QLabel だけにスタイルを当てる
        self.setStyleSheet("""
            QLabel#ZoomLabel {
                color: white;
                background-color: rgba(20, 20, 20, 140);
                border-radius: 10px;
                padding: 4px 14px;
                font-weight: bold;
                font-size: 18px;
            }
        """)

        self.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(1.0)

        self.opacity_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self.opacity_anim.setDuration(500)
        self.opacity_anim.setStartValue(1.0)
        self.opacity_anim.setEndValue(0.0)

        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.timeout.connect(self._start_fade)
        self.opacity_anim.finished.connect(self.hide)

    def show_zoom(self, value: float):
        percent = int(round(value * 100))
        self.setText(f"{percent}%")
        self.adjustSize()
        self.move(12, 12)

        # 直前のアニメを止めて、完全不透明に戻す
        self.opacity_anim.stop()
        self._opacity_effect.setOpacity(1.0)         # ← setWindowOpacity は使わない
        self.show()
        self.raise_()
        self._fade_timer.start(2000)  # 2秒後にフェード開始

    def _start_fade(self):
        start = float(self._opacity_effect.opacity())
        self.opacity_anim.stop()
        self.opacity_anim.setStartValue(start)
        self.opacity_anim.setEndValue(0.0)
        self.opacity_anim.setDuration(700)
        self.opacity_anim.start()

class DualRotateButton(QtWidgets.QWidget):
    """左右で別の動作をする 90度回転ボタン"""

    leftClicked = QtCore.pyqtSignal()   # 左側クリック → 左回転
    rightClicked = QtCore.pyqtSignal()  # 右側クリック → 右回転

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)  # ボタン押してなくてもマウス移動を拾う

        self._hover = False
        self._hover_side = None      # "left" / "right" / None
        self._pressed = False
        self._pressed_side = None    # "left" / "right" / None

    def sizeHint(self):
        return QtCore.QSize(72, 72)

    # ---- ヘルパ ----
    def _event_pos_x(self, e):
        # Qt6 / Qt5 両対応
        if hasattr(e, "position"):
            return float(e.position().x())
        return float(e.pos().x())

    def _side_from_x(self, x: float) -> str:
        return "left" if x < self.width() / 2 else "right"

    # ---- マウス・ホバー系 ----
    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self._hover_side = None
        self._pressed = False
        self._pressed_side = None
        self.update()

    def mouseMoveEvent(self, e):
        x = self._event_pos_x(e)
        side = self._side_from_x(x)
        if side != self._hover_side:
            self._hover_side = side
            self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            x = self._event_pos_x(e)
            self._pressed = True
            self._pressed_side = self._side_from_x(x)
            # 押した瞬間もその側を光らせたいので hover_side も更新
            self._hover_side = self._pressed_side
            self.update()

    def mouseReleaseEvent(self, e):
        if not self._pressed or e.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        x = self._event_pos_x(e)
        side = self._side_from_x(x)

        if side == self._pressed_side:
            if side == "left":
                self.leftClicked.emit()
            else:
                self.rightClicked.emit()

        self._pressed = False
        self._pressed_side = None
        self.update()

    # ---- 描画 ----
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 10

        # ベース背景色
        base_bg = QtGui.QColor("#2b2b2b")

        # 枠の色を状態で変える（全体）
        if self._pressed:
            border_color = QtGui.QColor("#f4e38a")   # 押しているとき
        elif self._hover:
            border_color = QtGui.QColor("#ffe58a")   # ← ホバー中をもっと明るく
        else:
            border_color = QtGui.QColor("#e6c15a")   # 通常

        border_pen = QtGui.QPen(border_color)
        border_pen.setWidth(2)
        p.setPen(border_pen)
        p.setBrush(base_bg)
        p.drawRoundedRect(rect, radius, radius)

        # ★ ホバー／押下している側だけ、半分を明るくする（背景だけ）
        if self._hover and self._hover_side in ("left", "right"):
            if self._pressed and self._pressed_side == self._hover_side:
                hl_bg = QtGui.QColor("#222222")    # 押してる側はちょい暗く（そのまま）
            else:
                hl_bg = QtGui.QColor("#454545")    # ← ホバー側をもう一段明るく

            # 枠線の“内側”だけを塗るように、ひと回り小さい領域を使う
            inner = rect.adjusted(2, 2, -2, -2)   # 2pxぶん内側 = border幅と合わせる

            half_rect = QtCore.QRect(inner)
            if self._hover_side == "left":
                half_rect.setWidth(inner.width() // 2)
            else:
                half_rect.setLeft(inner.left() + inner.width() // 2)

            p.save()
            p.setClipRect(half_rect)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(hl_bg)
            p.drawRoundedRect(inner, radius - 2, radius - 2)
            p.restore()

        # 中央の仕切り線
        mid_x = rect.center().x()
        line_pen = QtGui.QPen(QtGui.QColor("#3a3a3a"))
        line_pen.setWidth(1)
        p.setPen(line_pen)
        p.drawLine(mid_x, rect.top() + 4, mid_x, rect.bottom() - 4)

        # 文字描画（左右それぞれの2行を中央の仕切り線寄りに寄せる）
        p.setPen(QtGui.QColor("#e0e0e0"))
        font = self.font()
        font.setPointSize(11)
        p.setFont(font)

        fm = QtGui.QFontMetrics(font)

        # 文字を描く“内側”の領域
        inner_text = rect.adjusted(4, 4, -4, -4)

        line_h = fm.height()

        # 上下2行分の高さ：ちゃんと「2行＋少しの行間」にする
        gap = max(2, line_h // 8)          # ← 行間。もっと空けたければ //4, //3 に
        total_h = 2 * line_h + gap
        top_start = inner_text.center().y() - total_h // 2

        top_y = top_start
        bottom_y = top_y + line_h + gap

        # ★ 横方向：中央の仕切り線寄りに細い列を作る
        mid_x = inner_text.center().x()
        col_w = int(inner_text.width() * 0.30)  # 列の幅。広すぎ/狭すぎならここを調整

        # 左側：仕切り線の「すぐ左」
        left_x = mid_x - col_w
        # 右側：仕切り線の「すぐ右」
        right_x = mid_x

        # 左側2行
        left_top_rect = QtCore.QRect(left_x, int(top_y), col_w, line_h)
        left_bottom_rect = QtCore.QRect(left_x, int(bottom_y), col_w, line_h)

        # 右側2行
        right_top_rect = QtCore.QRect(right_x, int(top_y), col_w, line_h)
        right_bottom_rect = QtCore.QRect(right_x, int(bottom_y), col_w, line_h)

        p.drawText(left_top_rect,     QtCore.Qt.AlignmentFlag.AlignCenter, "左")
        p.drawText(right_top_rect,    QtCore.Qt.AlignmentFlag.AlignCenter, "右")
        p.drawText(left_bottom_rect,  QtCore.Qt.AlignmentFlag.AlignCenter, "回")
        p.drawText(right_bottom_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "転")

        p.end()

class SquareLabel(QtWidgets.QLabel):
    def __init__(self, parent=None, min_side=256, base_side=512):
        super().__init__(parent)
        self._min_side = int(min_side)
        self._base_side = int(base_side)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

        self.setMinimumSize(self._min_side, self._min_side)
        self.setScaledContents(False)

        # 追加: 再入防止フラグ
        self._resize_lock = False

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return max(self._min_side, int(w))

    def sizeHint(self) -> QtCore.QSize:
        s = max(self._min_side, self._base_side)
        return QtCore.QSize(s, s)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._min_side, self._min_side)

    def _apply_fixed_height(self, h: int):
        # 遅延適用: ここで初めて固定高を入れる（1フレーム後）
        try:
            self.setFixedHeight(h)
        finally:
            self._resize_lock = False

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        # ここでは“計算だけ”。サイズ制限の変更は次フレームへ遅延
        super().resizeEvent(e)
        if self._resize_lock:
            return
        w = max(self._min_side, self.width())
        h_target = self.heightForWidth(w)

        # 1px 単位の境界振動を避けるための閾値
        if abs(self.height() - h_target) <= 1:
            return

        self._resize_lock = True
        QtCore.QTimer.singleShot(0, lambda: self._apply_fixed_height(h_target))

class SuccessLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # 外観
        f = QtGui.QFont(); f.setPointSize(14); f.setBold(True)
        self.setFont(f)
        self.setStyleSheet("color: green; background: #ffffffcc; border-radius: 8px; padding: 6px;")
        self.hide()  # 初期は非表示

        # 統一タイマー（__init__では起動しない）
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def flash(self, pos: QtCore.QPoint, message: str, ok: bool = True, timeout: int = 1500):
        # 文言と色
        self.setText(message)
        self.setStyleSheet(
            f"color: {'green' if ok else '#d32f2f'}; background: #ffffffcc; border-radius: 8px; padding: 6px;"
        )

        # 位置→表示（毎回同じ経路）
        self.adjustSize()
        self.move(pos)
        self.show()
        self.raise_()
        QtCore.QTimer.singleShot(0, self.raise_)  # イベントループ復帰後にもう一押し

        # タイマーをここでだけ動かす（毎回同じ）
        self._timer.stop()
        if timeout and timeout > 0:
            self._timer.start(timeout)

class ColorChipButton(QtWidgets.QToolButton):
    """左クリック=この色を適用 / 右クリック=このチップの色を編集"""
    colorClicked = QtCore.pyqtSignal(QtGui.QColor)  # 左クリックで発火（色を適用したい側に接続）
    colorEdited  = QtCore.pyqtSignal(QtGui.QColor)  # 右クリックで編集した色が決まったら発火（保存など）

    def __init__(self, initial_color, tooltip="",
                 parent=None, w=20, h=14,
                 edit_title="このチップの色を編集",
                 group="view"):   # ★ 追加: "view" / "preview" などグループ名
        super().__init__(parent)
        self._color = QtGui.QColor(initial_color)
        self._edit_title = edit_title
        self._group = group          # ★ どのグループか覚える
        self._selected = False       # ★ 選択中フラグ

        self.setToolTip(tooltip)
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(w, h)
        self._apply_style()

    def set_edit_title(self, text: str):
        self._edit_title = text

    # ★ 追加: 選択状態の setter / getter
    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_style()

    def is_selected(self) -> bool:
        return self._selected

    def _apply_style(self):
        c = self._color

        # 色名（透過あり / なし）
        if c.alpha() < 255:
            fmt = QtGui.QColor.NameFormat.HexArgb
            name = c.name(fmt)
        else:
            name = c.name()

        # --- 非選択時の枠 ---
        normal_base_border    = "1px solid #666"
        normal_hover_border   = "2px solid rgba(255,255,255,0.8)"
        normal_pressed_border = "2px solid rgba(255,255,255,1.0)"

        # --- グループごとの「選択時の枠色」 ---
        if self._group == "preview":
            # プレビュー用：プレビュー枠と同じ #4b94d9 ベース
            sel_base_border    = "2px solid #4b94d9"                 # preview_area と同じ
            sel_hover_border   = "2px solid rgba(75, 148, 217, 0.9)" # ちょっとだけ明るく
            sel_pressed_border = "2px solid rgba(255, 255, 255, 1.0)"
        else:
            # 画像表示用：メイン表示枠と同じ #f28524 ベース
            sel_base_border    = "2px solid #f28524"                 # _apply_view_bg と同じ
            sel_hover_border   = "2px solid rgba(242, 133, 36, 0.9)" # ちょっとだけ明るく
            sel_pressed_border = "2px solid rgba(255, 255, 255, 1.0)"

        # 選択状態に応じて最終的な枠を決定
        if self._selected:
            base_border    = sel_base_border
            hover_border   = sel_hover_border
            pressed_border = sel_pressed_border
        else:
            base_border    = normal_base_border
            hover_border   = normal_hover_border
            pressed_border = normal_pressed_border

        self.setStyleSheet(
            "QToolButton {"
            f" background:{name};"
            f" border:{base_border};"
            " border-radius:3px;"
            " padding:0;"
            "}"
            "QToolButton:hover {"
            f" background:{name};"
            f" border:{hover_border};"
            "}"
            "QToolButton:pressed {"
            f" background:{name};"
            f" border:{pressed_border};"
            "}"
        )

    def set_color(self, color: QtGui.QColor):
        self._color = QtGui.QColor(color)
        self._apply_style()

    def color(self) -> QtGui.QColor:
        return QtGui.QColor(self._color)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.RightButton:
            opts = (QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
                    QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
            c = QtWidgets.QColorDialog.getColor(
                self._color, self, self._edit_title, options=opts)
            if c.isValid():
                self.set_color(c)
                self.colorEdited.emit(QtGui.QColor(c))
            return
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.colorClicked.emit(QtGui.QColor(self._color))
            return
        super().mousePressEvent(e)

class ActionPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, pos=QtCore.QPoint(0, 0),
                 on_save=None, on_pin=None, on_cancel=None, on_adjust=None,
                 is_fixed: bool = False):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._is_fixed = bool(is_fixed)

        # ▼▼ ドラッグ用の状態（パネル全体を掴んで動かす） ▼▼
        self._dragging = False
        self._drag_start_global = QtCore.QPoint()
        self._drag_start_pos = QtCore.QPoint()

        # --- ボタン類 ---
        btn_save   = QtWidgets.QPushButton("保存")
        btn_pin    = QtWidgets.QPushButton()  # ← テキストは後で決める
        btn_adjust = QtWidgets.QPushButton("調整")
        btn_cancel = QtWidgets.QPushButton("取消")

        # フォント
        font = btn_save.font()
        font.setPointSize(11)
        for b in (btn_save, btn_pin, btn_adjust, btn_cancel):
            b.setFont(font)
            b.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            # ★ 矢印キー操作のために、ボタンがフォーカスを奪わないようにする
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)


        # スタイル
        btn_save.setStyleSheet("""
            QPushButton {background: #4caf50; color: white; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #388e3c;}
        """)
        btn_pin.setStyleSheet("""
            QPushButton {background: #1e78ff; color: white; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #1a6bed;}
        """)
        btn_adjust.setStyleSheet("""
            QPushButton {background: #607d8b; color: white; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #546e7a;}
            QPushButton:disabled {background: #9aa7ad; color: rgba(255,255,255,0.6);}
        """)
        btn_cancel.setStyleSheet("""
            QPushButton {background: #e0e0e0; color: #333; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #bdbdbd;}
        """)

        # ルートは縦。上に nudge ホスト、下に 2x2 ボタン
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # --- 下段：従来の 2x2 ボタン ---
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.addWidget(btn_save,   0, 0)
        grid.addWidget(btn_pin,    0, 1)
        grid.addWidget(btn_adjust, 1, 0)
        grid.addWidget(btn_cancel, 1, 1)

        root.addLayout(grid, 0)

        # pin（固定/戻る）は常に表示。on_pin が無ければ disable にする（レイアウトは維持）
        if on_pin is None:
            btn_pin.setEnabled(False)

        # 全ボタンの高さ・最小幅をそろえる
        buttons = [btn_save, btn_pin, btn_adjust, btn_cancel]
        ref_h = max(b.sizeHint().height() for b in buttons)
        ref_w = max(b.sizeHint().width()  for b in buttons)
        for b in buttons:
            b.setFixedHeight(ref_h)
            b.setMinimumWidth(ref_w)

        # --- ショートカット/シグナル ---
        if on_save:   btn_save.clicked.connect(on_save)
        if on_pin:    btn_pin.clicked.connect(on_pin)
        if on_adjust: btn_adjust.clicked.connect(on_adjust)
        if on_cancel: btn_cancel.clicked.connect(on_cancel)

        self.sc_save = QShortcut(QKeySequence("S"), self)
        self.sc_save.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if on_save:
            self.sc_save.activated.connect(on_save)

        self.sc_pin = QShortcut(QKeySequence("F"), self)
        self.sc_pin.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if on_pin:
            self.sc_pin.activated.connect(on_pin)
        else:
            self.sc_pin.setEnabled(False)

        self.sc_adjust = QShortcut(QKeySequence("A"), self)
        self.sc_adjust.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if on_adjust:
            self.sc_adjust.activated.connect(on_adjust)

        self.sc_cancel = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self.sc_cancel.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if on_cancel:
            self.sc_cancel.activated.connect(on_cancel)

        # 参照保持
        self.btn_save   = btn_save
        self.btn_pin    = btn_pin
        self.btn_adjust = btn_adjust

        # ラベル（テキスト）確定
        btn_save.setText("保存 (S)")
        btn_cancel.setText("取消 (Esc)")
        btn_adjust.setText("調整 (A)")
        btn_pin.setText("戻る (F)" if self._is_fixed else "固定 (F)")

        # 最初の表示
        self.adjustSize()
        self.move(pos)
        self.show()

        # ===== ツールチップ（自作）セットアップ =====
        self._softtip = _TipPopup(None)

        def _adjust_tip_text():
            return "矩形を微調整 (A)" if self.btn_adjust.isEnabled() else "32/64pxモードでは微調整は無効です"

        class _TipFilter(QtCore.QObject):
            def __init__(self, softtip, text_fn, parent=None):
                super().__init__(parent)
                self._softtip = softtip
                self._text_fn = text_fn
                self._active = False  # このウィジェット上で表示中か

            # どんなイベントでもグローバル座標を安全に取る
            def _event_global_pos(self, obj, ev) -> QtCore.QPoint:
                # QHelpEvent (ToolTip)
                if hasattr(ev, "globalPos"):
                    try:
                        return ev.globalPos()
                    except Exception:
                        pass
                # QMouseEvent (MouseMove など)
                if hasattr(ev, "globalPosition"):
                    gp = ev.globalPosition()  # QPointF
                    return QtCore.QPoint(int(gp.x()), int(gp.y()))
                if hasattr(ev, "position"):
                    # QHoverEvent / QMouseEvent でも position() はある
                    p = ev.position()
                    try:
                        p = p.toPoint()  # QPointF → QPoint
                    except Exception:
                        p = QtCore.QPoint(int(p.x()), int(p.y()))
                    return obj.mapToGlobal(p)
                # 最後の保険
                return QtGui.QCursor.pos()

            def eventFilter(self, obj, ev):
                t = ev.type()
                if t == QtCore.QEvent.Type.ToolTip:
                    # 1回だけ表示し、以降は無視（点滅防止）
                    if not self._active:
                        text = self._text_fn() if callable(self._text_fn) else ""
                        if text:
                            gpos = self._event_global_pos(obj, ev)
                            self._softtip.show_text(text, gpos)
                            self._active = True
                    return True  # ネイティブ抑止

                elif t in (QtCore.QEvent.Type.Leave,
                        QtCore.QEvent.Type.Hide,
                        QtCore.QEvent.Type.WindowDeactivate):
                    self._softtip.hide()
                    self._active = False
                    return False

                elif t == QtCore.QEvent.Type.MouseMove:
                    # 位置だけ追従（文面は固定）
                    if self._active:
                        gpos = self._event_global_pos(obj, ev)
                        self._softtip.show_text(self._text_fn(), gpos)
                    return False

                return False

        # 取り付け（ネイティブtoolTipは空にして抑止）
        self._adj_filter = _TipFilter(self._softtip, _adjust_tip_text, self.btn_adjust)
        self.btn_adjust.installEventFilter(self._adj_filter)
        self.btn_adjust.setToolTip("")

    def set_adjusting(self, on: bool) -> None:
        """調整モードのON/OFFに応じてボタンラベルを切り替える。"""
        b = getattr(self, "btn_adjust", None)
        if b is not None:
            b.setText("戻る  (A)" if on else "調整 (A)")

    def enable_adjust(self, enabled: bool) -> None:
        ok = bool(enabled)
        try: self.btn_adjust.setEnabled(ok)
        except Exception: pass
        try:
            if hasattr(self, "sc_adjust"):
                self.sc_adjust.setEnabled(ok)
        except Exception:
            pass

    def mousePressEvent(self, ev):
        """土台（ボタン以外）を左クリックしたときだけドラッグ開始。"""
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # クリック位置に子ウィジェット（ボタンなど）があるか確認
            child = self.childAt(ev.pos())
            if child is None:
                # 何もない＝黒い余白部分 → ドラッグ開始
                self._dragging = True

                # Qt6: globalPosition() 優先
                if hasattr(ev, "globalPosition"):
                    gp = ev.globalPosition()
                    try:
                        gp = gp.toPoint()
                    except Exception:
                        gp = QtCore.QPoint(int(gp.x()), int(gp.y()))
                else:
                    gp = ev.globalPos()

                self._drag_start_global = gp
                self._drag_start_pos = self.pos()
                ev.accept()
                return
        # それ以外は通常処理（ボタンのクリックなど）
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging and (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            if hasattr(ev, "globalPosition"):
                gp = ev.globalPosition()
                try:
                    gp = gp.toPoint()
                except Exception:
                    gp = QtCore.QPoint(int(gp.x()), int(gp.y()))
            else:
                gp = ev.globalPos()

            delta = gp - self._drag_start_global
            self.move(self._drag_start_pos + delta)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            # ユーザーがアクションパネルをドラッグしたので、
            # 以降は矩形移動に自動追従させない
            try:
                parent = self.parent()
                mainwin = getattr(parent, "mainwin", None)
                if mainwin is not None:
                    mainwin._action_panel_detached = True
            except Exception:
                pass

            ev.accept()
            return
        super().mouseReleaseEvent(ev)

class SoftTip(QtWidgets.QFrame):
    """点滅しない・再表示ですぐ消えないカスタムツールチップ（黒固定・角丸内に収める）"""
    def __init__(self, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._radius = 8
        self._pad_h, self._pad_v = 9, 6
        self._max_w = 360

        self._lab = QtWidgets.QLabel(self)
        self._lab.setWordWrap(True)
        self._lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        f = self._lab.font(); f.setPointSizeF(max(8.0, f.pointSizeF() - 0.5))
        self._lab.setFont(f)
        self._lab.setStyleSheet("QLabel { color: white; }")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(self._pad_h, self._pad_v, self._pad_h, self._pad_v)
        lay.addWidget(self._lab)

        self._life = QtCore.QTimer(self); self._life.setSingleShot(True)
        self._life.timeout.connect(self._fade_out)
        self._grace_hide = QtCore.QTimer(self); self._grace_hide.setSingleShot(True)
        self._grace_hide.timeout.connect(lambda: self.hide() if not self._pinned else None)

        self._fade = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(140)

        self._host = None
        self._pinned = False
        self.setWindowOpacity(0.0)

    def show_text(self, host: QtWidgets.QWidget, text: str, gpos: QtCore.QPoint, msec: int = 2000):
        if not text:
            self.hide(); return
        self._life.stop(); self._grace_hide.stop(); self._fade.stop()

        same_host = (host is not None and host is self._host and self.isVisible())
        self._host = host
        self._pinned = True
        QtCore.QTimer.singleShot(120, self._unpin)

        self._lab.setMaximumWidth(self._max_w)
        self._lab.setText(text)
        self.adjustSize()
        if self.width() < 1 or self.height() < 1:
            return

        scr = QtWidgets.QApplication.screenAt(gpos) or QtWidgets.QApplication.primaryScreen()
        geo = (scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080))
        x = min(max(gpos.x() + 10, geo.left()),  geo.right()  - self.width())
        y = min(max(gpos.y() + 12, geo.top()),   geo.bottom() - self.height())
        self.move(int(x), int(y))

        if same_host:
            self.setWindowOpacity(1.0)
            self.raise_()
        else:
            self.setWindowOpacity(0.0)
            self.show(); self.raise_()
            self._fade.setStartValue(0.0); self._fade.setEndValue(1.0); self._fade.start()

        self._life.start(max(900, msec))

    def nudge_hide(self, delay_ms: int = 140):
        if self._pinned:
            return
        self._grace_hide.start(max(60, delay_ms))

    def _fade_out(self):
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.hide)
        self._fade.start()

    def _unpin(self):
        self._pinned = False

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        path = QtGui.QPainterPath(); path.addRoundedRect(QtCore.QRectF(r), self._radius, self._radius)
        p.fillPath(path, QtGui.QColor(0, 0, 0, 220))   # 背景は常に黒
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 40)); pen.setCosmetic(True); p.setPen(pen)
        p.drawPath(path)

class NudgePanel(QtWidgets.QFrame):
    """ActionPanel 内に埋め込む微調整パネル（カウンタ + 比率固定）"""
    def __init__(self, mainwin, nudge_cb):
        super().__init__(mainwin)
        self.mainwin = mainwin
        self.nudge_cb = nudge_cb  # def nudge_edge(side:str, delta:int)->int|None を想定
        self.setObjectName("NudgePanel")

        # ---- 角丸描画の設定（ここを変えれば色や角丸半径を調整可能）----
        self._radius = 10
        self._border_w = 2
        self._bg_color = QtGui.QColor("#111")       # 以前の background:#111
        self._border_color = QtGui.QColor("#f28524")  # 以前の border: ... #f28524

        # QSSの背景・枠は使わずに、子ウィジェットだけスタイル指定
        # ※ パネル本体（#NudgePanel）の background / border / border-radius は指定しない！
        self.setStyleSheet("""
        QToolButton { background:#222; color:#ddd; border:2px solid #f28524; border-radius:6px; padding:2px 8px; }
        QToolButton:hover { background:#2a2a2a; }
        QToolButton:disabled { color: rgba(255,255,255,0.45); border-color: rgba(242,133,36,0.35); }
        QLabel { color:#ddd; }
        QLabel:disabled { color: rgba(255,255,255,0.45); }
        """)

        # 背景は自前で塗るために透過を有効化（にじみ防止）
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)  # 残してOK（見た目は自前描画）

        self._val_labels: dict[str, QtWidgets.QLabel] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        def mk_row(jp_label: str, side: str) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget(self); w.setObjectName(f"row_{side}")
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)  # ← 8 → 4 に詰める

            btn_l = QtWidgets.QToolButton(w); btn_l.setText("▽")
            btn_r = QtWidgets.QToolButton(w); btn_r.setText("△")
            for b in (btn_l, btn_r):
                b.setAutoRepeat(True); b.setAutoRepeatDelay(250); b.setAutoRepeatInterval(30)

            lab_txt = QtWidgets.QLabel(jp_label, w)
            lab_txt.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
            lab_txt.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
            lab_txt.setFixedWidth(20)

            lab_val = QtWidgets.QLabel("0", w)
            lab_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lab_val.setFixedWidth(26)
            self._val_labels[side] = lab_val

            def apply(outward: bool):
                if not callable(self.nudge_cb):
                    return
                if side in ("top", "left"):
                    delta = -1 if outward else +1
                else:
                    delta = +1 if outward else -1
                applied = 0
                try:
                    res = self.nudge_cb(side, delta)
                    applied = int(res) if res is not None else (1 if delta != 0 else 0)
                except Exception:
                    applied = (1 if delta != 0 else 0)
                if applied != 0:
                    try:
                        cur = int(lab_val.text())
                    except ValueError:
                        cur = 0
                    lab_val.setText(str(cur + (+1 if outward else -1)))

            btn_l.clicked.connect(lambda: apply(False))
            btn_r.clicked.connect(lambda: apply(True))

            h.addWidget(btn_l)
            h.addWidget(lab_txt)
            h.addWidget(lab_val)
            h.addWidget(btn_r)
            return w

        # 4行
        root.addWidget(mk_row("上",  "top"))
        root.addWidget(mk_row("下",  "bottom"))
        root.addWidget(mk_row("左",  "left"))
        root.addWidget(mk_row("右",  "right"))

        # ---- 比率固定トグル + 基準表示（縦積み）----
        ratio_box = QtWidgets.QWidget(self)
        vb = QtWidgets.QVBoxLayout(ratio_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(3)

        self._btn_ratio = QtWidgets.QToolButton(ratio_box)
        self._btn_ratio.setText("比率固定")
        self._btn_ratio.setCheckable(True)
        self._btn_ratio.setToolTip("矩形のリサイズを比率固定にします（四隅ハンドルのみ有効）")
        self._btn_ratio.toggled.connect(self._on_ratio_toggled)

        vb.addWidget(self._btn_ratio, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self._aspect_base = QtWidgets.QLabel("基準: --", ratio_box)
        self._aspect_base.setStyleSheet("QLabel { color:#9aa0a6; }")
        vb.addWidget(self._aspect_base, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        root.addSpacing(4)
        root.addWidget(ratio_box, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        # ---- 起動時の状態を同期（Main -> Panel）----
        on = bool(getattr(self.mainwin.label, "_aspect_lock", False))
        self.set_aspect_button_state(on)
        base = getattr(self.mainwin.label, "_aspect_base_wh", None)
        self.update_aspect_base(base)

    # ====== 角丸描画（はみ出しゼロ）======

    def _rounded_path(self, rect: QtCore.QRectF) -> QtGui.QPainterPath:
        r = max(0.0, float(self._radius))
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, r, r)
        return path

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        # 角外は完全透過にするためにマスクをセット
        rect = QtCore.QRectF(self.rect()).adjusted(0, 0, -0.5, -0.5)
        path = self._rounded_path(rect)
        region = QtGui.QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # 枠線の半分が外に出ないように内側へオフセット
        inset = self._border_w / 2.0
        rect = QtCore.QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        path = self._rounded_path(rect)

        # 背景
        p.fillPath(path, self._bg_color)

        # 枠線
        pen = QtGui.QPen(self._border_color, self._border_w)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawPath(path)

    # ========== パネル外から呼べるAPI ==========
    def reset_counters(self):
        for lbl in self._val_labels.values():
            lbl.setText("0")

    def set_nudge_enabled(self, enabled: bool):
        for btn in self.findChildren(QtWidgets.QToolButton):
            if btn is self._btn_ratio:
                continue
            btn.setEnabled(bool(enabled))

    def set_aspect_button_state(self, on: bool):
        """比率固定ON/OFFに応じてUI反映（トグル表示・◁▷の有効/無効・ラベルの淡色化）"""
        on = bool(on)
        blocker = QtCore.QSignalBlocker(self._btn_ratio)
        self._btn_ratio.setChecked(on)
        del blocker

        try:
            self._btn_ratio.setText("比率固定 🔒" if on else "比率固定")
            self._btn_ratio.setToolTip("比率固定: ON" if on else "比率固定: OFF")
        except Exception:
            pass

        self.set_nudge_enabled(not on)
        for lbl in self._val_labels.values():
            lbl.setEnabled(not on)

    def update_aspect_base(self, wh):
        """基準: W × H 表示の更新。None なら --"""
        if wh and isinstance(wh, (tuple, list)) and len(wh) == 2 and all(isinstance(v, int) and v > 0 for v in wh):
            self._aspect_base.setText(f"基準: {wh[0]} × {wh[1]}")
        else:
            self._aspect_base.setText("基準: --")

    # ========== 内部ハンドラ ==========
    def _on_ratio_toggled(self, on: bool):
        """パネルからのトグル → Main に反映 → UIも反映"""
        try:
            if hasattr(self.mainwin, "set_aspect_lock"):
                self.mainwin.set_aspect_lock(bool(on))
        finally:
            self.set_aspect_button_state(bool(on))
            base = getattr(self.mainwin.label, "_aspect_base_wh", None)
            self.update_aspect_base(base)

class MovableNudgePanel(NudgePanel):
    def __init__(self, mainwin, nudge_cb):
        super().__init__(mainwin, nudge_cb)   # ← parent= は使わない
        # ドラッグできるツールウィンドウ化
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool |
            QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_off = QtCore.QPoint()

        # ▼▼ キーボードフォーカスを一切奪わない設定 ▼▼
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        try:
            # Qt5/Qt6両対応のつもり（未サポート環境は無視）
            self.setWindowFlag(QtCore.Qt.WindowType.WindowDoesNotAcceptFocus, True)
        except Exception:
            pass

        # 内部の子ウィジェット（ボタン等）もフォーカスを受けないようにする
        for w in self.findChildren(QtWidgets.QWidget):
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_off = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging and (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            self.move(ev.globalPosition().toPoint() - self._drag_off)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False

            # ユーザーが微調整パネルをドラッグしたので、
            # 以降は矩形移動に自動追従させない
            try:
                if hasattr(self, "mainwin") and self.mainwin is not None:
                    self.mainwin._nudge_detached = True
            except Exception:
                pass

            # 位置を保存（設定があれば）
            try:
                st = getattr(self.mainwin, "settings", None) or getattr(self.parent(), "settings", None)
                if st:
                    st.setValue("nudge_pos", self.pos())
            except Exception:
                pass

            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    # ------- ここから追加：矢印キーをラベルへ通す -------

    def showEvent(self, e):
        super().showEvent(e)
        # 念のため、表示後にラベルへフォーカスを戻す（キー操作を効かせる）
        try:
            self.mainwin.label.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

    def keyPressEvent(self, ev):
        # 万一このパネルがキーを受けても、ラベルへ転送して矢印移動を有効化
        try:
            QtWidgets.QApplication.sendEvent(self.mainwin.label, ev)
        except Exception:
            pass
        # 自分では処理しない（ベースへ渡さない）

    def keyReleaseEvent(self, ev):
        # 長押し・オートリピートの整合のため、リリースも転送
        try:
            QtWidgets.QApplication.sendEvent(self.mainwin.label, ev)
        except Exception:
            pass

class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, overwrite=False, thumb_scroll_step=24, hq_zoom=False):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setModal(True)

        # 設定ダイアログからは保存方式を操作しないため、元の値だけ保持
        self._orig_overwrite_mode   = bool(overwrite)

        v = QtWidgets.QVBoxLayout(self)

        # --- 保存（チェック一つだけ残す） ---
        grp_save = QtWidgets.QGroupBox("保存")
        gsave = QtWidgets.QVBoxLayout(grp_save)

        self.chk_prompt_save_on_load = QtWidgets.QCheckBox("読み込み時に保存先ダイアログを表示")
        # 初期値は open_options_dialog 側で注入されるのでここでは仮値
        self.chk_prompt_save_on_load.setChecked(False)
        gsave.addWidget(self.chk_prompt_save_on_load)

        # --- ビューア設定 ---
        grp_view = QtWidgets.QGroupBox("ビューア")
        gview = QtWidgets.QFormLayout(grp_view)

        self.spin_thumb_step = QtWidgets.QSpinBox()
        self.spin_thumb_step.setRange(1, 200)
        self.spin_thumb_step.setSingleStep(1)
        self.spin_thumb_step.setValue(int(thumb_scroll_step))
        self.spin_thumb_step.setSuffix(" px / ノッチ")

        self.chk_hq_zoom = QtWidgets.QCheckBox("HQズーム（高品質補間＋軽いシャープ）")
        self.chk_hq_zoom.setChecked(bool(hq_zoom))

        gview.addRow("サムネイルのスクロール量:", self.spin_thumb_step)
        gview.addRow(self.chk_hq_zoom)

        # レイアウト反映
        v.addWidget(grp_save)
        v.addWidget(grp_view)
        v.addStretch(1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def values(self) -> dict:
        # overwrite_* は“元の値のまま”返す（設定ダイアログでは変更しない）
        return {
            "overwrite_mode": self._orig_overwrite_mode,
            "thumb_scroll_step": self.spin_thumb_step.value(),
            "hq_zoom": self.chk_hq_zoom.isChecked(),
            "show_save_dialog_on_load": self.chk_prompt_save_on_load.isChecked(),
        }

class SaveDestinationDialog(QtWidgets.QDialog):
    """
    画像/フォルダ読み込み直後に表示する保存先ダイアログ。
    ・読み込み元と同じ/別フォルダ
    ・連番保存 or 上書き保存
    ・今後このダイアログを表示しない
    """
    def __init__(self, parent=None, *, src_dir="", dest_mode="same",
                 custom_dir="", overwrite=False, show_again=True):
        super().__init__(parent)
        self.setWindowTitle("保存先の選択")
        self.setModal(True)

        v = QtWidgets.QVBoxLayout(self)

        # --- 保存先 ---
        grp_dest = QtWidgets.QGroupBox("保存先")
        gdest = QtWidgets.QGridLayout(grp_dest)

        self.rad_same = QtWidgets.QRadioButton("読み込み元と同じフォルダ")
        self.rad_custom = QtWidgets.QRadioButton("別のフォルダを指定")
        self.rad_same.setChecked(dest_mode != "custom")
        self.rad_custom.setChecked(dest_mode == "custom")

        self.edit_dir = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QToolButton()
        self.btn_browse.setText("参照.")
        self.edit_dir.setText(custom_dir or src_dir)

        gdest.addWidget(self.rad_same,   0, 0, 1, 3)
        gdest.addWidget(self.rad_custom, 1, 0, 1, 1)
        gdest.addWidget(self.edit_dir,   1, 1, 1, 1)
        gdest.addWidget(self.btn_browse, 1, 2, 1, 1)

        # --- 保存方法 ---
        grp_mode = QtWidgets.QGroupBox("保存方法")
        gmode = QtWidgets.QVBoxLayout(grp_mode)
        self.rad_seq  = QtWidgets.QRadioButton("連番保存（既存ファイルは残す）")
        self.rad_ow   = QtWidgets.QRadioButton("上書き保存")
        self.rad_ow.setChecked(bool(overwrite))
        self.rad_seq.setChecked(not bool(overwrite))

        gmode.addWidget(self.rad_seq)
        gmode.addWidget(self.rad_ow)

        # --- 今後表示しない ---
        self.chk_no_again = QtWidgets.QCheckBox("今後はこのダイアログを表示しない")
        self.chk_no_again.setChecked(not bool(show_again))

        v.addWidget(grp_dest)
        v.addWidget(grp_mode)
        v.addWidget(self.chk_no_again)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)

        # enable/disable custom dir widgets
        def _toggle():
            on = self.rad_custom.isChecked()
            self.edit_dir.setEnabled(on)
            self.btn_browse.setEnabled(on)
        _toggle()
        self.rad_same.toggled.connect(_toggle)
        self.rad_custom.toggled.connect(_toggle)

        def _browse():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "保存先フォルダを選択", self.edit_dir.text() or src_dir)
            if d:
                self.edit_dir.setText(d)
        self.btn_browse.clicked.connect(_browse)

        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        self._result = None

    def _accept(self):
        dest_mode = "custom" if self.rad_custom.isChecked() else "same"
        dest_dir = self.edit_dir.text().strip()
        if dest_mode == "custom" and not dest_dir:
            QtWidgets.QMessageBox.warning(self, "保存先の選択", "保存先フォルダを指定してください。")
            return
        self._result = {
            "dest_mode": dest_mode,
            "custom_dir": dest_dir,
            "overwrite": self.rad_ow.isChecked(),
            "show_again": not self.chk_no_again.isChecked(),
        }
        self.accept()

    def values(self):
        return self._result or {}

class CropLabel(QtWidgets.QLabel):
    selectionMade = QtCore.pyqtSignal(QtCore.QRect, QtCore.QPoint)
    fixedSelectionMade = QtCore.pyqtSignal(QtCore.QRect, QtCore.QPoint)
    movedRect = QtCore.pyqtSignal(QtCore.QRect)

    def __init__(self, mainwin):
        super().__init__(mainwin)
        self.mainwin = mainwin

        self.drag_rect_img = None
        self.fixed_crop_rect_img = None    
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.drag_rect = None
        self.drag_origin = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.fixed_crop_mode = False
        self.fixed_crop_rect = None
        self.fixed_crop_size = None
        self.fixed_crop_drag_offset = None
        self.zoom_mode = False
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._gesture_start = None
        self._gesture_in_progress = False
        self._gesture_current = None
        self._gesture_mods = None              # ジェスチャ中に押された修飾キーを蓄積
        self._gesture_ever_shift = False       # 途中で一度でもShiftが押されたらTrueにラッチ
        self.setStyleSheet("""background: transparent; border: 2px solid #f28524; border-radius: 12px;""")
        self._pan_active = False
        self._pan_start_pos = None
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        self.setMouseTracking(True) 
        self._edge_lock_active = False
        self._edge_lock = {"left": False, "right": False, "top": False, "bottom": False}
        self._edge_locked_w = None  # ロック時の幅（px）
        self._edge_locked_h = None  # ロック時の高さ（px）
        # --- 右ドラッグのジェスチャ軌跡（滑らか＋フェード） ---
        self._gesture_path = QtGui.QPainterPath()
        self._gesture_points = []               # QPointF を蓄積
        self._gesture_opacity = 0.0             # 0.0〜1.0（描画時にアルファへ反映）
        # --- ジェスチャの水平ナビ判定しきい値（調整可） ---
        self._gesture_nav_min_dx = 120        # 始点→終点の水平移動の最低px
        self._gesture_nav_ratio  = 1.5        # Σ|dx| / Σ|dy| の最小比
        self._gesture_nav_bbox_ratio = 1.3    # 外接矩形の横長比 (w >= r*h)
        self._gesture_nav_angle_deg = 28      # 始点→終点の水平からの最大角度(°)
        # “まっすぐ度”と方向一貫性の判定用
        self._gesture_nav_curv_max = 1.35     # 経路長 / 直線距離 の上限（曲がり過ぎNG）
        self._gesture_nav_dir_consistency = 0.80  # セグメントの平均cos類似度

        # フェード用アニメーション（ウィジェット全体ではなく“軌跡だけ”）
        self._gesture_anim = QtCore.QPropertyAnimation(self, b"gestureOpacity", self)
        self._gesture_anim.setDuration(600)     # フェード時間（ms）
        self._gesture_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)
        self._gesture_anim.finished.connect(self._clear_gesture)

        #固定枠上にいるかどうか
        self._hovering_fixed_rect = False

        self._moving_fixed_rect = False          # 固定枠を移動中か
        self._move_anchor_img   = None           # つかんだ位置の矩形内オフセット（画像座標）

        self._aspect_lock = False
        self._aspect_ratio = None   # w/h
        self._aspect_base_wh = None  # ← 基準解像度 (W,H)
        self._aspect_anchor = None  # 角を固定するときのアンカー（画像座標の(QPoint)）

        self.adjust_mode = False
        self._resize_handle = None       # 'tl','tm','tr','ml','mr','bl','bm','br' のいずれか
        self._resize_start_rect_img = None
        self._handle_px = 8
        self.setMouseTracking(True)
        self._HANDLE_SIZE = 10   # ■の見た目のサイズ（描画とほぼ合わせる）
        self._EDGE_GRAB   = 8    # 辺の“掴める帯”の厚み(px)
        self._EDGE_PAD    = 6    # 角から少し内側にずらして帯を作る（角と競合しないように）

        self._aspect_primary = None   # 'w'（幅主導） or 'h'（高さ主導） or None

        self._aspect_anchor = None     # 反対側の角（画像座標）: QtCore.QPoint
        self._aspect_ratio = None      # w/h (float)
        self._aspect_handle = None     # 掴んだ角ハンドル名: 'tl','tr','bl','br'

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._recalc_pixmap_offsets()
        self.update()

    def _recalc_pixmap_offsets(self):
        """現在のラベルサイズと貼られているピクスマップから中央寄せオフセットを再計算。"""
        pm = self.pixmap()
        if not pm or pm.isNull():
            self._init_offset_x = 0
            self._init_offset_y = 0
            return

        # ラベル中央寄せオフセット（見た目のための余白）
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        self._init_offset_x = (lw - pw) // 2 if lw > pw else 0
        self._init_offset_y = (lh - ph) // 2 if lh > ph else 0

        # ★重要★ _view_rect_scaled は「ズーム後の全体ピクスマップ座標系」でクランプする
        try:
            zoom = float(getattr(self.mainwin, "zoom_scale", 1.0) or 1.0)
            base_w = getattr(self.mainwin, "base_display_width",  None) or pw
            base_h = getattr(self.mainwin, "base_display_height", None) or ph
            pm_w   = max(1, int(round(base_w * zoom)))
            pm_h   = max(1, int(round(base_h * zoom)))
        except Exception:
            # 何か取れなければ表示ピクスマップサイズで代替（従来動作）
            pm_w, pm_h = pw, ph

        vr = getattr(self, "_view_rect_scaled", QtCore.QRect(0, 0, pm_w, pm_h))
        self._view_rect_scaled = vr.intersected(QtCore.QRect(0, 0, pm_w, pm_h))

    def _sync_fixed_size_from_rect(self) -> None:
        """fixed_crop_rect_img の現在サイズを fixed_crop_size に反映する"""
        try:
            r = getattr(self, "fixed_crop_rect_img", None)
            if r is not None and r.width() > 0 and r.height() > 0:
                self.fixed_crop_size = (int(r.width()), int(r.height()))
        except Exception:
            pass
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._recalc_pixmap_offsets()
        self.update()

    def mousePressEvent(self, event):
        # --- 統一座標取り出し（Qt5/6 両対応） ---
        try:
            pt = event.position().toPoint()
        except AttributeError:
            pt = event.pos()

        # ========== ★ 最優先：右ボタン（ジェスチャ開始） ==========
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            # 修飾キーをラッチ
            try:
                m0 = event.modifiers() | QtWidgets.QApplication.keyboardModifiers()
            except Exception:
                m0 = QtWidgets.QApplication.keyboardModifiers()
            self._gesture_mods = m0
            self._gesture_ever_shift = bool(m0 & QtCore.Qt.KeyboardModifier.ShiftModifier)

            # --- ジェスチャ初期化 ---
            if hasattr(self, "_start_gesture"):
                self._start_gesture(pt)
            else:
                self._gesture_in_progress = True
                self._gesture_path = QtGui.QPainterPath()
                self._gesture_points = []
                self._gesture_opacity = 1.0
                self._gesture_path.moveTo(QtCore.QPointF(pt))
                self._gesture_points.append(QtCore.QPointF(pt))
                try:
                    self._gesture_anim.stop()
                except Exception:
                    pass
                self.update()

            # プレースホルダ表示中でも矢印カーソル固定
            try:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            except Exception:
                pass

            event.accept()
            return
        # ========== 右ボタンはここで終了（以降は左/中ボタンの既存処理） ==========

        # --- 調整モード中：左ボタンはリサイズ/移動の既存処理 ---
        if event.button() == QtCore.Qt.MouseButton.LeftButton and getattr(self, "adjust_mode", False):
            has_target = (self.drag_rect_img is not None) or \
                        (self.fixed_crop_mode and self.fixed_crop_rect_img is not None)
            if has_target:
                h = self._hit_handle(pt)
                if h:
                    self._resize_handle = h
                    self._resize_start_pos = QtCore.QPoint(pt)
                    base = self.fixed_crop_rect_img if self.fixed_crop_mode else self.drag_rect_img
                    if base is not None:
                        self._resize_start_rect_img = QtCore.QRect(base)
                    else:
                        self._resize_handle = None

                    self._aspect_handle  = h
                    self._aspect_primary = None

                    if hasattr(self.mainwin, "_hide_action_panel"):
                        if hasattr(self, "_maybe_suspend_nudge_for_drag"):
                            self._maybe_suspend_nudge_for_drag()
                        self.mainwin._hide_action_panel()

                    if getattr(self, "_aspect_lock", False):
                        h2 = self._resize_handle
                        if h2 and ("m" in h2):
                            self._resize_handle = None
                            event.accept()
                            return
                        base = self.fixed_crop_rect_img if self.fixed_crop_mode else self.drag_rect_img
                        if base is not None and h2 in ("tl","tr","bl","br"):
                            if   h2 == "tl": ax, ay = base.right(),  base.bottom()
                            elif h2 == "tr": ax, ay = base.left(),   base.bottom()
                            elif h2 == "bl": ax, ay = base.right(),  base.top()
                            else:            ax, ay = base.left(),   base.top()   # br
                            self._aspect_anchor = QtCore.QPoint(ax, ay)
                            if not self._aspect_ratio and base.width() > 0 and base.height() > 0:
                                self._aspect_ratio = float(base.width()) / float(base.height())
                    return  # ← 調整モードでハンドルに当たったらここで確定

        # =====★ ここが重要：調整モードの外にも効く“固定枠の移動開始”を共通で置く =====
        if event.button() == QtCore.Qt.MouseButton.LeftButton \
        and self.fixed_crop_mode and (self.fixed_crop_rect_img is not None):
            ix, iy = self.label_to_image_coords(pt.x(), pt.y())
            if self.fixed_crop_rect_img.contains(ix, iy):
                self._moving_fixed_rect = True
                self._move_anchor_img = (
                    ix - self.fixed_crop_rect_img.left(),
                    iy - self.fixed_crop_rect_img.top()
                )
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                if hasattr(self.mainwin, "_hide_action_panel"):
                    self.mainwin._hide_action_panel()
                event.accept()
                return
        # =====★ ここまで =====

        # --- 中ボタン：パン/パネルトグル（既存） ---
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._mid_down_pos = QtCore.QPoint(pt)
            self._mid_timer = QtCore.QElapsedTimer(); self._mid_timer.start()
            if self.mainwin.zoom_scale > 1.0:
                self._pan_active = True
                self._pan_start_pos = QtCore.QPoint(pt)
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            return

        # --- 通常の左ボタンドラッグ開始（既存） ---
        if not self.fixed_crop_mode:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                img_x, img_y = self.label_to_image_coords(pt.x(), pt.y())
                self._drag_start_img = (img_x, img_y)
                self.drag_rect_img = None
                if hasattr(self.mainwin, "_hide_action_panel"):
                    self.mainwin._hide_action_panel()

            dlg = getattr(self.mainwin, "_nudge_overlay", None)
            if dlg is not None and hasattr(dlg, "reset_counters"):
                dlg.reset_counters()

            self._edge_lock_active = False
            self._edge_lock = {"left": False, "right": False, "top": False, "bottom": False}
            self._edge_locked_w = None
            self._edge_locked_h = None
            self.update()

        super().mousePressEvent(event)

    def set_adjust_mode(self, on: bool):
        # 自分自身（CropLabel）のフラグを切り替えるだけ。self.label は使わない！
        self.adjust_mode = bool(on)

        # リサイズ・移動の進行中状態をクリア
        self._resize_handle = None
        self._resize_start_rect_img = None
        self._moving_fixed_rect = False
        self._aspect_anchor = None

        # カーソル形状・ハンドル表示などを即時反映
        if hasattr(self, "refresh_edit_ui"):
            self.refresh_edit_ui()
        else:
            self.update()

        # 操作対象にフォーカス
        self.setFocus()

    # 追加：現在編集対象の矩形（画像座標）を取得
    def _edit_rect_img(self) -> QtCore.QRect | None:
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            return QtCore.QRect(self.fixed_crop_rect_img)
        if self.drag_rect_img is not None:
            return QtCore.QRect(self.drag_rect_img)
        return None

    def _edit_rect_label(self) -> QtCore.QRect | None:
        # “いま編集対象の矩形（画像座標）”を取得
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            r_img = self.fixed_crop_rect_img
        else:
            r_img = getattr(self, "drag_rect_img", None)

        return self._imgrect_to_labelrect(r_img)

    def _draw_resize_handles(self, painter: QtGui.QPainter) -> None:
        # 調整モードでなければ出さない
        if not getattr(self, "adjust_mode", False):
            return

        rl = self._edit_rect_label()
        if not rl or rl.isNull():
            return

        s = 8  # ハンドルの一辺(px)
        painter.save()
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 180)))

        cx = rl.center().x()
        cy = rl.center().y()

        if getattr(self, "_aspect_lock", False):
            # 比率固定ON：四隅のみ
            pts = [
                QtCore.QPoint(rl.left(),  rl.top()),      # NW (tl)
                QtCore.QPoint(rl.right(), rl.top()),      # NE (tr)
                QtCore.QPoint(rl.right(), rl.bottom()),   # SE (br)
                QtCore.QPoint(rl.left(),  rl.bottom()),   # SW (bl)
            ]
        else:
            # 通常：角4 + 辺4
            pts = [
                QtCore.QPoint(rl.left(),  rl.top()),      # NW (tl)
                QtCore.QPoint(cx,         rl.top()),      # N  (tm)
                QtCore.QPoint(rl.right(), rl.top()),      # NE (tr)
                QtCore.QPoint(rl.right(), cy),            # E  (mr)
                QtCore.QPoint(rl.right(), rl.bottom()),   # SE (br)
                QtCore.QPoint(cx,         rl.bottom()),   # S  (bm)
                QtCore.QPoint(rl.left(),  rl.bottom()),   # SW (bl)
                QtCore.QPoint(rl.left(),  cy),            # W  (ml)
            ]

        for p in pts:
            painter.drawRect(p.x() - s // 2, p.y() - s // 2, s, s)

        painter.restore()

    # 追加：どのハンドルにヒットしているか
    def _hit_handle(self, pos: QtCore.QPoint) -> str | None:
        """
        pos: ラベル座標。まずハンドルを優先判定、外したら（比率固定OFFの場合のみ）辺の帯で判定。
        戻り値: "tl","tr","bl","br","ml","mr","tm","bm" もしくは None
        """
        r = self._edit_rect_label()  # いま編集対象の矩形（ラベル座標）
        if not r or r.isEmpty():
            return None

        H   = getattr(self, "_HANDLE_SIZE", 10)
        PAD = getattr(self, "_EDGE_PAD",    6)
        EDGE= getattr(self, "_EDGE_GRAB",   8)

        # 角の中心座標
        tlc = QtCore.QPoint(r.left(),  r.top())
        trc = QtCore.QPoint(r.right(), r.top())
        blc = QtCore.QPoint(r.left(),  r.bottom())
        brc = QtCore.QPoint(r.right(), r.bottom())

        # 中点の中心座標
        cx = r.center().x()
        cy = r.center().y()
        tmc = QtCore.QPoint(cx, r.top())
        bmc = QtCore.QPoint(cx, r.bottom())
        mlc = QtCore.QPoint(r.left(),  cy)
        mrc = QtCore.QPoint(r.right(), cy)

        def sq(cx, cy):
            return QtCore.QRect(int(cx - H/2), int(cy - H/2), H, H)

        # ---- 1) ハンドル（優先判定） ----
        # 比率固定ON：四隅のみ
        if getattr(self, "_aspect_lock", False):
            if   sq(tlc.x(), tlc.y()).contains(pos): return "tl"
            elif sq(trc.x(), trc.y()).contains(pos): return "tr"
            elif sq(blc.x(), blc.y()).contains(pos): return "bl"
            elif sq(brc.x(), brc.y()).contains(pos): return "br"
            # 辺の中点ハンドル＆辺帯は無効（以降も判定しない）
            return None
        else:
            # 通常：角 + 中点
            if   sq(tlc.x(), tlc.y()).contains(pos): return "tl"
            elif sq(trc.x(), trc.y()).contains(pos): return "tr"
            elif sq(blc.x(), blc.y()).contains(pos): return "bl"
            elif sq(brc.x(), brc.y()).contains(pos): return "br"
            elif sq(mlc.x(), mlc.y()).contains(pos): return "ml"
            elif sq(mrc.x(), mrc.y()).contains(pos): return "mr"
            elif sq(tmc.x(), tmc.y()).contains(pos): return "tm"
            elif sq(bmc.x(), bmc.y()).contains(pos): return "bm"

        # ---- 2) 辺の帯（比率固定OFFのときだけ）----
        w_inner = max(0, r.width()  - 2 * PAD)
        h_inner = max(0, r.height() - 2 * PAD)

        if w_inner > 0:
            top_strip    = QtCore.QRect(r.left() + PAD,  r.top()    - EDGE // 2, w_inner, EDGE)
            bottom_strip = QtCore.QRect(r.left() + PAD,  r.bottom() - EDGE // 2, w_inner, EDGE)
            if   top_strip.contains(pos):    return "tm"
            elif bottom_strip.contains(pos): return "bm"

        if h_inner > 0:
            left_strip   = QtCore.QRect(r.left()  - EDGE // 2, r.top() + PAD, EDGE, h_inner)
            right_strip  = QtCore.QRect(r.right() - EDGE // 2, r.top() + PAD, EDGE, h_inner)
            if   left_strip.contains(pos):  return "ml"
            elif right_strip.contains(pos): return "mr"

        return None

    def _update_resize_cursor(self, pos: QtCore.QPoint):
        if not getattr(self, "adjust_mode", False):
            return

        h = self._hit_handle(pos)
        if h in ("tl", "br"):
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        elif h in ("tr", "bl"):
            self.setCursor(QtCore.Qt.CursorShape.SizeBDiagCursor)
        elif h in ("ml", "mr"):
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        elif h in ("tm", "bm"):
            self.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        else:
            # 固定矩形モードのときは、中にいるなら掴める予告として OpenHand
            if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                r = self._fixed_rect_labelcoords()
                if r and r.contains(pos):
                    self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
                    return
            # それ以外（通常矩形 / 矩形外）はベースカーソル維持（何もしない）
    
    def refresh_edit_ui(self) -> None:
        """編集UIの状態をクリアして即時再描画（ハンドル表示/カーソル形状を更新）"""
        # 念のため、グローバルの override cursor が残っていたら解除
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

        # 内部状態クリア
        try:
            self._resize_handle = None
        except Exception:
            pass
        try:
            self._hovered_handle = None
        except Exception:
            pass

        # 現在のマウス位置（取れなければ中央を仮定）
        try:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
        except Exception:
            pos = QtCore.QPoint(self.width() // 2, self.height() // 2)

        if getattr(self, "adjust_mode", False):
            # ★ 調整モード中：
            #   - 非固定矩形      → 基本は ✚
            #   - 固定矩形モード → 基本は ↖ (Arrow)
            try:
                if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                    base = QtCore.Qt.CursorShape.ArrowCursor
                else:
                    base = QtCore.Qt.CursorShape.CrossCursor
                self.setCursor(base)
            except Exception:
                pass

            # ハンドル上／固定矩形内ではリサイズカーソルや OpenHand で上書き
            try:
                if hasattr(self, "_update_resize_cursor"):
                    self._update_resize_cursor(pos)
            except Exception:
                pass
        else:
            # 調整モードOFF：基本は ✚、固定枠の上にいるときだけ手のひら
            try:
                r = self._fixed_rect_labelcoords()
                if r and r.contains(pos):
                    self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            except Exception:
                pass

        self.update()

    def mouseMoveEvent(self, event):
        # Qt5/Qt6 両対応で QPoint を取得
        try:
            pt = event.position().toPoint()
        except AttributeError:
            pt = event.pos()

        # ========= 1) 右ドラッグ中のジェスチャ軌跡を“最優先”で更新 =========
        btns = event.buttons() if hasattr(event, "buttons") else QtCore.Qt.MouseButton.NoButton
        try:
            # PyQt6: enum フラグで判定
            right_drag = (btns & QtCore.Qt.MouseButton.RightButton) == QtCore.Qt.MouseButton.RightButton
        except TypeError:
            # PyQt5: int に落として判定
            right_drag = (int(btns) & int(QtCore.Qt.MouseButton.RightButton)) != 0

        if right_drag and getattr(self, "_gesture_in_progress", False):
            # 途中で押/離された修飾キーを取りこぼさないように累積
            try:
                curmods = event.modifiers() | QtWidgets.QApplication.keyboardModifiers()
            except Exception:
                curmods = QtWidgets.QApplication.keyboardModifiers()

            prevmods = getattr(self, "_gesture_mods", QtCore.Qt.KeyboardModifier.NoModifier)
            self._gesture_mods = (prevmods | curmods)
            if curmods & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self._gesture_ever_shift = True

            self._gesture_current = QtCore.QPoint(pt)
            # 既存のジェスチャ更新ヘルパ（軌跡更新→update() まで行う実装を想定）
            if hasattr(self, "_update_gesture"):
                self._update_gesture(pt)
            else:
                # 保険：自前で軽く更新
                self._gesture_path.lineTo(QtCore.QPointF(pt))
                self.update()
            return
        # ========= 右ドラッグ以外は従来処理へ =========

        # --- 固定枠の移動（従来どおおり） ---
        if (getattr(self, "_moving_fixed_rect", False)
            and getattr(self, "fixed_crop_mode", False)
            and getattr(self, "fixed_crop_rect_img", None) is not None):

            # ★ 固定枠をドラッグしたら、ActionPanel/Nudge の手動配置(=detached)を解除して
            #    以降の矢印キー移動で再び追従させる
            try:
                self.mainwin._action_panel_detached = False
                self.mainwin._nudge_detached = False
            except Exception:
                pass

            ix, iy = self.label_to_image_coords(pt.x(), pt.y())
            dx, dy = self._move_anchor_img or (0, 0)

            img = getattr(self.mainwin, "image", None)
            iw = getattr(img, "width",  0) or 0
            ih = getattr(img, "height", 0) or 0

            r = QtCore.QRect(self.fixed_crop_rect_img)
            w, h = r.width(), r.height()
            nx, ny = ix - dx, iy - dy

            new_rect = QtCore.QRect(nx, ny, w, h)  # クランプなし

            self.fixed_crop_rect_img = QtCore.QRect(new_rect)
            try:
                self.mainwin._crop_rect_img = QtCore.QRect(new_rect)
            except Exception:
                pass

            x1, y1 = self.image_to_label_coords(new_rect.left(),  new_rect.top())
            x2, y2 = self.image_to_label_coords(new_rect.right(), new_rect.bottom())
            r_lbl = QtCore.QRect(
                min(x1, x2), min(y1, y2),
                abs(x2 - x1) + 1, abs(y2 - y1) + 1
            )

            # ラベル矩形も同期（描画・他UIの参照用）
            try:
                self.mainwin._crop_rect = QtCore.QRect(r_lbl)
            except Exception:
                pass

            # ActionPanel は再生成せず位置だけ
            panel = getattr(self.mainwin, "_action_panel", None)
            if panel and panel.isVisible():
                try:
                    panel.move(self.mainwin._compute_action_panel_pos(r_lbl, True))
                except Exception:
                    pass

            # Nudge は新旧ヘルパ互換で位置合わせ
            ov = getattr(self.mainwin, "_nudge_overlay", None)
            if ov and ov.isVisible():
                gap = getattr(self.mainwin, "nudge_gap_px", 8)
                try:
                    if hasattr(self.mainwin, "_position_nudge_overlay_above_action"):
                        self.mainwin._position_nudge_overlay_above_action(ov, gap=gap)
                    else:
                        self.mainwin._position_nudge_overlay(ov, gap=gap)
                except Exception:
                    pass

            if hasattr(self.mainwin, "update_crop_size_label"):
                self.mainwin.update_crop_size_label(self.mainwin._crop_rect_img, img_space=True)
            if hasattr(self.mainwin, "safe_update_preview"):
                QtCore.QTimer.singleShot(
                    0,
                    lambda: self.mainwin.safe_update_preview(self.mainwin._crop_rect_img)
                )
            elif hasattr(self.mainwin, "_schedule_preview"):
                self.mainwin._schedule_preview(self.mainwin._crop_rect_img)

            self.update()
            return

        # --- 比率固定リサイズ（従来どおり） ---
        if (getattr(self, "adjust_mode", False)
            and getattr(self, "_resize_handle", None)
            and getattr(self, "_aspect_lock", False)
            and getattr(self, "_aspect_anchor", None) is not None
            and getattr(self, "_aspect_ratio", None)):

            r0 = self.fixed_crop_rect_img if self.fixed_crop_mode else self.drag_rect_img
            if r0 is not None:
                ax, ay = self._aspect_anchor.x(), self._aspect_anchor.y()
                gx, gy = self.label_to_image_coords(pt.x(), pt.y())
                ratio = float(self._aspect_ratio)

                dx, dy = gx - ax, gy - ay
                if dx == 0 and dy == 0:
                    return

                cand1_dx, cand1_dy = dx, (1 if dy >= 0 else -1) * abs(dx) / ratio
                cand2_dx, cand2_dy = (1 if dx >= 0 else -1) * abs(dy) * ratio, dy
                d1 = (cand1_dx - dx)**2 + (cand1_dy - dy)**2
                d2 = (cand2_dx - dx)**2 + (cand2_dy - dy)**2
                ndx, ndy = (cand1_dx, cand1_dy) if d1 <= d2 else (cand2_dx, cand2_dy)

                x2, y2 = int(round(ax + ndx)), int(round(ay + ndy))
                new_rect = QtCore.QRect(QtCore.QPoint(min(ax, x2), min(ay, y2)),
                                        QtCore.QPoint(max(ax, x2), max(ay, y2)))
                if new_rect.width()  < 1: new_rect.setWidth(1)
                if new_rect.height() < 1: new_rect.setHeight(1)

                if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                    self.fixed_crop_rect_img = new_rect
                    x1, y1 = self.image_to_label_coords(new_rect.left(),  new_rect.top())
                    x2, y2 = self.image_to_label_coords(new_rect.right(), new_rect.bottom())
                    rect_label = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    self.fixedSelectionMade.emit(rect_label, pt)
                elif self.drag_rect_img is not None:
                    self.drag_rect_img = new_rect
                    x1, y1 = self.image_to_label_coords(new_rect.left(),  new_rect.top())
                    x2, y2 = self.image_to_label_coords(new_rect.right(), new_rect.bottom())
                    rect_label = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    self.movedRect.emit(rect_label)

                self.update()
                return

        # --- 通常のリサイズ（従来どおり） ---
        if getattr(self, "adjust_mode", False) and getattr(self, "_resize_handle", None):
            gx, gy = self.label_to_image_coords(pt.x(), pt.y())
            r0 = QtCore.QRect(self._resize_start_rect_img)
            left, right = r0.left(), r0.right()
            top, bottom = r0.top(), r0.bottom()
            h = self._resize_handle
            if "l" in h: left   = gx
            if "r" in h: right  = gx
            if "t" in h: top    = gy
            if "b" in h: bottom = gy
            new_rect = QtCore.QRect(QtCore.QPoint(min(left, right),  min(top, bottom)),
                                    QtCore.QPoint(max(left, right),  max(top, bottom)))
            if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                self.fixed_crop_rect_img = new_rect
                x1, y1 = self.image_to_label_coords(new_rect.left(),  new_rect.top())
                x2, y2 = self.image_to_label_coords(new_rect.right(), new_rect.bottom())
                rect_label = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.fixedSelectionMade.emit(rect_label, pt)
            elif self.drag_rect_img is not None:
                self.drag_rect_img = new_rect
                x1, y1 = self.image_to_label_coords(new_rect.left(),  new_rect.top())
                x2, y2 = self.image_to_label_coords(new_rect.right(), new_rect.bottom())
                rect_label = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.movedRect.emit(rect_label)
            self.update()
            return

        # --- 調整モードのカーソル更新（ベース＋上書き） ---
        if getattr(self, "adjust_mode", False):
            # 1) まずベースカーソルを毎フレームリセット
            #    ・通常矩形      → ✚
            #    ・固定矩形モード → ↖
            try:
                if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                    base = QtCore.Qt.CursorShape.ArrowCursor
                else:
                    base = QtCore.Qt.CursorShape.CrossCursor
                self.setCursor(base)
            except Exception:
                pass

            # 2) その上で、ハンドル／辺の帯／固定矩形内なら
            #    _update_resize_cursor() 側でリサイズカーソルや OpenHand に“上書き”
            try:
                if hasattr(self, "_update_resize_cursor"):
                    self._update_resize_cursor(pt)
            except Exception:
                pass

        # --- パン（従来どおり） ---
        if getattr(self, '_pan_active', False):
            dx = pt.x() - self._pan_start_pos.x()
            dy = pt.y() - self._pan_start_pos.y()
            self.mainwin.pan_image(-dx, -dy)
            self._pan_start_pos = QtCore.QPoint(pt)
            return

        # --- 通常ドラッグの矩形生成（従来どおり） ---
        if self.fixed_crop_mode:
            if self.fixed_crop_drag_offset is not None and self.fixed_crop_rect_img is not None:
                new_topleft = pt - self.fixed_crop_drag_offset
                img_left, img_top = self.label_to_image_coords(new_topleft.x(), new_topleft.y())
                crop_w, crop_h = self.fixed_crop_size
                self.fixed_crop_rect_img = QtCore.QRect(img_left, img_top, crop_w, crop_h)
                self.update()
                x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(),  self.fixed_crop_rect_img.top())
                x2, y2 = self.image_to_label_coords(self.fixed_crop_rect_img.right(), self.fixed_crop_rect_img.bottom())
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.fixedSelectionMade.emit(rect, pt)
        else:
            if btns & QtCore.Qt.MouseButton.LeftButton:
                if not hasattr(self, '_drag_start_img') or self._drag_start_img is None:
                    x0, y0 = self.label_to_image_coords(pt.x(), pt.y())
                    self._drag_start_img = (x0, y0)
                    self.drag_rect_img = None
                    if hasattr(self.mainwin, "_hide_action_panel"):
                        self.mainwin._hide_action_panel()
                    dlg = getattr(self.mainwin, "_nudge_overlay", None)
                    if dlg is not None and hasattr(dlg, "reset_counters"):
                        dlg.reset_counters()

                img_x, img_y = self.label_to_image_coords(pt.x(), pt.y())
                x1, y1 = self._drag_start_img
                if getattr(self.mainwin, "multiple_lock_enabled", False):
                    mw = getattr(self.mainwin, "multiple_w", 64)
                    mh = getattr(self.mainwin, "multiple_h", 64)
                    self.drag_rect_img = self._apply_multiple_and_keep_inside(x1, y1, img_x, img_y, mw, mh)
                else:
                    left   = min(x1, img_x)
                    top    = min(y1, img_y)
                    right  = max(x1, img_x)
                    bottom = max(y1, img_y)
                    self.drag_rect_img = QtCore.QRect(left, top, right - left, bottom - top)

                self.update()
                x1, y1 = self.image_to_label_coords(self.drag_rect_img.left(),  self.drag_rect_img.top())
                x2, y2 = self.image_to_label_coords(self.drag_rect_img.right(), self.drag_rect_img.bottom())
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.movedRect.emit(rect)

        #  固定モードのホバーでカーソル更新 ===
        try:
            no_left = (btns & QtCore.Qt.MouseButton.LeftButton) == QtCore.Qt.MouseButton.NoButton
        except TypeError:
            no_left = (int(btns) & int(QtCore.Qt.MouseButton.LeftButton)) == 0

        if getattr(self, "fixed_crop_mode", False) \
        and not getattr(self, "adjust_mode", False) \
        and not getattr(self, "_moving_fixed_rect", False) \
        and not getattr(self, "_pan_active", False) \
        and no_left:
            # 矩形上なら OpenHand、外なら Cross を内部で切替える想定
            if hasattr(self, "refresh_edit_ui"):
                self.refresh_edit_ui()

    def _finalize_adjust_interaction(self, release_pos: QtCore.QPoint | None = None):
        # 1) フラグ類リセット
        self._resize_handle = None
        self._moving_fixed_rect = False
        self._aspect_anchor = None

        # 2) 現在の矩形（画像座標）を取得
        rect_img = None
        is_fixed = False
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            rect_img = QtCore.QRect(self.fixed_crop_rect_img)
            is_fixed = True
        elif self.drag_rect_img is not None:
            rect_img = QtCore.QRect(self.drag_rect_img)

        if rect_img is None:
            self.update()
            return

        # 3) _crop_rect_img を更新（プレビューが古くならないように先に反映）
        try:
            self.mainwin._crop_rect_img = QtCore.QRect(rect_img)
        except Exception:
            pass

        if hasattr(self.mainwin, "update_crop_size_label"):
            self.mainwin.update_crop_size_label(self.mainwin._crop_rect_img, img_space=True)
        if hasattr(self.mainwin, "safe_update_preview"):
            QtCore.QTimer.singleShot(
                0,
                lambda: self.mainwin.safe_update_preview(self.mainwin._crop_rect_img)
            )
        elif hasattr(self.mainwin, "_schedule_preview"):
            self.mainwin._schedule_preview(self.mainwin._crop_rect_img)

        # 4) ラベル座標に変換（信号はラベル矩形で投げる実装に合わせる）
        x1, y1 = self.image_to_label_coords(rect_img.left(),  rect_img.top())
        x2, y2 = self.image_to_label_coords(rect_img.right(), rect_img.bottom())
        r_lbl = QtCore.QRect(
            min(x1, x2), min(y1, y2),
            abs(x2 - x1) + 1,
            abs(y2 - y1) + 1
        )

        # --- ここから下を修正ポイントとして差し替え ---

        # MainWindow と「ユーザーが中ボタンでパネルを隠したかどうか」のフラグ
        mw = getattr(self, "mainwin", None) or self.window()
        panel_hidden = bool(getattr(mw, "_panel_hidden_by_user", False)) if mw is not None else False

        # 5) パネル再表示（ユーザーが「今はいらない」と言っているときは出さない）
        if mw is not None and hasattr(mw, "show_action_panel") and not panel_hidden:
            mw.show_action_panel(r_lbl, is_fixed)

        # 5.5) Nudge オーバーレイがあれば、ActionPanel の少し上に寄せ直して前面へ
        if mw is not None:
            ov = getattr(mw, "_nudge_overlay", None)
            # ★調整モードONかつ可視のとき、かつ「ユーザーが隠していない」場合だけ触る
            if ov and ov.isVisible() and bool(getattr(mw, "_adjust_mode", False)) and not panel_hidden:
                gap = getattr(mw, "nudge_gap_px", 4)
                try:
                    QtCore.QTimer.singleShot(
                        0,
                        lambda o=ov, g=gap: mw._position_nudge_overlay_above_action(o, g)
                    )
                    ov.raise_()
                except Exception:
                    pass

        # ★ 5.9) （新）固定枠サイズを確定値に同期して保存
        if is_fixed:
            try:
                # ヘルパーを作っている場合はそちらを呼んでもOK: self._sync_fixed_size_from_rect()
                if rect_img.width() > 0 and rect_img.height() > 0:
                    self.fixed_crop_size = (int(rect_img.width()), int(rect_img.height()))
            except Exception:
                pass

        # 6) クリック位置（ラベル座標）。無ければ現在カーソル位置を使う
        pt = release_pos or self.mapFromGlobal(QtGui.QCursor.pos())

        # 7) 適切なシグナルを emit（固定モードなら fixedSelectionMade）
        if is_fixed:
            self.fixedSelectionMade.emit(r_lbl, pt)
        else:
            self.selectionMade.emit(r_lbl, pt)

        self.update()

    def mouseReleaseEvent(self, event):
        # Qt5/Qt6 両対応で QPoint を取得
        try:
            pt = event.position().toPoint()
        except AttributeError:
            pt = event.pos()

        def _resume_nudge():
            mw = getattr(self, "mainwin", None) or self.window()
            if not mw or not hasattr(mw, "ensure_nudge_visibility"):
                return

            # ★ 中ボタントグルで「パネル類を隠した」状態では Nudge を勝手に復帰させない
            if getattr(mw, "_panel_hidden_by_user", False):
                return

            try:
                mw.ensure_nudge_visibility()   # ← _suspend_nudge_overlay(False) は使わない
            except Exception:
                pass

        # === 右ボタン解放：ジェスチャ終了（フォルダ表示中でも有効） ===
        if (event.button() == QtCore.Qt.MouseButton.RightButton
            and getattr(self, "_gesture_in_progress", False)):

            # 最終点を更新してから判定（軌跡に最後の点を反映）
            self._gesture_current = QtCore.QPoint(pt)

            # あなたの既存ロジックに合わせて分類関数を使用
            dir_txt = None
            try:
                dir_txt = self._classify_horizontal_gesture()
            except Exception:
                dir_txt = None

            if dir_txt not in ("left", "right"):
                # 方向が出なかったら終了掃除だけ
                try:
                    if hasattr(self, "_end_gesture_and_fade"):
                        self._end_gesture_and_fade()
                    else:
                        # フェード無しで即消去
                        self._gesture_in_progress = False
                        self._gesture_points = []
                        self._gesture_start = None
                        self._gesture_current = None
                        self._gesture_mods = None
                        self._gesture_ever_shift = False
                        self.update()
                except Exception:
                    pass
                event.accept()
                return

            direction = +1 if dir_txt == "right" else -1

            # 修飾キーは「期間中に累積したもの」を使う（Shift はラッチ）
            try:
                mods = getattr(self, "_gesture_mods", None)
                if mods is None:
                    try:
                        mods = event.modifiers() | QtWidgets.QApplication.keyboardModifiers()
                    except Exception:
                        mods = QtWidgets.QApplication.keyboardModifiers()
                if getattr(self, "_gesture_ever_shift", False):
                    mods = mods | QtCore.Qt.KeyboardModifier.ShiftModifier
            except Exception:
                mods = QtCore.Qt.KeyboardModifier.NoModifier

            # ナビ実行（フォルダ/画像どちらでもサムネ行を移動 → メインに同期）
            mw = getattr(self, "mainwin", None) or self.window()
            try:
                if mw and hasattr(mw, "navigate_from_gesture"):
                    mw.navigate_from_gesture(direction=direction, modifiers=mods)
                elif mw and hasattr(mw, "_move_thumb_focus"):
                    mw._move_thumb_focus(+1 if direction > 0 else -1)
                elif mw:
                    (mw.show_next_image if direction > 0 else mw.show_prev_image)()
            except Exception:
                pass

            # 終了掃除（軌跡フェード or 即消去）
            try:
                if hasattr(self, "_end_gesture_and_fade"):
                    self._end_gesture_and_fade()
                else:
                    self._gesture_in_progress = False
                    self._gesture_points = []
                    self._gesture_start = None
                    self._gesture_current = None
                    self._gesture_mods = None
                    self._gesture_ever_shift = False
                    if hasattr(self, "_gesture_anim"):
                        self._gesture_anim.start()
                    else:
                        self.update()
            except Exception:
                pass

            # プレースホルダ中は矢印、それ以外はクロスに戻す
            try:
                if getattr(self.mainwin, "_placeholder_active", False):
                    self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                else:
                    self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            except Exception:
                pass

            event.accept()
            return  # ★ 右ボタン処理はここで完了

        # === 左ボタン：調整モードの確定など（従来処理） ===
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # --- 調整モード中の確定処理（リサイズ/移動） ---
            if getattr(self, "adjust_mode", False) and (
                getattr(self, "_resize_handle", None) or getattr(self, "_moving_fixed_rect", False)
            ):
                was_resizing = bool(getattr(self, "_resize_handle", None))
                size_changed = False

                # ハンドルでのリサイズだった場合のみ、サイズが変わったかを判定
                if was_resizing:
                    before = getattr(self, "_resize_start_rect_img", None)
                    after = None
                    if getattr(self, "fixed_crop_mode", False) and getattr(self, "fixed_crop_rect_img", None):
                        after = self.fixed_crop_rect_img
                    elif getattr(self, "drag_rect_img", None):
                        after = self.drag_rect_img
                    if before is not None and after is not None:
                        try:
                            size_changed = (before.width() != after.width()) or (before.height() != after.height())
                        except Exception:
                            size_changed = False

                # 状態クリア
                self._resize_handle = None
                self._moving_fixed_rect = False
                self._resize_start_rect_img = None
                self._aspect_anchor = None
                try:
                    self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                except Exception:
                    pass

                # サイズが変わっていれば Nudge のカウンタをリセット
                if size_changed:
                    dlg = getattr(self.mainwin, "_nudge_overlay", None)
                    if dlg is not None and hasattr(dlg, "reset_counters"):
                        try:
                            dlg.reset_counters()
                        except Exception:
                            pass

                # パネル＆Nudge再表示などの確定処理
                self._finalize_adjust_interaction(pt)

                # Nudge を復帰（寄せ直し＋再表示）
                _resume_nudge()

                event.accept()
                return

            # --- 非調整モードでの固定枠移動終了のフォールバック ---
            if getattr(self, "_moving_fixed_rect", False):
                self._moving_fixed_rect = False
                self._move_anchor_img = None
                try:
                    self.unsetCursor()
                except Exception:
                    try:
                        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    except Exception:
                        pass

                # Nudge を復帰
                _resume_nudge()

                event.accept()
                return

            # 比率固定アンカーの掃除（保険）
            if getattr(self, "_aspect_anchor", None) is not None:
                self._aspect_anchor = None
                self._aspect_primary = None 
                self._aspect_handle = None

            # 調整モード外で万一ハンドルフラグが残っていた場合の掃除
            if not getattr(self, "adjust_mode", False) and getattr(self, "_resize_handle", None):
                self._resize_handle = None
                self._resize_start_rect_img = None
                event.accept()
                return

        # 既定処理へ
        super().mouseReleaseEvent(event)

        # --- 中ボタン: クリックならパネルの表示/非表示をトグル。ドラッグならパン終了のみ ---
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            # パンしていたら終わらせる
            if getattr(self, '_pan_active', False):
                self._pan_active = False
                self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

            # クリック判定
            dist = 9999
            if getattr(self, '_mid_down_pos', None) is not None:
                d = pt - self._mid_down_pos
                dist = abs(d.x()) + abs(d.y())
            elapsed = self._mid_timer.elapsed() if getattr(self, '_mid_timer', None) else 9999
            is_click = (dist <= 5 and elapsed <= 300)

            if is_click and hasattr(self.mainwin, "toggle_action_panel"):
                self.mainwin.toggle_action_panel()

            # パン終了後のカーソル復帰は“中ボタンのときだけ”
            r = self._fixed_rect_labelcoords() if (self.fixed_crop_mode and self.fixed_crop_rect_img is not None) else None
            if r and r.contains(pt):
                self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

            # 後片付け
            self._mid_down_pos = None
            self._mid_timer = None
            return  # ← 中ボタンのときだけここで抜ける

        # --- 固定枠モード ---
        if self.fixed_crop_mode:
            if self.fixed_crop_rect_img is None:
                return
            self.fixed_crop_drag_offset = None
            x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(), self.fixed_crop_rect_img.top())
            x2, y2 = self.image_to_label_coords(
                self.fixed_crop_rect_img.left() + self.fixed_crop_rect_img.width(),
                self.fixed_crop_rect_img.top()  + self.fixed_crop_rect_img.height()
            )
            rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.fixedSelectionMade.emit(rect, pt)
            r = self._fixed_rect_labelcoords()
            self.setCursor(
                QtCore.Qt.CursorShape.OpenHandCursor if (r and r.contains(pt))
                else QtCore.Qt.CursorShape.CrossCursor
            )
            self._hovering_fixed_rect = bool(r and r.contains(pt))

            # Nudge を復帰
            _resume_nudge()
            return

        # --- 通常ドラッグの確定 ---
        if hasattr(self, "_drag_start_img") and self._drag_start_img is not None and self.drag_rect_img is not None:
            gx1, gy1 = self.drag_rect_img.left(), self.drag_rect_img.top()
            gx2 = self.drag_rect_img.left() + self.drag_rect_img.width()
            gy2 = self.drag_rect_img.top()  + self.drag_rect_img.height()
            if abs(gx2-gx1) > 5 and abs(gy2-gy1) > 5:
                x1, y1 = self.image_to_label_coords(gx1, gy1)
                x2, y2 = self.image_to_label_coords(gx2, gy2)
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                self.selectionMade.emit(rect, pt)
            self._drag_start_img = None
            self.update()

            # Nudge を復帰
            _resume_nudge()
            return

    def wheelEvent(self, event):
        # 画像未ロード時やピクスマップ未生成時は何もしない
        if getattr(self.mainwin, "image", None) is None or self.mainwin.img_pixmap is None:
            event.ignore()
            return

        if event.angleDelta().y() > 0:
            self.mainwin.zoom_in()
        else:
            self.mainwin.zoom_out()
        event.accept()
        
    def enterEvent(self, event):
        # 左ボタン押下中にラベルへ入ってきたときの処理だが、
        # いまカーソルが微調整ダイアログ上なら何もしない（= ダイアログのドラッグを妨げない）
        if QtWidgets.QApplication.mouseButtons() & QtCore.Qt.MouseButton.LeftButton:
            dlg = getattr(self.mainwin, "_nudge_overlay", None)
            if dlg and dlg.isVisible():
                gp = QtGui.QCursor.pos()  # グローバル座標
                if dlg.geometry().contains(gp):
                    super().enterEvent(event)
                    return

            # ここから従来の処理（開始点記録＋パネル隠す 等）
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            img_x0, img_y0 = self.label_to_image_coords(pos.x(), pos.y())
            self._drag_start_img = (img_x0, img_y0)
            self.drag_rect_img = None

            try:
                self.mainwin._hide_action_panel()
            except Exception:
                pass

            dlg = getattr(self.mainwin, "_nudge_overlay", None) or getattr(self.mainwin, "_nudge_overlay", None)
            if dlg is not None and hasattr(dlg, "reset_counters"):
                dlg.reset_counters()

            self.update()

        super().enterEvent(event)

    def clear_rubberBand(self):
        self.drag_rect = None
        self.drag_origin = None
        self.drag_rect_img = None 
        self.update()
        # ★ボタンも消す（固定枠モードなら _hide_action_panel 内で無視される）
        if hasattr(self.mainwin, "_hide_action_panel"):
            self.mainwin._hide_action_panel()

    def clear_fixed_crop(self):
        self.fixed_crop_mode = False
        self.fixed_crop_rect = None
        self.fixed_crop_size = None
        self.fixed_crop_drag_offset = None
        self.fixed_crop_rect_img = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self._hovering_fixed_rect = False
        self.update()
    
    def leaveEvent(self, event):
        # ラベルの外に出たらカーソルを初期状態に
        if not getattr(self, '_pan_active', False):
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self._hovering_fixed_rect = False
        super().leaveEvent(event)

    def image_to_label_coords(self, gx, gy):
        """元画像ピクセル -> QLabel座標"""
        if getattr(self.mainwin, "image", None) is None:
            return 0, 0
        img_w = int(self.mainwin.image.width)
        img_h = int(self.mainwin.image.height)

        init_x, init_y, pm_w, pm_h, vr = self._current_geometry()
        px = float(gx) * (pm_w / float(img_w))
        py = float(gy) * (pm_h / float(img_h))

        lx = int(round((px - vr.x()) + init_x))
        ly = int(round((py - vr.y()) + init_y))
        return lx, ly

    def _current_geometry(self):
        """
        ラベル内に実際に描かれているピクスマップの幾何を、その場で算出。
        init_x/init_y は「ラベル中央寄せによる余白」。
        pm_w/pm_h は“ズーム後の全体ピクスマップ”のサイズ。
        vr はその中から現在表示している矩形（パンでずれる分）。
        """
        pm = self.pixmap()
        if pm and not pm.isNull():
            pw, ph = pm.width(), pm.height()
        else:
            pw, ph = self.width(), self.height()

        lw, lh = self.width(), self.height()
        init_x = (lw - pw) // 2 if lw > pw else 0
        init_y = (lh - ph) // 2 if lh > ph else 0

        zoom   = float(getattr(self.mainwin, "zoom_scale", 1.0) or 1.0)
        base_w = getattr(self.mainwin, "base_display_width",  None) or pw
        base_h = getattr(self.mainwin, "base_display_height", None) or ph
        pm_w   = max(1, int(round(base_w * zoom)))
        pm_h   = max(1, int(round(base_h * zoom)))

        vr = getattr(self, "_view_rect_scaled", QtCore.QRect(0, 0, pm_w, pm_h))
        vr = vr.intersected(QtCore.QRect(0, 0, pm_w, pm_h))
        return init_x, init_y, pm_w, pm_h, vr

    def label_to_image_coords(self, lx, ly):
        """QLabel座標 -> 元画像ピクセル"""
        if getattr(self.mainwin, "image", None) is None:
            return 0, 0
        img_w = int(self.mainwin.image.width)
        img_h = int(self.mainwin.image.height)

        init_x, init_y, pm_w, pm_h, vr = self._current_geometry()
        px = float(vr.x() + (float(lx) - init_x))
        py = float(vr.y() + (float(ly) - init_y))

        gx = int(round(px * (float(img_w) / pm_w)))
        gy = int(round(py * (float(img_h) / pm_h)))
        return gx, gy
    
    def _imgrect_to_labelrect(self, r_img: QtCore.QRect | None) -> QtCore.QRect | None:
        """画像座標の QRect -> ラベル座標の QRect に変換する小道具。"""
        if r_img is None:
            return None
        x1, y1 = self.image_to_label_coords(r_img.left(),  r_img.top())
        x2, y2 = self.image_to_label_coords(r_img.right(), r_img.bottom())
        return QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    
    def _fixed_rect_labelcoords(self) -> QtCore.QRect | None:
        """固定枠（画像座標）→ ラベル座標に換算して返す"""
        if self.fixed_crop_rect_img is None:
            return None
        x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(),
                                            self.fixed_crop_rect_img.top())
        x2, y2 = self.image_to_label_coords(self.fixed_crop_rect_img.left() + self.fixed_crop_rect_img.width(),
                                            self.fixed_crop_rect_img.top()  + self.fixed_crop_rect_img.height())
        return QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)) 

    def _maybe_suspend_nudge_for_drag(self):
        """ドラッグ中に微調整パネルを一時非表示にするかの判定。
        固定モードの矩形移動中だけは隠さない。"""
        mw = getattr(self, "mainwin", None) or self.window()
        if not mw or not hasattr(mw, "_suspend_nudge_overlay"):
            return
        # ★例外：固定モードで矩形移動中は隠さない
        if self.fixed_crop_mode and getattr(self, "_moving_fixed_rect", False):
            return
        mw._suspend_nudge_overlay(True)

    def _drag_rect_labelcoords(self) -> QtCore.QRect | None:
        if self.drag_rect_img is None:
            return None
        x1, y1 = self.image_to_label_coords(self.drag_rect_img.left(),
                                            self.drag_rect_img.top())
        x2, y2 = self.image_to_label_coords(self.drag_rect_img.left() + self.drag_rect_img.width(),
                                            self.drag_rect_img.top()  + self.drag_rect_img.height())
        return QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)) 
    
    def _apply_multiple_and_keep_inside(self, x1, y1, x2, y2, mw, mh):
        """
        アンカー優先版：
        - 開始点(ax, ay)は固定
        - 端に届いたら、その端手前で収まる最大の倍数幅/高さ（floor）に切り捨て
        - 画像外からの座標は内側へクランプ
        """
        img = getattr(self.mainwin, "image", None)
        if img is None:
            return QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        img_w, img_h = img.width, img.height

        # アンカー固定（開始点）＆現在点を画像内にクランプ
        ax = max(0, min(x1, img_w))
        ay = max(0, min(y1, img_h))
        bx = max(0, min(x2, img_w))
        by = max(0, min(y2, img_h))

        # それぞれの軸で「端までの残距離」を計算（ドラッグ方向に応じて）
        if bx >= ax:
            avail_x = min(bx - ax, img_w - ax)         # 右方向：右端まで
            snap_w  = (avail_x // mw) * mw             # floor倍数
            left, right = ax, ax + snap_w
        else:
            avail_x = min(ax - bx, ax - 0)             # 左方向：左端まで
            snap_w  = (avail_x // mw) * mw
            left, right = ax - snap_w, ax

        if by >= ay:
            avail_y = min(by - ay, img_h - ay)         # 下方向：下端まで
            snap_h  = (avail_y // mh) * mh
            top, bottom = ay, ay + snap_h
        else:
            avail_y = min(ay - by, ay - 0)             # 上方向：上端まで
            snap_h  = (avail_y // mh) * mh
            top, bottom = ay - snap_h, ay

        # 最終クランプ（理論上不要だが保険）
        left   = max(0, min(left,   img_w))
        right  = max(0, min(right,  img_w))
        top    = max(0, min(top,    img_h))
        bottom = max(0, min(bottom, img_h))

        if right < left:   right  = left
        if bottom < top:   bottom = top

        return QtCore.QRect(int(left), int(top), int(right - left), int(bottom - top))

    def start_fixed_crop(self, crop_size):
        """
        固定切り出し枠を画像中央に配置し、ズーム・パン状態でも正しくラベル上に表示
        """
        self.fixed_crop_mode = True
        self.fixed_crop_size = crop_size

        if self.mainwin.image is None:
            self.fixed_crop_rect = None
            self.update()
            return

        img_w, img_h = self.mainwin.image.width, self.mainwin.image.height
        crop_w, crop_h = crop_size

        # 1. 元画像中央に crop_w×crop_h の矩形を配置（画像ピクセル座標）
        left = (img_w - crop_w) // 2
        top  = (img_h - crop_h) // 2
        
        self.fixed_crop_rect_img = QtCore.QRect(left, top, crop_w, crop_h)

        pm = self.pixmap()
        if not pm:
            self.fixed_crop_rect = None
            self.update()
            return

        # 2. 元画像の矩形 → ラベル座標に変換（ズーム・パン考慮）
        x1, y1 = self.image_to_label_coords(left, top)
        x2, y2 = self.image_to_label_coords(left + crop_w, top + crop_h)

        rect_x = min(x1, x2)
        rect_y = min(y1, y2)
        rect_w = abs(x2 - x1)
        rect_h = abs(y2 - y1)

        self.fixed_crop_rect = QtCore.QRect(rect_x, rect_y, rect_w, rect_h)
        self.update()

    def paintEvent(self, event):
        # まず QLabel 側の描画（背景やpixmap）を済ませる
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        try:
            drew_rect = False  # 矩形を描いたかどうか（調整ハンドル表示の判断に使う）
            has_image = getattr(self.mainwin, "image", None) is not None

            # ===== 画像があるときだけ 矩形（固定/ドラッグ）を描く =====
            if has_image:
                # 固定切り出し枠
                if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
                    x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(), self.fixed_crop_rect_img.top())
                    x2, y2 = self.image_to_label_coords(
                        self.fixed_crop_rect_img.left() + self.fixed_crop_rect_img.width(),
                        self.fixed_crop_rect_img.top()  + self.fixed_crop_rect_img.height()
                    )
                    rect_x = min(x1, x2)
                    rect_y = min(y1, y2)
                    rect_w = abs(x2 - x1)
                    rect_h = abs(y2 - y1)
                    pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, QtCore.Qt.PenStyle.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(QtGui.QColor(30, 120, 255, 60))
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                    drew_rect = True

                # 自由ドラッグ枠
                elif self.drag_rect_img is not None:
                    gx1, gy1 = self.drag_rect_img.left(), self.drag_rect_img.top()
                    gx2 = self.drag_rect_img.left() + self.drag_rect_img.width()
                    gy2 = self.drag_rect_img.top()  + self.drag_rect_img.height()
                    x1, y1 = self.image_to_label_coords(gx1, gy1)
                    x2, y2 = self.image_to_label_coords(gx2, gy2)
                    rect_x = min(x1, x2)
                    rect_y = min(y1, y2)
                    rect_w = abs(x2 - x1)
                    rect_h = abs(y2 - y1)

                    # 倍数モードの色分け
                    pen_color = QtGui.QColor("red")
                    fill_color = QtGui.QColor(255, 0, 0, 60)
                    if getattr(self.mainwin, "multiple_lock_enabled", False):
                        if getattr(self.mainwin, "multiple_w", 64) == 32:
                            pen_color = QtGui.QColor(150, 255, 100)   # 32倍
                            fill_color = QtGui.QColor(150, 255, 100, 60)
                        else:  # 64倍
                            pen_color = QtGui.QColor(0, 170, 80)
                            fill_color = QtGui.QColor(0, 170, 80, 60)

                    painter.setPen(QtGui.QPen(pen_color, 2, QtCore.Qt.PenStyle.SolidLine))
                    painter.setBrush(QtGui.QBrush(fill_color))
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                    drew_rect = True

            # ===== 共通：ジェスチャー軌跡は“画像の有無に関係なく”描く =====
            path = getattr(self, "_gesture_path", None)
            op   = float(getattr(self, "_gesture_opacity", 0.0))
            if path and not path.isEmpty() and op > 0.0:
                # Qt5/6 両対応のレンダーヒント
                rh = getattr(QtGui.QPainter, "Antialiasing", None)
                if rh is None and hasattr(QtGui.QPainter, "RenderHint"):
                    rh = getattr(QtGui.QPainter.RenderHint, "Antialiasing", None)
                if rh is not None:
                    painter.setRenderHint(rh, True)

                # 下地（太め・半透明）
                base = QtGui.QPen(QtGui.QColor(0, 0, 0, int(130 * op)))
                base.setWidth(10)
                base.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                base.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
                painter.setPen(base)
                painter.drawPath(path)

                # 本線（細め・明るい）
                top = QtGui.QPen(QtGui.QColor(255, 170, 0, int(220 * op)))
                top.setWidth(4)
                top.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                top.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
                painter.setPen(top)
                painter.drawPath(path)

            # ===== 調整モードならハンドルを最前面に =====
            if getattr(self, "adjust_mode", False) and drew_rect:
                self._draw_resize_handles(painter)

        finally:
            painter.end()

    def keyPressEvent(self, event):
        # ★ Ctrl+←/→ は自分で処理しない（ショートカットに渡す）
        mods = event.modifiers()
        if (mods & QtCore.Qt.KeyboardModifier.ControlModifier) and \
        (event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right)):
            event.ignore()
            return
        
        key = event.key()
        moved = False
        is_fixed = False
        rect = None
        step = 1  # ← 矢印キーの1ステップ量（必要なら Shift で加速等に拡張可）

        # --- 固定枠：画像座標で平行移動（サイズは不変） ---
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            rect = QtCore.QRect(self.fixed_crop_rect_img)
            if   key == QtCore.Qt.Key.Key_Left:  rect.moveLeft(rect.left() - step);  moved = True
            elif key == QtCore.Qt.Key.Key_Right: rect.moveLeft(rect.left() + step);  moved = True
            elif key == QtCore.Qt.Key.Key_Up:    rect.moveTop (rect.top()  - step);  moved = True
            elif key == QtCore.Qt.Key.Key_Down:  rect.moveTop (rect.top()  + step);  moved = True
            if moved:
                self.fixed_crop_rect_img = QtCore.QRect(rect)
                is_fixed = True

        # --- 通常ドラッグ矩形：画像座標で平行移動 ---
        elif self.drag_rect_img is not None:
            rect = QtCore.QRect(self.drag_rect_img)
            if   key == QtCore.Qt.Key.Key_Left:  rect.translate(-step, 0); moved = True
            elif key == QtCore.Qt.Key.Key_Right: rect.translate( step, 0); moved = True
            elif key == QtCore.Qt.Key.Key_Up:    rect.translate(0, -step); moved = True
            elif key == QtCore.Qt.Key.Key_Down:  rect.translate(0,  step); moved = True
            if moved:
                self.drag_rect_img = QtCore.QRect(rect)
                is_fixed = False

        # --- 矢印キーで何か動かした場合のみ、こちら側でUI同期（シグナルは出さない） ---
        if moved and rect is not None:
            # 1) メイン側の“現在の矩形（画像座標）”を同期
            try:
                self.mainwin._crop_rect_img = QtCore.QRect(rect)
            except Exception:
                pass

            # 2) ラベル座標へ変換（right()/bottom() を使い +1 で幅/高に一致）
            x1, y1 = self.image_to_label_coords(rect.left(),  rect.top())
            x2, y2 = self.image_to_label_coords(rect.right(), rect.bottom())
            label_rect = QtCore.QRect(min(x1, x2), min(y1, y2),
                                    abs(x2 - x1) + 1, abs(y2 - y1) + 1)

            # 3) 内部のラベル矩形も保持（描画側が参照する場合の保険）
            try:
                self.mainwin._crop_rect = QtCore.QRect(label_rect)
            except Exception:
                pass

            # 4) 保存パネル(ActionPanel)は再生成せず位置だけ更新
            panel = getattr(self.mainwin, "_action_panel", None)
            if (
                panel
                and panel.isVisible()
                and not getattr(self.mainwin, "_action_panel_detached", False)
            ):
                try:
                    panel.move(self.mainwin._compute_action_panel_pos(label_rect, is_fixed))
                except Exception:
                    pass

            # 5) 微調整パネル(MovableNudgePanel)は「自動追従モード」のときだけ再配置
            ov = getattr(self.mainwin, "_nudge_overlay", None)
            if (
                ov
                and ov.isVisible()
                and not getattr(self.mainwin, "_nudge_detached", False)
            ):
                try:
                    gap = getattr(self.mainwin, "nudge_gap_px", 4)
                    if hasattr(self.mainwin, "_position_nudge_overlay_above_action"):
                        # ActionPanel に吸着（内部でグローバル座標化＆画面端クランプ）
                        self.mainwin._position_nudge_overlay_above_action(ov, gap)
                    else:
                        # フォールバック（万一ヘルパが無い場合のみ）
                        if panel and panel.isVisible():
                            ag = panel.frameGeometry()
                            ov.move(QtCore.QPoint(ag.left(), ag.top() - ov.height() - gap))
                        else:
                            anchor = self.mapToGlobal(QtCore.QPoint(label_rect.right(), label_rect.top()))
                            x = anchor.x() - ov.width() + 4
                            y = anchor.y() - ov.height() - gap
                            ov.move(QtCore.QPoint(x, y))
                except Exception:
                    pass

            # 6) サイズ表示・プレビュー更新（画像外OK。必要に応じてクリップは内部で）
            try:
                if hasattr(self.mainwin, "update_crop_size_label"):
                    self.mainwin.update_crop_size_label(self.mainwin._crop_rect_img, img_space=True)
            except Exception:
                pass
            try:
                if hasattr(self.mainwin, "safe_update_preview"):
                    QtCore.QTimer.singleShot(0, lambda: self.mainwin.safe_update_preview(self.mainwin._crop_rect_img))
                elif hasattr(self.mainwin, "_schedule_preview"):
                    self.mainwin._schedule_preview(self.mainwin._crop_rect_img)
            except Exception:
                pass

            # 7) 自分で描画更新して終了（イベントはここで止める）
            self.update()
            event.accept()
            return

        # --- 矢印以外や未移動は既定処理へ ---
        super().keyPressEvent(event)

    # --- ジェスチャ用の擬似プロパティ（アニメーション対象） ---
    def getGestureOpacity(self) -> float:
        return float(getattr(self, "_gesture_opacity", 0.0))

    def setGestureOpacity(self, v: float) -> None:
        self._gesture_opacity = max(0.0, min(float(v), 1.0))
        self.update()  # 値が変わるたび再描画

    gestureOpacity = QtCore.pyqtProperty(float, fget=getGestureOpacity, fset=setGestureOpacity)

    # --- ジェスチャの開始・更新・終了処理 ---
    def _start_gesture(self, pos: QtCore.QPoint) -> None:
        if hasattr(self, "_gesture_anim"):
            self._gesture_anim.stop()
        if hasattr(self, "_gesture_fade_timer"):
            self._gesture_fade_timer.stop()  # ← 追加（任意）
        self._gesture_in_progress = True
        self._gesture_start = pos
        self._gesture_current = pos
        p = QtCore.QPointF(pos)
        self._gesture_points = [p]
        self._gesture_path = QtGui.QPainterPath(p)
        self.setGestureOpacity(1.0)

    def _update_gesture(self, pos: QtCore.QPoint) -> None:
        if not self._gesture_points:
            self._start_gesture(pos)
            return
        p_prev = self._gesture_points[-1]
        p_curr = QtCore.QPointF(pos)
        # 中点スムージング
        mid = QtCore.QPointF((p_prev.x() + p_curr.x()) / 2.0,
                             (p_prev.y() + p_curr.y()) / 2.0)
        if len(self._gesture_points) == 1:
            self._gesture_path.lineTo(mid)
        else:
            self._gesture_path.quadTo(p_prev, mid)  # QPointF でOK
        self._gesture_points.append(p_curr)
        self.update()

    def _end_gesture_and_fade(self):
        pts = getattr(self, "_gesture_points", None) or []
        if len(pts) < 2:
            self._gesture_path = QtGui.QPainterPath()
            self._gesture_opacity = 0.0
            self.update()
            return

        # 既存の点列から滑らかパスを作り直す（QPointF 使用）
        p0 = pts[0]
        path = QtGui.QPainterPath(QtCore.QPointF(float(p0.x()), float(p0.y())))
        for i in range(1, len(pts)):
            p_prev = pts[i - 1]
            p_last = pts[i]
            path.quadTo(QtCore.QPointF(float(p_prev.x()), float(p_prev.y())),
                        QtCore.QPointF(float(p_last.x()), float(p_last.y())))
        self._gesture_path = path
        self._gesture_opacity = 1.0

        if not hasattr(self, "_gesture_fade_timer"):
            self._gesture_fade_timer = QtCore.QTimer(self)
            self._gesture_fade_timer.timeout.connect(self._on_gesture_fade_tick)
        self._gesture_fade_timer.start(16)  # ~60fps
        self.update()

    def _on_gesture_fade_tick(self):
        self._gesture_opacity = max(0.0, float(getattr(self, "_gesture_opacity", 0.0)) - 0.08)
        if self._gesture_opacity <= 0.0:
            self._gesture_fade_timer.stop()
            self._gesture_path = QtGui.QPainterPath()  # None にしない（isEmptyで安全）
        self.update()

    def _classify_horizontal_gesture(self) -> str | None:
        """
        軌跡全体から『水平スワイプ』かどうかを判定。
        条件をすべて満たしたとき 'left' / 'right' を返し、そうでなければ None。
        """
        pts = getattr(self, "_gesture_points", None)
        if not pts or len(pts) < 2:
            if not (self._gesture_start and self._gesture_current):
                return None
            dx = float(self._gesture_current.x() - self._gesture_start.x())
            dy = float(self._gesture_current.y() - self._gesture_start.y())
            if abs(dx) < self._gesture_nav_min_dx:
                return None
            angle = math.degrees(math.atan2(abs(dy), abs(dx))) if dx != 0 else 90.0
            if angle > self._gesture_nav_angle_deg:
                return None
            return "right" if dx > 0 else "left"

        # 1) 直線距離・経路長・外接矩形
        path_len = 0.0
        horiz = 0.0
        vert  = 0.0
        minx = maxx = float(pts[0].x())
        miny = maxy = float(pts[0].y())
        for i in range(1, len(pts)):
            dx = float(pts[i].x() - pts[i-1].x())
            dy = float(pts[i].y() - pts[i-1].y())
            path_len += math.hypot(dx, dy)
            horiz += abs(dx)
            vert  += abs(dy)
            x = float(pts[i].x()); y = float(pts[i].y())
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)

        dx_total = float(pts[-1].x() - pts[0].x())
        dy_total = float(pts[-1].y() - pts[0].y())
        straight_len = math.hypot(dx_total, dy_total)
        bbox_w = maxx - minx
        bbox_h = maxy - miny
        angle  = math.degrees(math.atan2(abs(dy_total), abs(dx_total))) if dx_total != 0 else 90.0

        # 2) 方向一貫性（各セグメントの方向が全体ベクトルにどれだけ揃っているか）
        if straight_len == 0:
            return None
        ux, uy = dx_total / straight_len, dy_total / straight_len  # 全体方向の単位ベクトル
        cos_sum = 0.0
        segs = 0
        for i in range(1, len(pts)):
            dx = float(pts[i].x() - pts[i-1].x())
            dy = float(pts[i].y() - pts[i-1].y())
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                continue
            cos = (dx / seg_len) * ux + (dy / seg_len) * uy  # [-1,1]
            cos_sum += max(0.0, cos)  # 逆向きは0として加算
            segs += 1
        cos_avg = (cos_sum / segs) if segs else 0.0

        # 3) しきい値チェック（曲がり過ぎ・蛇行・縦成分優勢・角度）
        if abs(dx_total) < self._gesture_nav_min_dx:
            return None
        if (path_len / max(straight_len, 1e-6)) > self._gesture_nav_curv_max:
            return None                 # 曲がりくねっている（例のような弧を描く動き）
        if cos_avg < self._gesture_nav_dir_consistency:
            return None                 # 方向がバラバラ（往復/蛇行）
        if horiz < vert * self._gesture_nav_ratio:
            return None                 # 縦揺れが大きすぎる
        if bbox_w < bbox_h * self._gesture_nav_bbox_ratio:
            return None                 # 外接矩形が縦長に近い
        if angle > self._gesture_nav_angle_deg:
            return None                 # 最終ベクトルが水平から離れすぎ

        return "right" if dx_total > 0 else "left"

    def _clear_gesture(self) -> None:
        self._gesture_points.clear()
        self._gesture_path = QtGui.QPainterPath()
        self.update()

class _ThumbTask(QtCore.QRunnable):
    def __init__(self, fn, row):
        super().__init__()
        self.fn = fn
        self.row = row

    def run(self):
        self.fn(self.row)

class _DirOverlayTask(QtCore.QRunnable):
    """
    フォルダ/zip内の画像から“1枚だけ”をPNGにして dirOverlayReady を投げる（軽量版）
    ※ 超巨大画像はプレビューをあきらめてスキップする
    """
    def __init__(self, model, row: int, path: str, gen: int):
        super().__init__()
        self.model = model
        self.row = row
        self.path = path
        self.gen = gen

    def run(self):
        from PIL import Image, ImageOps
        from io import BytesIO

        # ★ フォルダが切り替わっていたら即終了
        if self.gen != getattr(self.model, "_gen", 0):
            return

        # ★ 計測開始
        t0 = _dbg_time(f"[overlay] start row={self.row} path={self.path}")

        # これ以上デカい画像は中身プレビューをあきらめる（フォルダ/ZIPアイコンだけ）
        MAX_PIXELS_FOR_OVERLAY = 100_000_000  # 例: 約1000万ピクセル

        # dirpath 直下 → サブフォルダ…の順に、自然順で画像を1枚だけ探す
        def first_image_recursive(
            dirpath: str,
            depth: int = 0,
            max_depth: int = 4
        ) -> str | None:
            """dirpath 直下に画像が無ければ、子フォルダをたどりながら再帰的に 1 枚探す"""

            # ★ フォルダ切り替え後は探索自体を打ち切る
            if self.gen != getattr(self.model, "_gen", 0):
                return None

            if depth > max_depth:
                return None

            try:
                entries = vfs_listdir(dirpath)
            except Exception:
                return None

            # 1) 直下の画像ファイルを探す
            files = [
                e["uri"]
                for e in entries
                if not e.get("is_dir") and is_image_name(e.get("uri", ""))
            ]
            if files:
                try:
                    files.sort(key=natural_key)
                except Exception:
                    files.sort()
                return files[0]

            # 2) 直下に画像が無ければ、サブフォルダを自然順で巡回
            dirs = [e["uri"] for e in entries if e.get("is_dir")]
            if not dirs:
                return None
            try:
                dirs.sort(key=natural_key)
            except Exception:
                dirs.sort()

            for sub in dirs:
                # ★ ここでも世代が変わってたら即やめる
                if self.gen != getattr(self.model, "_gen", 0):
                    return None
                pick = first_image_recursive(sub, depth + 1, max_depth)
                if pick:
                    return pick
            return None

        # 1) このフォルダ/zip の直下 → 子フォルダ…の順で 1 枚探す
        pick = first_image_recursive(self.path, 0, 3)
        if not pick:
            _dbg_time(f"[overlay] end   row={self.row} path={self.path} (no pick)", t0)
            return  # 何も無ければ何もしない

        # ★ 画像を開く前にも一応チェック（フォルダ切り替え済みなら無駄仕事しない）
        if self.gen != getattr(self.model, "_gen", 0):
            _dbg_time(f"[overlay] end   row={self.row} path={self.path} (gen changed before open)", t0)
            return

        # サムネイルサイズから、オーバーレイ用の最大ピクセルを決める
        try:
            max_px = max(self.model.thumb_size)
            max_px = max(256, min(512, int(max_px * 1.2)))
        except Exception:
            max_px = 256

        # 2) 画像を軽く開いて、デカすぎればスキップ、それ以外は縮小＆PNG化
        try:
            if is_zip_uri(pick):
                # zip / rar / 7z 内ファイル
                zp, inner = parse_zip_uri(pick)
                zf = _open_zip_cached(zp)

                # ケース非依存名を解決
                inner2 = _zip_resolve_inner(zp, inner)
                try:
                    f = zf.open(inner2, "r")
                except KeyError:
                    f = zf.open(inner, "r")

                with f:
                    im = Image.open(f)
                    w, h = im.size
                    if w * h > MAX_PIXELS_FOR_OVERLAY:
                        log_debug(f"[overlay] skip huge image in zip ({w}x{h}) for {pick!r}")
                        _dbg_time(f"[overlay] end   row={self.row} path={self.path} (huge zip image)", t0)
                        return

                    # ★ ここでも途中でフォルダ変わってたらやめる
                    if self.gen != getattr(self.model, "_gen", 0):
                        _dbg_time(f"[overlay] end   row={self.row} path={self.path} (gen changed mid-zip)", t0)
                        return

                    if getattr(im, "format", None) == "JPEG":
                        try:
                            im.draft("RGB", (max_px, max_px))
                        except Exception:
                            pass
                    im.load()
            else:
                # 物理ファイル
                im = Image.open(pick)
                w, h = im.size
                if w * h > MAX_PIXELS_FOR_OVERLAY:
                    log_debug(f"[overlay] skip huge image ({w}x{h}) for {pick!r}")
                    _dbg_time(f"[overlay] end   row={self.row} path={self.path} (huge file)", t0)
                    return

                if self.gen != getattr(self.model, "_gen", 0):
                    _dbg_time(f"[overlay] end   row={self.row} path={self.path} (gen changed mid-file)", t0)
                    return

                if getattr(im, "format", None) == "JPEG":
                    try:
                        im.draft("RGB", (max_px, max_px))
                    except Exception:
                        pass
                im.load()

            # カラーモード統一
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")

            # 先に縮小してから EXIF 回転（小さい画像を回転するので軽い）
            im.thumbnail((max_px, max_px), Image.BILINEAR)
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            bio = BytesIO()
            im.save(bio, format="PNG")
            png_bytes = bio.getvalue()
        except Exception as e:
            log_debug("[overlay] error while building dir overlay:", e)
            _dbg_time(f"[overlay] end   row={self.row} path={self.path} (error)", t0)
            return

        # ★ emit 直前にも世代チェック（切り替え後のゴミを飛ばす）
        if self.gen != getattr(self.model, "_gen", 0):
            _dbg_time(f"[overlay] end   row={self.row} path={self.path} (gen changed before emit)", t0)
            return

        # 3) モデルへ通知（UIスレッド側で _apply_dir_overlay → _compose_folder_pm が走る）
        try:
            self.model.dirOverlayReady.emit(
                self.row, self.path, png_bytes, self.gen
            )
        except Exception:
            # モデルが既に破棄されている等
            pass

        # ★ 計測終了
        _dbg_time(f"[overlay] end   row={self.row} path={self.path}", t0)

class ThumbnailListModel(QtCore.QAbstractListModel):
    # ワーカ→UI 受け渡し
    thumbReady = QtCore.pyqtSignal(int, str, bytes, int)  # row, path, png, gen
    dirReady = QtCore.pyqtSignal(int, str)                # ベースのフォルダ絵を即時セット
    dirOverlayReady = QtCore.pyqtSignal(int, str, bytes, int)  # 中身プレビューPNGが出来たら上書き

    # 共有キャッシュ（パス→(mtime, QPixmap)）
    _cache: dict[str, tuple[object, bytes]] = {}

    def __init__(self, image_list, thumb_size=(80, 120)):
        super().__init__()
        self.image_list = list(image_list)
        self.thumb_size = thumb_size
        self.thumbnails = [None] * len(self.image_list)

        self._pool = QtCore.QThreadPool.globalInstance()  # 共有プールを使う

        # 共有キャッシュ、保留行、強制再生成フラグ
        self._cache: dict[str, tuple[object, bytes]] = {}
        self._pending_rows: set[int] = set()
        self._force_rebuild: set[str] = set()

        # ★ このモデル“インスタンス”に固有の世代トークン
        #    （新インスタンスごとに変わればOK。加算式は不要）
        self._gen = (id(self) & 0x7fffffff)

        # サムネ完成シグナル（row, path, png_bytes, gen）
        self.thumbReady.connect(self._apply_thumb)
        self.dirReady.connect(self._set_dir_thumb)
        self.dirOverlayReady.connect(self._apply_dir_overlay) 

    def invalidate_path(self, path: str):
        # VFS/zip:// も壊さない統一キー
        try:
            key = norm_vpath(path)
        except Exception:
            key = path

        # 既存キャッシュ破棄＆次の1回は強制再生成
        try:
            self._cache.pop(key, None)
            self._force_rebuild.add(key)
        except Exception:
            pass

        # row を“同じ正規化キー”で探す
        row = -1
        try:
            for i, p in enumerate(self.image_list):
                try:
                    if norm_vpath(p) == key:
                        row = i
                        break
                except Exception:
                    if p == path:
                        row = i
                        break
        except Exception:
            row = -1

        if row < 0:
            return

        # 既存スロットを空にして再キュー
        try:
            if 0 <= row < len(self.thumbnails):
                self.thumbnails[row] = None
        except Exception:
            pass

        try:
            self._pending_rows.discard(row)
            self._pending_rows.add(row)
            self._pool.start(_ThumbTask(self._generate_thumb, row))
        except Exception:
            pass

        # 表示更新通知
        try:
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])
        except Exception:
            pass

    def _system_dir_icon(self, path: str) -> QtGui.QIcon:
        """サムネ用：フォルダ or zip のアイコンを返す
        - 物理フォルダ          → 通常のフォルダアイコン
        - .zip 本体 / zip ルート → OS の zip アイコン
        - zip 内のサブフォルダ   → 通常のフォルダアイコン
        - zip 内の zip/CBZ       → OS の zip アイコン
        """
        import os
        prov = getattr(self, "_icon_provider", None)
        if prov is None:
            self._icon_provider = QtWidgets.QFileIconProvider()
            prov = self._icon_provider

        ico = QtGui.QIcon()
        target_info = None

        try:
            if is_zip_uri(path):
                zp, inner = parse_zip_uri(path)
                # inner == "" → zip のルート (zip://...! )
                if inner == "" or inner == "/":
                    target_info = QtCore.QFileInfo(zp)
                else:
                    # zip内の "xxx.zip"/"xxx.cbz" は zip アイコンを使う
                    if is_archive_name(inner):
                        base = os.path.basename(inner)
                        target_info = QtCore.QFileInfo(base)
                    # それ以外(inner がサブフォルダなど)は target_info=None のまま
            elif is_archive_file(path):
                # 物理 zip パス（vfs_is_dir でフォルダ扱いしているケース）
                target_info = QtCore.QFileInfo(path)
            else:
                # ふつうの物理フォルダ
                target_info = QtCore.QFileInfo(path)

            if target_info is not None:
                ico = prov.icon(target_info)
        except Exception:
            ico = QtGui.QIcon()

        if ico.isNull():
            # ここに来るのは「zip 内フォルダ」など → 通常のフォルダアイコン
            st = QtWidgets.QApplication.style()
            sp = getattr(QtWidgets.QStyle, "SP_DirIcon",
                         QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
            ico = st.standardIcon(sp)

        return ico

    @QtCore.pyqtSlot(int, str)
    def _set_dir_thumb(self, row: int, path: str):
        # モデル更新中のズレをガード
        if not (0 <= row < len(self.image_list)):
            return
        if self.image_list[row] != path:
            return

        # まずは“フォルダだけ”の絵を即時反映
        pm = self._compose_folder_pm(path, None)   # ★ overlay なし
        self.thumbnails[row] = pm
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])

        # 中身プレビュー（フォルダ内の画像から1枚作る）を非同期で作成 → 後で上書き
        try:
            self._pool.start(_DirOverlayTask(self, row, path, self._gen))
        except Exception:
            pass
        
    @QtCore.pyqtSlot(int, str, bytes, int)
    def _apply_dir_overlay(self, row: int, path: str, png_bytes: bytes, gen: int):
        # 世代ズレや差し替え後のインデックスずれをガード
        if gen != getattr(self, "_gen", 0):
            return
        if not (0 <= row < len(self.image_list)):
            return
        # パス確認（差し替えレース対策）
        cur_path = self.image_list[row]
        if cur_path != path:
            return

        img = QtGui.QImage.fromData(png_bytes, "PNG")
        pm = self._compose_folder_pm(path, img)  # フォルダに合成
        self.thumbnails[row] = pm
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])

    def _compose_folder_pm(self, path: str, overlay_img: QtGui.QImage | None) -> QtGui.QPixmap:
        w, h = self.thumb_size
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtCore.Qt.GlobalColor.transparent)

        # ベースのフォルダ / zip アイコン
        icon = self._system_dir_icon(path)
        side = int(min(w, h) * 0.90)
        try:
            base = icon.pixmap(QtCore.QSize(side, side))
        except Exception:
            base = icon.pixmap(side, side)

        target = QtCore.QRect((w - side) // 2, (h - side) // 2 - 2, side, side)

        p = QtGui.QPainter(pm)
        try:
            rh = (getattr(QtGui.QPainter, "Antialiasing", None) or
                getattr(QtGui.QPainter.RenderHint, "Antialiasing", None))
            if rh is not None:
                p.setRenderHint(rh, True)

            # 1) フォルダ自体
            p.drawPixmap(target, base, base.rect())

            # 2) 中身プレビュー（あれば）
            if overlay_img is not None and not overlay_img.isNull():
                # アイコン上の「オーバーレイ専用枠」
                # - 左右 12% だけ余白を空ける
                # - 上から少し下げて、下側も多めに残す（ジッパー等が見えるように）
                margin_x      = int(side * 0.12)
                margin_top    = int(side * 0.32)
                margin_bottom = int(side * 0.22)

                overlay_rect = QtCore.QRect(
                    target.left() + margin_x,
                    target.top()  + margin_top,
                    side - margin_x * 2,
                    max(1, side - margin_top - margin_bottom),
                )

                img = overlay_img
                iw, ih = img.width(), img.height()
                if iw > 0 and ih > 0 and overlay_rect.width() > 0 and overlay_rect.height() > 0:
                    # 枠に収まるように contain スケーリング
                    scale = min(overlay_rect.width() / iw, overlay_rect.height() / ih)
                    dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))

                    dst = QtCore.QRect(
                        overlay_rect.center().x() - dw // 2,
                        overlay_rect.center().y() - dh // 2,
                        dw, dh,
                    )

                    # 角丸クリップ
                    path = QtGui.QPainterPath()
                    r = max(2, int(min(dst.width(), dst.height()) * 0.06))
                    path.addRoundedRect(QtCore.QRectF(dst), r, r)
                    p.save()
                    p.setClipPath(path)
                    p.drawImage(dst, img)
                    p.restore()
        finally:
            p.end()

        return pm

    def _generate_thumb(self, row: int):
        if not (0 <= row < len(self.image_list)):
            return

        path = self.image_list[row]
        key = norm_vpath(path)

        try:
            is_dir = vfs_is_dir(path)
        except Exception:
            is_dir = False

        if is_dir:
            # ★ フォルダ／圧縮ファイルアイコンは別経路（オーバーレイ側）で処理
            self.dirReady.emit(row, path)
            return

        # --- ここからは“画像ファイル”だけ ---
        # デバッグ用タイマー開始
        t0 = _dbg_time(f"[thumb] gen start row={row} key={key}")

        # 署名取得（ここは今の自分のコードに合わせて：
        #   ・もう _sig_for(path) に変えているならそのままでOK
        #   ・まだ os.stat(path) なら、まずは os.stat 版のままでも良い）
        try:
            # ★ 圧縮ファイル内も含めてちゃんとキャッシュ効かせたいなら _sig_for を使う
            sig = _sig_for(path)
        except Exception as e:
            log_debug(f"[thumb] sig error for {path!r}: {e}")
            sig = None

        # キャッシュヒットチェック
        if key not in self._force_rebuild:
            cached = self._cache.get(key)
            if sig is not None and cached and cached[0] == sig:
                # 通常の「署名一致ヒット」
                png_bytes = cached[1]
                log_debug(f"[thumb] HIT  row={row} key={key}")
                self.thumbReady.emit(row, path, png_bytes, self._gen)
                _dbg_time(f"[thumb] gen end (HIT) row={row} key={key}", t0)
                return

            # ★ 署名が取れなかった（sig is None）場合でも、
            #    一度キャッシュしてある絵があればそれをそのまま再利用する。
            #    → 物理ファイルが消えていても「最後に見えていたサムネ」を維持できる。
            if sig is None and cached:
                png_bytes = cached[1]
                log_debug(f"[thumb] HIT(reuse) row={row} key={key} (sig=None)")
                self.thumbReady.emit(row, path, png_bytes, self._gen)
                _dbg_time(f"[thumb] gen end (HIT-reuse) row={row} key={key}", t0)
                return

        log_debug(f"[thumb] MISS row={row} key={key} sig={sig}")

        # サムネ生成（画像のみ）
        try:
            img = make_fixed_thumbnail_any(path, self.thumb_size)
        except Exception:
            from PIL import Image
            img = Image.new("RGB", self.thumb_size, (60, 60, 60))

        from io import BytesIO
        bio = BytesIO()
        img.save(bio, format="PNG")
        png_bytes = bio.getvalue()

        if sig is not None:
            self._cache[key] = (sig, png_bytes)
            log_debug(f"[thumb] PUT  row={row} key={key}")

        self.thumbReady.emit(row, path, png_bytes, self._gen)
        _dbg_time(f"[thumb] gen end (MISS) row={row} key={key}", t0)

    @QtCore.pyqtSlot(int, str, bytes, int)
    def _apply_thumb(self, row: int, path: str, png_bytes: bytes, gen: int) -> None:
        # 1) 古いジョブは無視（フォルダ/リスト更新後の遅延対策）
        if gen != getattr(self, "_gen", 0):
            return

        # 2) 行範囲ガード（削除後に index が詰まっても安全）
        if not (0 <= row < len(self.image_list)):
            return

        # 3) パス一致確認（行がずれた/差し替わったレース対策）
        cur = self.image_list[row]
        norm = lambda p: os.path.normcase(os.path.abspath(p or ""))
        if norm(cur) != norm(path):
            # 見つかれば新しい行に差し替え。無ければ破棄。
            try:
                row = next(i for i, p in enumerate(self.image_list) if norm(p) == norm(path))
            except StopIteration:
                return

        # 4) thumbnails の長さも必ず image_list に合わせる（念のため）
        if len(self.thumbnails) < len(self.image_list):
            self.thumbnails.extend([None] * (len(self.image_list) - len(self.thumbnails)))
        if not (0 <= row < len(self.thumbnails)):
            return

        # 5) 画像に反映
        img = QtGui.QImage.fromData(png_bytes, "PNG")
        pix = QtGui.QPixmap.fromImage(img)
        self.thumbnails[row] = pix

        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])

    # ---- Qt Model 標準 ----
    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.image_list)

    def data(self, index, role):
        # インデックス無効は即 None
        if not index.isValid():
            return None

        row = index.row()
        total = len(self.image_list)

        # ★安全ガード：範囲外は即 None（リセット直後の古い要求対策）
        if row < 0 or row >= total:
            return None

        path = self.image_list[row]
        is_dir = False
        try:
            is_dir = vfs_is_dir(path)
        except Exception:
            pass

        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            # 同じく安全ガード：thumbnails 側も範囲内のみ参照
            thumb = None
            try:
                thumb = self.thumbnails[row]
            except Exception:
                thumb = None

            if thumb is None:
                # プレースホルダ（薄灰）
                placeholder = QtGui.QPixmap(self.thumb_size[0], self.thumb_size[1])
                placeholder.fill(QtGui.QColor(60, 60, 60))

                # ★ジョブ投入も“範囲内の行だけ”
                if row not in self._pending_rows:
                    self._pending_rows.add(row)
                    # ワーカは row 番号に依存するので、投入前にもう一度行/パスを固定化しておくのが堅い
                    self._pool.start(_ThumbTask(self._generate_thumb, row))
                return QtGui.QIcon(placeholder)

            return QtGui.QIcon(thumb)

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            try:
                name = vfs_display_name(path, bool(is_dir))
            except Exception:
                name = str(path)
            return name

        if role == QtCore.Qt.ItemDataRole.UserRole:
            # ビュー側が安全に判定できるよう dict で返す
            return {"path": path, "is_dir": bool(is_dir)}

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if is_dir:
                try:
                    name = os.path.basename(path)
                except Exception:
                    name = str(path)
                return f"フォルダ: {name}\nダブルクリックで開く"
            else:
                # 画像メタは失敗しても落ちないように
                try:
                    img = open_image_any(path)
                    size_str = f"{img.width} x {img.height}"
                except Exception:
                    size_str = "取得失敗"
                try:
                    size_kb = os.path.getsize(path) // 1024
                except Exception:
                    size_kb = "?"
                return (f"ファイル名: {os.path.basename(path)}\n"
                        f"解像度: {size_str}\n"
                        f"サイズ: {size_kb} KB")

        return None

    def reset_items(self, image_list, thumb_size=None):
        """同一インスタンスのまま内容を差し替える。フォルダ遷移で使う。"""
        t0 = _dbg_time(f"[thumb] reset_items gen={getattr(self, '_gen', 0)} size={len(image_list)}")

        self.beginResetModel()

        # ★いまキューに溜まっている古いフォルダ用サムネ生成タスクをキャンセル
        try:
            self._pool.clear()   # まだ開始してない QRunnable を全部捨てる
        except Exception:
            pass

        # 世代を進めて、旧タスクの結果を無視させる
        self._gen = (self._gen + 1) & 0x7fffffff

        # リストを差し替え（外部参照を断つため必ず list() でコピー）
        self.image_list = list(image_list)
        if thumb_size is not None:
            self.thumb_size = thumb_size

        # 表示用サムネ配列と、進行中フラグをリセット
        self.thumbnails = [None] * len(self.image_list)
        try:
            self._pending_rows.clear()
        except Exception:
            self._pending_rows = set()

        # 強制再生成フラグはクリア
        try:
            self._force_rebuild.clear()
        except Exception:
            self._force_rebuild = set()

        self.endResetModel()

        _dbg_time(f"[thumb] reset_items done gen={self._gen} size={len(self.image_list)}", t0)

    def remove_paths(self, paths: list[str]) -> None:
        """
        現在の image_list から指定パスをまとめて削除して詰め直す。

        - paths: 削除したいエントリのパス文字列のリスト（フォルダ/zip/画像全部含む）
        """
        if not paths:
            return

        # 完全一致で比較（zip:// などの仮想パスもそのまま扱う）
        remove_set = set(paths)

        new_image_list = [p for p in self.image_list if p not in remove_set]
        if len(new_image_list) == len(self.image_list):
            # 何も変わらなかった
            return

        # reset_items が thumbnails / _pending_rows / _gen などを安全に張り替えてくれる
        self.reset_items(new_image_list, thumb_size=self.thumb_size)


class InfoBanner(QtWidgets.QFrame):
    """
    中央下部の「アイコン + 名前(フォルダ/zip/画像)」用バナー。
    ・長い名前は ElideMiddle で「…」省略
    ・でもそのせいでレイアウトや左右のカラムを押し広げないよう、
      sizeHint を控えめに返す
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("InfoBanner")

        self._full_text = ""

        self.setStyleSheet("""
            QFrame#InfoBanner {
                background-color: #353535;   /* 矩形サイズパネルと同じ灰色 */
                color: #dddddd;
                border-top: 1px solid #444444;
            }
        """)

        # --- 内部パーツ ---
        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setFixedSize(18, 18)
        self._icon_label.setScaledContents(True)

        self._text_label = QtWidgets.QLabel()
        self._text_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter |
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._text_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.NoTextInteraction
        )

        # ★ InfoBanner の文字を少し大きく
        f = self._text_label.font()
        f.setPointSize(f.pointSize() + 2)   # もっと大きくしたければ +2 とかに
        self._text_label.setFont(f)

        self._icon_label.setStyleSheet("background: transparent; border: none;")
        self._text_label.setStyleSheet("background: transparent; border: none;")

        # レイアウト：左右に stretch は入れず、全体を中央寄せ
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(6)
        lay.addWidget(self._icon_label)
        lay.addWidget(self._text_label)
        # アイコン＋テキストの「かたまり」を中央に配置
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # 「横には広がってもいいけど、欲張りすぎない」感じのポリシー
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        self.setSizePolicy(sp)

    # ---- API ----
    def set_content(self, icon: QtGui.QIcon | None, text: str | None):
        """アイコンとテキストをまとめてセット"""
        # アイコン
        if icon and not icon.isNull():
            sz = self._icon_label.size()
            pm = icon.pixmap(sz.width(), sz.height())
            self._icon_label.setPixmap(pm)
        else:
            self._icon_label.clear()

        # テキスト（まずフルテキストを覚える）
        self._full_text = text or ""
        # 一旦フルテキストをそのまま入れる
        self._text_label.setText(self._full_text)
        # 実際の幅に合わせて省略（※ resizeEvent 側でも再実行される）
        self._apply_elide()

    def clear(self):
        self._full_text = ""
        self._icon_label.clear()
        self._text_label.clear()

    # ---- QWidget overrides ----
    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        # 幅が変わったら、そのたびに省略し直す
        self._apply_elide()

    def sizeHint(self) -> QtCore.QSize:
        """
        QSplitter に「そんなに横幅要らないよ」と伝えるためのヒント。
        長さ 1000 文字とかでもここは 400px くらいを返す。
        """
        h = max(22, self.fontMetrics().height() + 4)
        return QtCore.QSize(400, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        h = max(22, self.fontMetrics().height() + 4)
        return QtCore.QSize(50, h)

    # ---- 内部ヘルパ ----
    def _apply_elide(self):
        """現在の幅に合わせて _full_text を … で省略して表示する"""
        if not self._full_text:
            self._text_label.clear()
            return

        # InfoBanner 全体の幅を基準に、テキストに使える幅をざっくり見積もる
        w = self.width()
        if w <= 0:
            return

        # 左右マージンぶんを引く
        w -= 8 * 2
        # アイコンが出ているなら、その幅＋隙間ぶんも引く
        if self._icon_label.pixmap() is not None:
            w -= self._icon_label.width()
            w -= 6  # spacing

        w = max(10, w)

        fm = self._text_label.fontMetrics()
        elided = fm.elidedText(
            self._full_text,
            QtCore.Qt.TextElideMode.ElideMiddle,
            w
        )
        self._text_label.setText(elided or self._full_text)

class CropperApp(QtWidgets.QMainWindow):
    # 取り扱う画像拡張子のリスト
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')
   
    def __init__(self):
        super().__init__()
        self._suspend_chain_clear = 0

        # 欠損画像処理用のフラグ
        self._handling_missing_image = False   # 欠損画像ハンドリング中かどうか（サムネからの再入防止）
        self._missing_all_ok = False           # 「この連続分を全てOK」が有効な間 True

        # ★ 中ボタンで「今はパネル要らない」を覚えておくフラグ
        self._panel_hidden_by_user = False

        # ★ パネル/微調整パネルが矩形に「自動追従するかどうか」
        #   False: 矩形移動に追従する（デフォルト）
        #   True : ユーザーがドラッグで位置を決めたので、以降は追従しない
        self._action_panel_detached = False
        self._nudge_detached = False

        # ★ パネルを隠した時点の矩形（画像座標）を記憶しておく（Nudge 再配置の判定用）
        self._panel_hide_rect_img = None

        # ポータブルINI + 保存色の復元
        import os, sys
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)            # exe の場所
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__)) # .py の場所

        config_dir = os.path.join(app_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        ini_path = os.path.join(config_dir, "settings.ini")

        self.settings = QtCore.QSettings(ini_path, QtCore.QSettings.Format.IniFormat)
    
        self.overwrite_mode   = bool(self.settings.value("overwrite_mode", False, type=bool))

        self.show_save_dialog_on_load = bool(self.settings.value("show_save_dialog_on_load", False, type=bool))
        self.save_dest_mode  = str(self.settings.value("save_dest_mode", "same"))
        self.save_custom_dir = str(self.settings.value("save_custom_dir", "", type=str))
        # 現在の保存先表示を初期化
        self.save_folder = str(getattr(self, "save_folder", ""))

        self.thumb_scroll_step = int(self.settings.value("thumb_scroll_step", 50,   type=int))
        self.hq_zoom          = bool(self.settings.value("hq_zoom", False,          type=bool))

        # ★ ループ状態を設定から復元（なければ False）
        self._thumb_loop_enabled = bool(self.settings.value("thumb_loop_enabled", False, type=bool))

        # --- DnD後の保存先ダイアログ遅延用フラグ（Explorerフリーズ回避） ---
        self._defer_save_dialog_once = False   # ← ここに追加

        # 背景色（既定は白）→ 保存されていれば上書き
        self.preview_bg_color = QtGui.QColor("#ffffff")
        saved = self.settings.value("preview_bg_color", type=str)
        if saved:
            c = QtGui.QColor(saved)
            if c.isValid():
                self.preview_bg_color = c

        # QColorDialog の Custom colors を復元
        self._load_custom_colors()

        #メイン画面の背景色の復元
        self.view_bg_color = QtGui.QColor("#191919")
        saved3 = self.settings.value("view_bg_color", type=str)
        if saved3:
            c3 = QtGui.QColor(saved3)
            if c3.isValid():
                self.view_bg_color = c3
        
        # （色チップを使っているなら初期表示も保存色で）
        if hasattr(self, "preview_color_chip"):
            self.preview_color_chip.setStyleSheet(
                f"background:{self.preview_bg_color.name(QtGui.QColor.NameFormat.HexArgb)}; "
                "border:1px solid #666; border-radius:3px;"
            )

        self.setAcceptDrops(True)

        self.shortcut_delete = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        self.shortcut_delete.activated.connect(self.delete_current_image)

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1000, 700)
        self.setStyleSheet("QMainWindow { background: #121212; }")
        self.base_display_width = None
        self.base_display_height = None
        self.save_folder = None

        # --- 左カラム：プレビュー用 QLabel 宣言 ---
        self.preview_label = SquareLabel(self, min_side=256, base_side=512)

        #プレビュー背景の初期色（白）

        self._set_preview_placeholder()

        self._apply_preview_bg_to_label() 

        # --- 追加情報パネル（サブパネル） ---
        self.sub_panel = QtWidgets.QWidget()
        #self.sub_panel.setMinimumHeight(370)
        self.sub_panel.setMinimumHeight(0)
        self.sub_panel.setSizePolicy(QSizePolicy.Policy.Preferred,
                                     QSizePolicy.Policy.MinimumExpanding)
        self.sub_panel.setStyleSheet("background: #353535; border-radius: 10px;")
        self.sub_panel_layout = QtWidgets.QVBoxLayout(self.sub_panel)
        self.sub_panel_layout.setContentsMargins(12, 10, 12, 10)

        # 解像度ラベル
        self.crop_size_label = QtWidgets.QLabel("0 x 0")
        self.crop_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.crop_size_label.setStyleSheet("""
            QLabel {
                color: #222;
                background: #fff;
                border: 1px solid #2c405a;
                border-radius: 8px;
                padding: 10px 20px;   /* ← フォント関連は消す */
            }
        """)

        # ← ここでフォントを“コードで”指定（pixel 単位が安全）
        f = self.crop_size_label.font()
        f.setPixelSize(32)
        f.setBold(True)
        # 好みで候補を並べる（最初に見つかったものが使われます）
        try:
            f.setFamilies(["Consolas", "Cascadia Mono", "Source Code Pro", "Meiryo", "Monospace"])
        except AttributeError:
            f.setFamily("Consolas")  # Qt の古い版向けフォールバック
        self.crop_size_label.setFont(f)

        # ---- このフォントで必要幅を計算 ----
        fm = QtGui.QFontMetrics(self.crop_size_label.font())
        reserve = fm.horizontalAdvance("99999 x 99999")  # 想定最大桁ぶん確保
        padding_lr = 40  # QSSの左右padding 20px×2
        self.crop_size_label.setFixedWidth(reserve + padding_lr + 2)

        # ★ 画像の水平反転ボタン（2行表示＆正方形）
        self.btn_flip_horizontal = QtWidgets.QPushButton("水平\n反転")
        self.btn_flip_horizontal.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # 正方形っぽくちょっと大きめに
        self.btn_flip_horizontal.setFixedSize(72, 72)
        # いちおうサイズポリシーも固定にしておく
        self.btn_flip_horizontal.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.btn_flip_horizontal.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.btn_flip_horizontal.setStyleSheet("""
        QPushButton {
            background:#2b2b2b;
            color:#e0e0e0;
            border:2px solid #e6c15a;      /* ← 黄色っぽい外枠 */
            border-radius:10px;
            padding:0px;
            font-size: 11pt;
        }
        QPushButton:hover  {
            background:#454545;        /* ← ちょっと明るく */
            border-color:#ffe58a;      /* ← 枠ももう一段明るく */
        }
        QPushButton:pressed{
            background:#222222;
            border-color:#cfa744;          /* 押したとき少し濃く */
        }
        """)
        self.btn_flip_horizontal.clicked.connect(self.on_flip_horizontal)

        # ★ 垂直反転ボタン（仕様は水平反転と同じ）
        self.btn_flip_vertical = QtWidgets.QPushButton("垂直\n反転")
        self.btn_flip_vertical.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_flip_vertical.setFixedSize(72, 72)
        self.btn_flip_vertical.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_flip_vertical.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.btn_flip_vertical.setStyleSheet("""
        QPushButton {
            background:#2b2b2b;
            color:#e0e0e0;
            border:2px solid #e6c15a;
            border-radius:10px;
            padding:0px;
            font-size: 11pt;
        }
        QPushButton:hover  {
            background:#333333;
            border-color:#f4e38a;
        }
        QPushButton:pressed{
            background:#222222;
            border-color:#cfa744;
        }
        """)
        self.btn_flip_vertical.clicked.connect(self.on_flip_vertical)

        # ★ 90度回転ボタン（左右で別動作）
        self.btn_rotate_90 = DualRotateButton(self.sub_panel)
        self.btn_rotate_90.leftClicked.connect(self.on_rotate_left_90)
        self.btn_rotate_90.rightClicked.connect(self.on_rotate_right_90)

        # --- 色設定UI：縦配置---
        row = QtWidgets.QHBoxLayout()

        # プレビュー背景（ボタン）
        self.btn_pick_bg = QtWidgets.QPushButton("プレビュー領域")
        self.btn_pick_bg.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_pick_bg.setMinimumHeight(26)
        self.btn_pick_bg.setStyleSheet("""
        QPushButton { background:#2b2b2b; color:#e0e0e0; border:1px solid #3a3a3a; border-radius:6px; padding:4px 10px; }
        QPushButton:hover  { background:#333333; }
        QPushButton:pressed{ background:#222222; }
        """)

        # 画像表示領域（ボタン）
        self.btn_pick_view = QtWidgets.QPushButton("画像表示領域")
        self.btn_pick_view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_pick_view.setMinimumHeight(26)
        self.btn_pick_view.setStyleSheet(self.btn_pick_bg.styleSheet())
    
        # 完全に“飾り”にする
        for b in (self.btn_pick_bg, self.btn_pick_view):
            b.setEnabled(False)  # クリック不可＆キーボードもフォーカスしない
            b.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)  # hover/pressも拾わない
            b.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            b.setToolTip("")  # 紛らわしいのでツールチップも消す

        # === 色チップ：左=現在色、+クイック2個（右クリック編集可） ===
        # 画像表示領域
        self.view_chip_cur = ColorChipButton(
            self._load_color("view_chip_cur", "#ffffff"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="画像表示領域の『左チップ』を編集",
            group="view",
        )
        self.view_chip_q1 = ColorChipButton(
            self._load_color("view_quick_1", "#000000"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="画像表示領域の『中央チップ』を編集",
            group="view",
        )
        self.view_chip_q2 = ColorChipButton(
            self._load_color("view_quick_2", "#15ff00"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="画像表示領域の『右チップ』を編集",
            group="view",
        )

        # プレビュー領域
        self.preview_chip_cur = ColorChipButton(
            self._load_color("preview_chip_cur", "#ffffff"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="プレビュー領域の『左チップ』を編集",
            group="preview",
        )
        self.preview_chip_q1 = ColorChipButton(
            self._load_color("preview_quick_1", "#000000"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="プレビュー領域の『中央チップ』を編集",
            group="preview",
        )
        self.preview_chip_q2 = ColorChipButton(
            self._load_color("preview_quick_2", "#15ff00"),
            tooltip="左クリック：適用　右クリック：カスタム",
            edit_title="プレビュー領域の『右チップ』を編集",
            group="preview",
        )

        # ★ チップ選択状態を設定から復元
        #    画像表示領域(view)は初回起動時の既定を「中央=1」にする
        #    プレビュー領域(preview)は従来どおり「左=0」
        view_chips = (self.view_chip_cur, self.view_chip_q1, self.view_chip_q2)
        preview_chips = (self.preview_chip_cur, self.preview_chip_q1, self.preview_chip_q2)

        # --- 画像表示領域(view) ---
        try:
            view_sel = int(self.settings.value("view_chip_selected", 1, type=int))
        except Exception:
            view_sel = 1

        # --- プレビュー領域(preview) ---
        try:
            preview_sel = int(self.settings.value("preview_chip_selected", 0, type=int))
        except Exception:
            preview_sel = 0

        # 範囲外なら既定へ戻す
        if not (0 <= view_sel < len(view_chips)):
            view_sel = 1
        if not (0 <= preview_sel < len(preview_chips)):
            preview_sel = 0

        # 選択状態を反映
        for i, w in enumerate(view_chips):
            w.set_selected(i == view_sel)
        for i, w in enumerate(preview_chips):
            w.set_selected(i == preview_sel)
            
        # 下段の行レイアウト（ボタン間の隙間は少し）
        chip_row_view = QtWidgets.QHBoxLayout()
        chip_row_view.setContentsMargins(0, 0, 0, 0)
        chip_row_view.setSpacing(6)
        for w in (self.view_chip_cur, self.view_chip_q1, self.view_chip_q2):
            chip_row_view.addWidget(w)

        chip_row_prev = QtWidgets.QHBoxLayout()
        chip_row_prev.setContentsMargins(0, 0, 0, 0)
        chip_row_prev.setSpacing(6)
        for w in (self.preview_chip_cur, self.preview_chip_q1, self.preview_chip_q2):
            chip_row_prev.addWidget(w)

        # クリック（左）→ 即適用 ＋ チップの選択状態更新
        for w in (self.view_chip_cur, self.view_chip_q1, self.view_chip_q2):
            w.colorClicked.connect(
                lambda c, chip=w: self._on_view_chip_clicked(chip, c)
            )

        for w in (self.preview_chip_cur, self.preview_chip_q1, self.preview_chip_q2):
            w.colorClicked.connect(
                lambda c, chip=w: self._on_preview_chip_clicked(chip, c)
            )

        # 右クリック編集：保存して即適用＋選択状態更新
        self.view_chip_cur.colorEdited.connect(
            lambda c, chip=self.view_chip_cur: (
                self._save_color("view_chip_cur", c),
                self._on_view_chip_clicked(chip, c)
            )
        )
        self.view_chip_q1.colorEdited.connect(
            lambda c, chip=self.view_chip_q1: (
                self._save_color("view_quick_1", c),
                self._on_view_chip_clicked(chip, c)
            )
        )
        self.view_chip_q2.colorEdited.connect(
            lambda c, chip=self.view_chip_q2: (
                self._save_color("view_quick_2", c),
                self._on_view_chip_clicked(chip, c)
            )
        )

        self.preview_chip_cur.colorEdited.connect(
            lambda c, chip=self.preview_chip_cur: (
                self._save_color("preview_chip_cur", c),
                self._on_preview_chip_clicked(chip, c)
            )
        )
        self.preview_chip_q1.colorEdited.connect(
            lambda c, chip=self.preview_chip_q1: (
                self._save_color("preview_quick_1", c),
                self._on_preview_chip_clicked(chip, c)
            )
        )
        self.preview_chip_q2.colorEdited.connect(
            lambda c, chip=self.preview_chip_q2: (
                self._save_color("preview_quick_2", c),
                self._on_preview_chip_clicked(chip, c)
            )
        )

        # --- 各ペアを縦（ボタンの下にチップ列）でまとめる ---
        view_col = QtWidgets.QVBoxLayout()
        view_col.setSpacing(4)
        view_col.addWidget(self.btn_pick_view, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        view_col.addLayout(chip_row_view)

        prev_col = QtWidgets.QVBoxLayout()
        prev_col.setSpacing(4)
        prev_col.addWidget(self.btn_pick_bg, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        prev_col.addLayout(chip_row_prev)

        # 左側：色チップ 2列（プレビュー/画像表示）
        row = QtWidgets.QHBoxLayout()
        row.addLayout(prev_col)
        row.addSpacing(12)
        row.addLayout(view_col)

        # 右側：保存方法（連番/上書き）ラジオ
        self.rad_seq_quick = QtWidgets.QRadioButton("連番保存")
        self.rad_ow_quick  = QtWidgets.QRadioButton("上書き保存")

        # 初期状態を設定の値に同期
        self.rad_seq_quick.setChecked(not bool(getattr(self, "overwrite_mode", False)))
        self.rad_ow_quick.setChecked(bool(getattr(self, "overwrite_mode", False)))

        # 変更 → 即時保存（True になった側だけ発火）
        self.rad_seq_quick.toggled.connect(
            lambda on: self._on_quick_save_mode_changed(False) if on else None
        )
        self.rad_ow_quick.toggled.connect(
            lambda on: self._on_quick_save_mode_changed(True)  if on else None
        )

        # グループ共通の見た目
        group_style = """
            QGroupBox {
                border: 1px solid #ffffff;
                border-radius: 6px;
                margin-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #ffffff;
            }
        """

        # --- 左のグループ：保存方法 ---
        grp_mode = QtWidgets.QGroupBox("保存方法")
        mode_lay = QtWidgets.QVBoxLayout(grp_mode)
        mode_lay.setContentsMargins(12, 18, 12, 8)
        mode_lay.setSpacing(4)
        mode_lay.addWidget(self.rad_seq_quick)
        mode_lay.addWidget(self.rad_ow_quick)
        mode_lay.addStretch(1)
        grp_mode.setStyleSheet(group_style)

        # --- 右のグループ：保存先 ---
        self.rad_dest_same_quick   = QtWidgets.QRadioButton("読込み元と同一")
        self.rad_dest_custom_quick = QtWidgets.QRadioButton("フォルダ指定")

        # ★ ini の保存先設定をラジオに反映
        dest_mode = getattr(self, "save_dest_mode", "same")
        if dest_mode == "custom":
            self.rad_dest_custom_quick.setChecked(True)
        else:
            self.rad_dest_same_quick.setChecked(True)

        self.btn_dest_browse_quick = QtWidgets.QToolButton()
        self.btn_dest_browse_quick.setText("参照")

        # マウスカーソルを「クリックできる」感じに
        self.btn_dest_browse_quick.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )

        # 見た目（背景グレー＋ホバー反応）
        self.btn_dest_browse_quick.setStyleSheet("""
            QToolButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #aaaaaa;
                border-radius: 3px;
                padding: 0px 1px;   /* ← 上下0, 左右1 にしてほぼ余白ナシ */
                font-size: 9pt;
                min-width: 0px;     /* 念のため最小幅もゼロにしておく */
            }
            QToolButton:hover:enabled {
                background-color: #777777;
            }
            QToolButton:pressed {
                background-color: #444444;
            }
            QToolButton:disabled {
                background-color: #333333;
                color: #777777;
                border-color: #555555;
            }
        """)
        self.btn_dest_browse_quick.setFixedHeight(20)

        self.btn_dest_browse_quick.clicked.connect(self._on_quick_browse_dest)

        grp_dest = QtWidgets.QGroupBox("保存先")
        gdest = QtWidgets.QGridLayout(grp_dest)
        gdest.setContentsMargins(12, 18, 12, 8)
        gdest.setSpacing(4)
        gdest.addWidget(self.rad_dest_same_quick,   0, 0, 1, 3)

        # 1行目：ラジオ＋参照ボタン
        gdest.addWidget(self.rad_dest_custom_quick, 1, 0, 1, 2)  # 2列ぶん使う
        gdest.addWidget(self.btn_dest_browse_quick, 1, 2, 1, 1)

        grp_dest.setStyleSheet(group_style)

        def _toggle_dest_quick():
            """クイックUIの保存先ラジオが変わったときの共通処理"""
            use_custom = self.rad_dest_custom_quick.isChecked()

            # 参照ボタン ON/OFF
            self.btn_dest_browse_quick.setEnabled(use_custom)

            # 内部モードを更新
            self.save_dest_mode = "custom" if use_custom else "same"
            try:
                self.settings.setValue("save_dest_mode", self.save_dest_mode)
            except Exception:
                pass

            if use_custom:
                # フォルダ指定：既存のカスタム保存先を優先して適用
                d = (
                    getattr(self, "save_custom_dir", "")
                    or getattr(self, "save_folder", "")
                    or os.path.dirname(getattr(self, "image_path", "") or "")
                )
                if d:
                    self.save_custom_dir = d
                    try:
                        self.settings.setValue("save_custom_dir", d)
                    except Exception:
                        pass
                    # 実際の保存先として固定
                    self.save_folder = d
            else:
                # 読み込み元と同一：固定保存先を解除
                # → 保存時は毎回 image_path のフォルダが使われる
                self.save_folder = ""

            # 右下「保存先：」表示を実効値で更新
            try:
                self._update_save_folder_label()
            except Exception:
                pass
            
        # ラジオボタンの切り替えと内部状態を連動
        self.rad_dest_same_quick.toggled.connect(_toggle_dest_quick)
        self.rad_dest_custom_quick.toggled.connect(_toggle_dest_quick)

        # 起動時の状態にあわせて一度だけ反映
        _toggle_dest_quick()

        
        # 左の色チップ列を一旦ウィジェット化
        left = QtWidgets.QWidget()
        left.setLayout(row)

        # 「色チップ列」＋「保存方法」＋「保存先」を横並びで包む
        row_widget = QtWidgets.QWidget()
        wrap = QtWidgets.QHBoxLayout(row_widget)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setSpacing(16)
        wrap.addWidget(left)
        wrap.addStretch(1)
        wrap.addWidget(grp_mode,  0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        wrap.addWidget(grp_dest,  0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # 横に広がらず、中身ぶんだけの幅で左寄せ表示にする
        row_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.color_row_widget = row_widget

        # ---- レイアウトに追加 ----

        # 上側の余白
        self.sub_panel_layout.addStretch(2)

        # 解像度ラベル（灰色エリアの中央に来る）
        self.sub_panel_layout.addWidget(
            self.crop_size_label,
            alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
        )

        # ラベルとボタンの間の余白
        self.sub_panel_layout.addStretch(1)

        flip_row = QtWidgets.QHBoxLayout()
        flip_row.setContentsMargins(0, 0, 0, 0)
        flip_row.setSpacing(8)  # ボタン同士のすき間

        flip_row = QtWidgets.QHBoxLayout()
        flip_row.setContentsMargins(0, 0, 0, 0)
        flip_row.setSpacing(8)


        # ★ 一括切り取りボタン
        self.btn_batch_crop = QtWidgets.QPushButton("一括\n切り取り")
        self.btn_batch_crop.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_batch_crop.setFixedSize(72, 72)
        self.btn_batch_crop.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_batch_crop.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )

        # ★ 一括切り取りだけ紫枠にする
        self.btn_batch_crop.setStyleSheet("""
        QPushButton {
            background:#2b2b2b;
            color:#e0e0e0;
            border:2px solid #b36bff;      /* ← 紫の外枠 */
            border-radius:10px;
            padding:0px;
            font-size: 11pt;
        }
        QPushButton:hover  {
            background:#454545;
            border-color:#d7adff;
        }
        QPushButton:pressed{
            background:#222222;
            border-color:#8f3dff;
        }
        """)

        self.btn_batch_crop.clicked.connect(self.on_batch_crop_clicked)

        # 左：水平反転ボタン
        flip_row.addWidget(
            self.btn_flip_horizontal,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        # 中：垂直反転ボタン
        flip_row.addWidget(
            self.btn_flip_vertical,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        # 右：左右分割の 90度回転ボタン
        flip_row.addWidget(
            self.btn_rotate_90,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        # ★ 一括切り取りボタン
        flip_row.addWidget(
            self.btn_batch_crop,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        flip_row.addStretch(1)
        self.sub_panel_layout.addLayout(flip_row)

        # 念のため：サブパネルは縦に広がれるように
        self.sub_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        main_widget = QtWidgets.QWidget()
        # 中央域の余白も暗色で塗る（背景色のみ）
        main_widget.setAutoFillBackground(True)
        main_widget.setStyleSheet("background: #121212;")
        self.elide_gap_px = 150
        self.nudge_gap_px = 3   # 微調整ダイアログとパネルの間隔

        # --- ドラッグサイズを N の倍数に丸める（枠内優先）
        self.multiple_lock_enabled = False
        self.multiple_w = 64   # 幅用の倍数（先に 64 固定でOK。後で 32 などにも拡張可）
        self.multiple_h = 64   # 高さ用の倍数（上に同じ）

        # --- 外側縦レイアウト ---
        outer_layout = QtWidgets.QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(0)

        # --- 横並びメインレイアウト（左・中央・右カラム） ---
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(3)

        # --- 左カラム（縦レイアウト） ---
        self.preview_area = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(self.preview_area)
        preview_layout.setContentsMargins(14, 20, 14, 20)
        preview_layout.addWidget(
            self.preview_label,
            alignment=QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        )
        
        # ←← ここに「一時上限」を適用（起動直後だけ広がり過ぎ防止）
        self._preview_initial_cap = True
        self.preview_area.setMaximumWidth(384 + 14*2)  # 384（基本512にしたいが起動直後だけ抑える）
        self.preview_area.setMinimumWidth(self.preview_label.minimumWidth() + 14*2)

        #表示後のサイズ変更までキャップを残すためのフラグ
        self._window_shown = False
        self._first_shown_size = None

        #起動直後だけラベル自体も 384 上限にしておく
        self.preview_label.setMaximumSize(384, 384)

        if hasattr(self, "sub_panel"):
            self.sub_panel.setMaximumWidth(384 + 14*2)
        if hasattr(self, "color_row_widget"):
            self.color_row_widget.setMaximumWidth(384 + 14*2)

        self.preview_area.setStyleSheet("border: 2px solid #4b94d9; border-radius: 14px; background: #222;")
        self.preview_label.setStyleSheet("background: transparent; border: none; border-radius: 0px;")

        preview_column = QtWidgets.QVBoxLayout()
        preview_column.setContentsMargins(0, 0, 0, 0)
        preview_column.setSpacing(8)

        preview_column.addWidget(self.preview_area)

        if hasattr(self, "color_row_widget"):
            preview_column.addWidget(
                self.color_row_widget, 0,
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
            )
        
        preview_column.addWidget(self.sub_panel, 1)

        main_layout.addLayout(preview_column, stretch=0)

        self._qimage_src_pil = None       # メイン表示用のPillow画像を保持
        self._qimage_main = None          # 変換後QImage（必要なら保持）
        self._preview_src_pil = None      # プレビュー用Pillow画像を保持
        self._preview_qimage = None       # 変換後QImage（必要なら保持）

        self._preview_src_key = None

        # --- 中央カラム用パネル ---
        self.central_panel = QtWidgets.QWidget()
        self.central_panel.setStyleSheet("background: #121212;")
        self.central_panel.setObjectName("mainCanvas") 
        self.central_layout = QtWidgets.QVBoxLayout(self.central_panel)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(8)

        # --- 横並びパネル（戻るボタン、情報ラベル、進むボタン） ---
        self.info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout(self.info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        self.btn_prev = QtWidgets.QPushButton("◁")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # 中央の情報表示用バナー（OSアイコン＋テキスト）
        self.info_banner = InfoBanner(self)

        self.btn_next = QtWidgets.QPushButton("▷")
        self.btn_next.setFixedWidth(30)
        self.btn_next.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        arrow_btn_style = (
            "QPushButton {"
            "  color: rgba(224, 224, 224, 255);"
            "  background-color: rgba(63, 63, 63, 255);"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(110, 110, 110, 255);"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(46, 46, 46, 255);"
            "}"
            "QPushButton:disabled {"
            "  color: rgba(102, 102, 102, 255);"
            "  background-color: rgba(43, 43, 43, 255);"
            "}"
        )
        self.btn_prev.setStyleSheet(arrow_btn_style)
        self.btn_next.setStyleSheet(arrow_btn_style)

        # ★ ホバー時に出すメッセージ（ツールチップ）
        self.btn_prev.setToolTip("左クリック：前の画像へ\n右クリック：前のフォルダへ")
        self.btn_next.setToolTip("左クリック：次の画像へ\n右クリック：次のフォルダへ")

        info_layout.addWidget(self.btn_prev)
        info_layout.addWidget(self.info_banner, stretch=1)
        info_layout.addWidget(self.btn_next)

        # --- 画像表示用 CropLabel ---
        self.label = CropLabel(self)
        # --- プレースホルダ用フラグ初期化 & ダブルクリック監視 ---
        self._placeholder_active = False
        self._placeholder_path = ""
        self._placeholder_selected = False      # ← クリック選択の反転状態
        self._placeholder_hit_rect = None       # ← クリック当たり判定
        self._panel_hidden_by_placeholder = False
        self.label.installEventFilter(self)     # ← ここで main window の eventFilter に渡す
        self._apply_view_bg()
        self.label.setMinimumSize(1, 1)
        self.central_layout.addWidget(self.label)
        self.central_layout.addWidget(self.info_panel)
        
        main_layout.addWidget(self.central_panel, stretch=1)

        # --- 右カラム：サムネイルリスト（枠付きエリア） ---
        self.listview = CustomListView()
        self.listview.setMaximumWidth(200)
        self.listview.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.listview.setIconSize(QtCore.QSize(160, 240))
        self.listview.setGridSize(QtCore.QSize(180, 260))
        self.listview.setSpacing(16)
        self.listview.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.listview.setMovement(QtWidgets.QListView.Movement.Static)

        # ★ ファイル名は中央省略にして拡張子が見えるようにする
        self.listview.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)

        # ピクセル単位のスクロールにする
        self.listview.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        # ホイール1ノッチあたりの移動量（px）を調整
        sb = self.listview.verticalScrollBar()
        sb.setSingleStep(24)   # ←好みで 12〜48 あたり
        # PageUp/Down の移動量（“1ページ”）も調整したいなら
        sb.setPageStep(self.listview.gridSize().height()) 
    
        # 枠コンテナを作って中に listview を入れる（★このブロックを丸ごと置き換え）
        self.thumb_area = QtWidgets.QWidget(self)
        self.thumb_area.setObjectName("thumbArea")
        thumb_layout = QtWidgets.QVBoxLayout(self.thumb_area)
        thumb_layout.setContentsMargins(14, 20, 14, 20)  # プレビューと同じ雰囲気
        thumb_layout.setSpacing(0)

        # === ここから追加: サムネ用ナビバー（戻る／進む／上へ） ===
        # 履歴状態
        self._nav_history: list[str] = []
        self._nav_pos: int = -1
        # === nav debug / epoch ===
        self._nav_epoch = 0        # open_folder を呼ぶたびに ++
        self._nav_debug = True     # True でログを出す（不要になったら False）
        # フォルダごとの“最後にフォーカスしてた子（フォルダ/画像）”
        self._last_focus_by_dir: dict[str, str] = {}

        # 最後に open_folder を呼んだ時刻（currentChanged デバウンス用）
        self._nav_epoch_at = 0.0
        # ループ状態は設定から復元済みならその値を使う（未設定なら False）
        self._thumb_loop_enabled = bool(getattr(self, "_thumb_loop_enabled", False))

        # ナビバー本体
        self.thumb_nav_bar = QtWidgets.QWidget(self.thumb_area)
        _nav_l = QtWidgets.QHBoxLayout(self.thumb_nav_bar)
        _nav_l.setContentsMargins(0, 0, 6, 6)   # 下に少し余白
        _nav_l.setSpacing(6)

        st = self.style()
        self.btn_nav_back = QtWidgets.QToolButton(self.thumb_nav_bar)
        self.btn_nav_back.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack))
        self.btn_nav_back.setToolTip("戻る")
        self.btn_nav_back.setAutoRaise(True)

        self.btn_nav_fwd = QtWidgets.QToolButton(self.thumb_nav_bar)
        self.btn_nav_fwd.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward))
        self.btn_nav_fwd.setToolTip("進む")
        self.btn_nav_fwd.setAutoRaise(True)

        self.btn_nav_up = QtWidgets.QToolButton(self.thumb_nav_bar)
        self.btn_nav_up.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp))
        self.btn_nav_up.setToolTip("上へ")
        self.btn_nav_up.setAutoRaise(True)

        self.btn_nav_reload = QtWidgets.QToolButton(self.thumb_nav_bar)
        self.btn_nav_reload.setIcon(self._create_thumb_nav_icon("reload"))
        self.btn_nav_reload.setToolTip("再読み込み")
        self.btn_nav_reload.setAutoRaise(True)

        self.btn_nav_loop = QtWidgets.QToolButton(self.thumb_nav_bar)
        self.btn_nav_loop.setIcon(self._create_thumb_nav_icon("loop"))
        self.btn_nav_loop.setToolTip("リピート表示（末尾の次で先頭に戻る）")
        self.btn_nav_loop.setAutoRaise(True)
        self.btn_nav_loop.setCheckable(True)
        self.btn_nav_loop.setChecked(self._thumb_loop_enabled)

        _nav_l.addWidget(self.btn_nav_back)
        _nav_l.addWidget(self.btn_nav_fwd)
        _nav_l.addWidget(self.btn_nav_up)
        _nav_l.addWidget(self.btn_nav_reload)
        _nav_l.addWidget(self.btn_nav_loop)
        _nav_l.addStretch(1)

        # --- ボタンを大きくする調整 ---
        ICON_PX = 22          # ← アイコンの実サイズ（28〜34くらいお好みで）
        BTN_SIDE = ICON_PX + 8  # ← ボタン自体の1辺。余白ぶん少し大きめ

        for b in (self.btn_nav_back, self.btn_nav_fwd, self.btn_nav_up, self.btn_nav_reload, self.btn_nav_loop):
            b.setIconSize(QtCore.QSize(ICON_PX, ICON_PX))
            b.setFixedSize(BTN_SIDE, BTN_SIDE)               # クリック範囲も拡大
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

        # ★ 再読み込みボタン用：クリック中だけ一段明るく
        self.btn_nav_reload.setObjectName("thumbReloadButton")

        self.btn_nav_reload.setStyleSheet(
            """
            #thumbReloadButton:pressed {
                background-color: rgba(255, 255, 255, 70);   /* クリック中はナビバーより明るく */
            }
            """
        )

        # 再読み込み／リピートだけ少し小さめにしたい場合
        RL_ICON_PX = 19
        for b in (self.btn_nav_reload, self.btn_nav_loop):
            b.setIconSize(QtCore.QSize(RL_ICON_PX, RL_ICON_PX))
            b.setFixedSize(BTN_SIDE, BTN_SIDE)  # クリック範囲は同じでOKならそのまま
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

        # ナビバーの余白とボタン間の間隔も少し広げる
        _nav_l.setContentsMargins(6, 6, 6, 6)
        _nav_l.setSpacing(6)

        # クリック動作
        self.btn_nav_back.clicked.connect(lambda: self._nav_go(-1))
        self.btn_nav_fwd.clicked.connect(lambda: self._nav_go(+1))
        self.btn_nav_up.clicked.connect(self._nav_up)
        self.btn_nav_reload.clicked.connect(self._nav_reload)
        self.btn_nav_loop.toggled.connect(self._on_thumb_loop_toggled)

        # 親に名前を付けて、その親をQSSターゲットに
        self.thumb_nav_bar.setObjectName("thumbNav")

        # ボタン側は枠なしのフラット、フォーカス枠も出さない（念のため）
        for b in (self.btn_nav_back, self.btn_nav_fwd, self.btn_nav_up, self.btn_nav_reload, self.btn_nav_loop):
            b.setAutoRaise(True)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

                # QSS: 外枠1本だけ / ボタンは枠なし・ホバーでうっすら
        self.thumb_nav_bar.setStyleSheet(f"""
        #thumbNav {{
            background: rgba(18,18,18,160);
            border: 1px solid #3b3b3b;
            border-radius: 12px;
        }}
        #thumbNav QToolButton {{
            border: none;
            background: transparent;
            padding: 0px;
            margin: 0px;
            border-radius: {BTN_SIDE//2 - 3}px;
        }}
        #thumbNav QToolButton:hover {{ background: rgba(255,255,255,0.40); }}
        #thumbNav QToolButton:pressed {{ background: rgba(255,255,255,0.16); }}
        #thumbNav QToolButton:disabled {{ color: rgba(255,255,255,0.35); }}
        #thumbNav QToolButton:checked {{ background: rgba(70, 150, 240, 0.45); }}
        """)

        # 軽いドロップシャドウで浮かせる
        try:
            eff = QtWidgets.QGraphicsDropShadowEffect(self.thumb_nav_bar)
            eff.setBlurRadius(24)
            eff.setOffset(0, 4)
            eff.setColor(QtGui.QColor(0,0,0,140))
            self.thumb_nav_bar.setGraphicsEffect(eff)
        except Exception:
            pass

        # ダブルクリックでフォルダ／画像を開く
        self.listview.doubleClicked.connect(self.on_thumb_double_clicked)

        # レイアウトに積む順番：ナビバー → リストビュー
        thumb_layout.addWidget(self.thumb_nav_bar)
        thumb_layout.addWidget(self.listview)

        # 初期のボタン活性／非活性
        self._update_nav_buttons()

        # 黄緑系の外枠（色はお好みで微調整OK）
        self.thumb_area.setStyleSheet("""
            QWidget {
                border: 2px solid rgb(100, 217, 138);  /* 黄緑 */
                border-radius: 14px;
                background: #222;
            }
        """)

        self.thumb_area.setMinimumWidth(180)

        main_layout.addWidget(self.thumb_area, stretch=0)

        outer_layout.addLayout(main_layout, stretch=1)

        self.setCentralWidget(main_widget)
 
        # --- ◁▷ ボタンのクリックイベント接続（左クリックは“サムネ行”移動に統一） ---
        # 旧接続を外す（存在しない環境でも例外は握りつぶす）
        try: self.btn_prev.clicked.disconnect()
        except Exception: pass
        try: self.btn_next.clicked.disconnect()
        except Exception: pass

        # 左クリック → 必ず“移動直前に”復元方針をセットしてから移動
        self.btn_prev.clicked.connect(self._on_nav_prev_clicked)
        self.btn_next.clicked.connect(self._on_nav_next_clicked)

        # 右クリック（コンテキストメニュー）は従来通り：隣接フォルダへジャンプ
        self.btn_prev.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_prev.customContextMenuRequested.connect(
            lambda _pos: self._jump_sibling_folder(-1, require_images=False)
        )
        self.btn_next.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_next.customContextMenuRequested.connect(
            lambda _pos: self._jump_sibling_folder(+1, require_images=False)
        )
       
        self._custom_edit_in_toggle = False

        menubar = self.menuBar()
        file_menu = menubar.addMenu("ファイル")
        open_action = file_menu.addAction("画像を開く")
        open_action.triggered.connect(self.open_image)
        file_menu.addSeparator()
        save_folder_select = file_menu.addAction("保存先フォルダを設定")
        save_folder_select.triggered.connect(self.set_save_folder)

        # 固定切り出しメニュー
        crop_menu = menubar.addMenu("固定切り出し")
        self.crop_action_group = QActionGroup(self)
        self.crop_action_group.setExclusive(True)
        self.crop_actions = {}
        fixed_sizes = [
            ("1024 × 1024", (1024, 1024)),
            ("1216 × 832", (1216, 832)),
            ("832 × 1216", (832, 1216)),
            ("1536 × 1536", (1536, 1536)),
            ("1536 × 1024", (1536, 1024)),
            ("1024 × 1536", (1024, 1536)),
        ]
        for label, size in fixed_sizes:
            act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda checked, s=size: self.fixed_crop_triggered(s))
            crop_menu.addAction(act)
            self.crop_action_group.addAction(act)
            self.crop_actions[size] = act

        # カスタムサイズ：ON/OFFトグル（チェック付き／ダイアログ出さない）
        self.custom_toggle_action = QAction("カスタムサイズを使う", self, checkable=True)
        self.custom_toggle_action.setChecked(False)
        self.custom_toggle_action.toggled.connect(self.on_custom_toggle)

        # カスタムサイズ：編集ダイアログ（チェックなし／サイズ入力だけ）
        self.custom_edit_action = QAction("カスタムサイズを編集...", self)
        self.custom_edit_action.triggered.connect(self.on_custom_edit)

        # 互換性のため、古い変数名も指しておく（他所で self.custom_action を参照していても壊れない）
        self.custom_action = self.custom_toggle_action

        # メニューに追加（元の custom_action 1個の代わりに2個）
        # menu.addAction(self.custom_action) ← これは削除して、
        crop_menu.addAction(self.custom_toggle_action)
        crop_menu.addAction(self.custom_edit_action)

        # 現在のカスタムサイズを保持（未設定なら None）
        self.custom_size = getattr(self, "custom_size", None)  # 既にあれば引き継ぐ
        self.update_custom_edit_action_text()

        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setFocus() 
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.selectionMade.connect(self.on_crop)
        self.label.fixedSelectionMade.connect(self.on_fixed_crop_move)
        self.label.movedRect.connect(self.on_crop_rect_moved)

        # 制約メニュー（シンプル：32 / 64 だけ）
        constraint_menu = menubar.addMenu("スナップ単位")

        self.act_multiple_32 = QAction("32 px", self, checkable=True)
        self.act_multiple_64 = QAction("64 px", self, checkable=True)

        constraint_menu.addAction(self.act_multiple_32)
        constraint_menu.addAction(self.act_multiple_64)
    
        # 設定
        act_settings = menubar.addAction("設定") 
        act_settings.triggered.connect(self.open_options_dialog)

        # ▼ メニューバーを常に読みやすい色に（ダーク背景でも消えない）
        menubar.setStyleSheet("""
        QMenuBar {
            background: #121212;
            color: #e0e0e0;
        }
        QMenuBar::item {
            background: transparent;
            color: #e0e0e0;
            padding: 3px 8px;
        }
        QMenuBar::item:selected {
            background: #2a2a2a;
            color: #ffffff;
        }
        QMenuBar::item:pressed {
            background: #333333;
        }

        /* ドロップダウン側（必要なければこのブロックは削除可） */
        QMenu {
            background: #1b1b1b;
            color: #e0e0e0;
            border: 1px solid #2a2a2a;
        }
        QMenu::item:selected {
            background: #2a2a2a;
            color: #ffffff;
        }
        QMenu::separator {
            height: 1px;
            background: #3a3a3a;
            margin: 4px 6px;
        }
        """)

        self._apply_thumb_scroll_step()
        # どこか __init__ の終わりあたり
        self._opening_folder = False               # 再入防止フラグ
        self._open_folder_watch = QtCore.QTimer(self)
        self._open_folder_watch.setSingleShot(True)
        self._open_folder_watch.timeout.connect(lambda: log_debug("[open_folder][watchdog] still running..."))

        QtCore.QTimer.singleShot(0, lambda: _enable_dark_titlebar(self))

        # 既存状態をメニューに反映（起動時）
        def _sync_multiple_actions():
            if getattr(self, "multiple_lock_enabled", False):
                if getattr(self, "multiple_w", 64) == 32:
                    self.act_multiple_32.setChecked(True)
                    self.act_multiple_64.setChecked(False)
                else:
                    self.act_multiple_32.setChecked(False)
                    self.act_multiple_64.setChecked(True)
            else:
                # 丸めOFF（どちらも未チェック）
                self.act_multiple_32.setChecked(False)
                self.act_multiple_64.setChecked(False)

        _sync_multiple_actions()

        def _apply_current_multiple_to_dragrect():
            if self.multiple_lock_enabled and getattr(self.label, "drag_rect_img", None) is not None:
                r = self.label.drag_rect_img
                self.label.drag_rect_img = self.label._apply_multiple_and_keep_inside(
                    r.left(), r.top(), r.right(), r.bottom(),
                    self.multiple_w, self.multiple_h
                )
                self.label.update()

        def _reset_edge_lock_state():
            self.label._edge_lock_active = False
            self.label._edge_lock = {"left": False, "right": False, "top": False, "bottom": False}
            self.label._edge_locked_w = None
            self.label._edge_locked_h = None

        def _pick_multiple(val: int, checked: bool):
            if checked:
                # 選択 → 丸めON（32/64を設定）
                self.multiple_w = self.multiple_h = val
                self.multiple_lock_enabled = True

                # 丸めONにしたら、調整モードは強制OFF & 微調整オーバーレイも閉じる
                if hasattr(self, "set_adjust_mode"):
                    self.set_adjust_mode(False)

                # 旧ダイアログ閉じる系は全廃、オーバーレイを閉じる
                if hasattr(self, "close_nudge_overlay"):
                    self.close_nudge_overlay()
                else:
                    # 念のためのフォールバック掃除
                    w = getattr(self, "_nudge_overlay", None)
                    if w is not None:
                        try:
                            w.close()
                        except Exception:
                            pass
                        finally:
                            setattr(self, "_nudge_overlay", None)

                # 既存パネルが出ていれば「調整」ボタンを隠す
                panel = getattr(self, "_action_panel", None)
                if panel:
                    try:
                        panel.enable_adjust(False)
                    except Exception:
                        pass

                _reset_edge_lock_state()

                # もう片方を外す（シグナルブロックで無限ループ回避）
                if val == 32:
                    with QSignalBlocker(self.act_multiple_64):
                        self.act_multiple_64.setChecked(False)
                else:
                    with QSignalBlocker(self.act_multiple_32):
                        self.act_multiple_32.setChecked(False)

                _apply_current_multiple_to_dragrect()

            else:
                # OFFにするのは「両方未チェック」のときだけ
                if not self.act_multiple_32.isChecked() and not self.act_multiple_64.isChecked():
                    self.multiple_lock_enabled = False
                    _reset_edge_lock_state()

                    # ★ 丸めOFFなら、既存パネルの「調整」ボタンを再表示
                    panel = getattr(self, "_action_panel", None)
                    if panel:
                        try:
                            panel.enable_adjust(True)
                        except Exception:
                            pass

            self.label.update()

        # 片方がチェック済みなら何もしない（切替中

        self.act_multiple_32.toggled.connect(lambda checked: _pick_multiple(32, checked))
        self.act_multiple_64.toggled.connect(lambda checked: _pick_multiple(64, checked))

        # --- ズーム倍率表示ラベルをCropLabel上に追加 ---
        self.zoom_label = ZoomLabel(self.label)
        # 念のため遅延版を外す（無ければ except に落ちるだけ）
        try:
            self.listview.clicked.disconnect(self._on_thumb_clicked_delayed)
        except Exception:
            pass
        try:
            self.listview.clicked.disconnect(self.on_thumbnail_clicked)
        except Exception:
            pass
        self.listview.clicked.connect(self.on_thumbnail_clicked)

        # ダブルクリックは従来どおり（フォルダを開くなど）
        try:
            self.listview.doubleClicked.disconnect(self.on_thumb_double_clicked)
        except Exception:
            pass
        self.listview.doubleClicked.connect(self.on_thumb_double_clicked)

        # 遅延タイマー関連はもう使わない
        self._pending_index = None
        try:
            self._click_timer.stop()
        except Exception:
            pass
        self._click_timer = None   # 参照が残っても hasattr チェックで安全に

        self.listview.setStyleSheet("""
        QListView {
            border: none;                 /* 親の枠継承を打ち消す */
            background: transparent;      /* 親の背景も透過に */
            color: #e0e0e0;               /* 非選択時の文字色を明示 */
        }
        QListView::item {
            color: #e0e0e0;               /* 念のためアイテムにも指定 */
        }
        QListView::item:selected {
            border: 2px solid #ff6600;
            background: #223344;
            color: #ffffff;               /* 選択時は白でクッキリ */
        }
        """)

        # さらに（スタイルシートが効かない環境も拾うため）パレットでも固定
        pal = self.listview.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e0e0e0"))        # 非選択
        pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))  # 選択
        self.listview.setPalette(pal)

        sb = self.listview.verticalScrollBar()

        def _make_arrow_png(path: str, direction: str, size: int = 10):
            if os.path.exists(path):
                return
            pm = QtGui.QPixmap(size, size)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor("#111111"))  # 矢印色（黒系）
            if direction == "up":
                poly = QtGui.QPolygon([
                    QtCore.QPoint(1, size-2), QtCore.QPoint(size//2, 2), QtCore.QPoint(size-2, size-2)
                ])
            else:  # "down"
                poly = QtGui.QPolygon([
                    QtCore.QPoint(1, 2), QtCore.QPoint(size//2, size-2), QtCore.QPoint(size-2, 2)
                ])
            p.drawPolygon(poly)
            p.end()
            pm.save(path, "PNG")

        tmp_dir = os.path.join(tempfile.gettempdir(), "gkiritori_scroll")
        os.makedirs(tmp_dir, exist_ok=True)
        up_png   = os.path.join(tmp_dir, "up.png").replace("\\", "/")
        down_png = os.path.join(tmp_dir, "down.png").replace("\\", "/")
        _make_arrow_png(up_png, "up")
        _make_arrow_png(down_png, "down")

        # （Fusionスタイルは任意）矢印を画像で明示して描かせる
        # sb.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

        sb.setStyleSheet(f"""
        QScrollBar {{ border: none; }}

        QScrollBar:vertical {{
            background: transparent;
            width: 16px;
            margin: 16px 0 16px 0;          /* 上下の▲▼分の余白 */
        }}

        QScrollBar::handle:vertical {{
            background: #5a5a5a;
            min-height: 24px;
            border-radius: 3px;
        }}

        /* ▲▼ボタンの台座（明るい灰色） */
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {{
            background: #cfcfcf;
            border: 1px solid #666;
            height: 16px;
            subcontrol-origin: margin;
        }}
        QScrollBar::sub-line:vertical {{ subcontrol-position: top; }}
        QScrollBar::add-line:vertical {{ subcontrol-position: bottom; }}

        /* ▲▼の絵を明示する（ここがないと描かれない） */
        QScrollBar::up-arrow:vertical   {{ image: url("{up_png}");   width: 10px; height: 10px; }}
        QScrollBar::down-arrow:vertical {{ image: url("{down_png}"); width: 10px; height: 10px; }}

        /* 溝 */
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: #2a2a2a;
        }}
        """)
    
        self.image = None
        self.image_path = None
        self.img_qt = None
        self.img_pixmap = None

        # --- パン/ズーム高速化用キャッシュ ---
        self._base_pixmap_dirty = True      # 画像が切り替わったら True にして作り直し
        self._scaled_pixmap = None          # 現在のズーム倍率で拡大済みのQPixmap
        self._scaled_key = None             # (base_key, base_w, base_h, zoom)

        # --- Repaint throttling ---
        self._repaint_hz = 60  # 30/60/120 など
        self._repaint_timer = QtCore.QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.timeout.connect(self._repaint_now)

        # --- Preview throttling ---
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setInterval(16)  # 約30fps（16にすれば約60fps）
        self._preview_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._preview_timer.timeout.connect(self._preview_now)
        self._pending_preview_rect = None  # 直近のプレビュー矩形
        
        # --- 無誤差プレビュー用のベースと倍率 ---
        self._preview_base_pixmap = None
        self._preview_sx = 1.0
        self._preview_sy = 1.0
        self._preview_src_size = None

        self.image_list = []
        self.current_index = -1
        self.model = None

        self.zoom_scale = 1.0   # --- 追加: ズーム倍率 ---
        self.shortcut_prev = QShortcut(QKeySequence("Ctrl+Left"), self)
        self.shortcut_next = QShortcut(QKeySequence("Ctrl+Right"), self)
      
        self.shortcut_prev.activated.connect(self.show_prev_image)
        self.shortcut_next.activated.connect(self.show_next_image)

        self._action_panel = None
        self._crop_rect = None
        self._success_label = None
        self._fixed_crop_rect = None

        self._adjust_mode = False   # 調整モードのON/OFFフラグ

        # ==== 下部バー（1段・三分割・独自実装） ====
        self.bottom_bar = QtWidgets.QWidget(self)
        self.bottom_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,  # 横は広がる
            QtWidgets.QSizePolicy.Policy.Fixed       # 縦は固定
        )
        self.bottom_layout = QtWidgets.QHBoxLayout(self.bottom_bar)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(8)

        # 左：アイコン + フルパス（可変＋省略）
        self.path_panel = QtWidgets.QWidget(self.bottom_bar)
        path_layout = QtWidgets.QHBoxLayout(self.path_panel)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(4)

        self.path_icon_label = QtWidgets.QLabel(self.path_panel)
        self.path_icon_label.setFixedSize(18, 18)
        self.path_icon_label.setScaledContents(True)
        self.path_icon_label.hide()  # 最初はアイコンなしなので隠しておく

        self.path_label = QtWidgets.QLabel(self.path_panel)
        self.path_label.setWordWrap(False)

        # クリックできる感＋色を「保存先」と揃える
        self.path_label.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.path_label.setStyleSheet("""
            QLabel {
                color: #8ab4f8;              /* 保存先ラベルのリンク色と同じ */
                text-decoration: underline;
            }
            QLabel:hover {
                color: #b3d1ff;              /* ホバー時の色も保存先と同じ */
            }
        """)

        self.path_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

        self._path_full_text = ""
        self.path_label.installEventFilter(self)

        path_layout.addWidget(self.path_icon_label)
        path_layout.addWidget(self.path_label)

        # 中：進捗（固定・隙間一定）
        self.progress_widget = QtWidgets.QWidget(self.bottom_bar)
        pl = QtWidgets.QHBoxLayout(self.progress_widget)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)  # 隙間は固定pxで作る

        # 進捗バー（固定幅・右側に据え置き）
        self.progress = ClickableProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress.clickedValueChanged.connect(self.on_progress_jump)
        self.progress.setMinimum(1)

        self.progress.setStyleSheet("QProgressBar { color: #e0e0e0; }")

        # ラベル（5/7）は右寄せ。増えた分は左へだけ伸びる
        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                       QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                        QtWidgets.QSizePolicy.Policy.Preferred)
        
        self.status_label.setStyleSheet("color: #e0e0e0;")

        # ★追加：ラベル幅を「最大想定文字列」で固定して、プログレス位置のブレを防ぐ
        fm = self.status_label.fontMetrics()

        sample_text = "99999 / 99999"
        fixed_w = fm.horizontalAdvance(sample_text)
        self.status_label.setMinimumWidth(fixed_w)
        self.status_label.setMaximumWidth(fixed_w)

        # ラベル用の箱（横に広がれる入れ物）
        label_box = QtWidgets.QWidget(self.progress_widget)
        lb = QtWidgets.QHBoxLayout(label_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(0)
        lb.addWidget(self.status_label)
        label_box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Preferred)

        # ラベルとバーの固定隙間（px）
        self.fixed_gap_px = getattr(self, "fixed_gap_px", 8)  # 好みで 4〜12 など

        # 並び： [label_box(左にだけ広がる)] [固定gap] [progress(固定幅)]
        pl.addWidget(label_box, 1)
        pl.addSpacing(self.fixed_gap_px)
        pl.addWidget(self.progress, 0)

        # 進捗コンテナは高さだけ固定（幅は中身に合わせる）
        self.progress_widget.setFixedHeight(24)
        self.progress_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Fixed)

        # 右：保存先（可変＋省略）
        self.save_folder_label = QtWidgets.QLabel("")
        self.save_folder_label.setWordWrap(False)

        # ★ リンクだけをクリック判定にする設定
        self.save_folder_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.save_folder_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse |
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.save_folder_label.setOpenExternalLinks(False)
        self.save_folder_label.linkActivated.connect(self._open_save_folder_link)

        # プレフィックス「保存先：」の色だけを QSS で指定
        self.save_folder_label.setStyleSheet("QLabel { color:#e0e0e0; }")

        # ★ リンク用の色と状態フラグ
        self._save_link_color_normal = "#8ab4f8"   # 通常
        self._save_link_color_hover  = "#b3d1ff"   # ホバー
        self._save_link_hovered = False

        self.save_folder_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred
        )

        self._save_full_text = ""

        self.save_folder_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        # ★ ホバーイベントを取る
        self.save_folder_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.save_folder_label.installEventFilter(self)


        # 並び：左（可変）- 中（固定） - 右（可変）
        self.bottom_layout.addWidget(self.path_panel,        1)
        self.bottom_layout.addWidget(self.progress_widget,   0)
        self.bottom_layout.addWidget(self.save_folder_label, 1)

        # 初期ストレッチ（あとで動的に上書き）
        self.bottom_layout.setStretchFactor(self.path_panel,        1)
        self.bottom_layout.setStretchFactor(self.progress_widget,   0)
        self.bottom_layout.setStretchFactor(self.save_folder_label, 1)

        # ★ 起動直後、settings.ini の保存先設定を実際の save_folder に反映する
        try:
            mode = getattr(self, "save_dest_mode", "same")
            if mode == "custom":
                # 前回のカスタム保存先（フォルダ指定）があれば、そのまま適用
                custom = getattr(self, "save_custom_dir", "") or ""
                if custom:
                    self._apply_save_folder_programmatically(custom)
            else:
                # 「読み込み元と同一」モードのときは save_folder を空にしておく
                # （実際の保存先は毎回 image_path のフォルダになる）
                self.save_folder = ""
            # ラベルも現在の状態で一度更新
            self._update_save_folder_label()
        except Exception:
            pass

        outer_layout.addWidget(self.bottom_bar, stretch=0)


        self._set_progress_visible(False)

        self._install_folder_shortcuts()
    
    def _set_progress_visible(self, on: bool) -> None:
        # 進捗（カウントラベル＋バー）をまとめて表示/非表示
        self.status_label.setVisible(on)
        self.progress_widget.setVisible(on)
        # 非表示時は見た目のゴミ防止で値もゼロ相当に
        if not on:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        # レイアウト再調整
        self.update_progress_alignment()
    
    def _clear_progress_display(self) -> None:
        """
        画像がフォーカスされていない時用に、
        進捗表示をニュートラル状態（- / -、0%相当）にリセットする。
        """
        # 初期化途中などで呼ばれても安全に
        if not hasattr(self, "status_label") or not hasattr(self, "progress"):
            return

        # 枚数を「- / -」に
        self.status_label.setText("- / -")

        # バーは 0/1 レンジで 0 にして、塗りつぶし無し＝色なし状態にする
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        # レイアウトのセンタリングを維持
        self.update_progress_alignment()

    def update_progress_alignment(self):
        """下部バー内で、進捗ウィジェットの中心を self.label の中心Xに合わせる。
           左右ラベルのストレッチ比を動的に調整する。
        """
        # ★追加：進捗ウィジェットが非表示ならレイアウト調整はスキップ
        if not getattr(self, "progress_widget", None) or not self.progress_widget.isVisible():
            # 非表示でもパス/保存先の省略表示は更新しておく
            self._update_path_elision()
            self._update_save_elision()
            return

        if not (getattr(self, "bottom_bar", None) and getattr(self, "progress_widget", None)):
            return
        if self.bottom_bar.width() <= 0 or self.label.width() <= 0:
            return

        # ウィンドウ座標での左端
        bar_left   = self.bottom_bar.mapTo(self, QPoint(0, 0)).x()
        label_left = self.label.mapTo(self, QPoint(0, 0)).x()

        bar_w  = self.bottom_bar.width()
        prog_w = self.progress_widget.width()
        label_center_x = label_left + self.label.width() / 2

        # 進捗左端の理想位置（バー左基準）
        target_left_in_bar = (label_center_x - prog_w / 2) - bar_left

        left_px  = max(0, min(bar_w - prog_w, int(round(target_left_in_bar))))
        right_px = max(0, bar_w - prog_w - left_px)

        # 0:0 を避ける
        if left_px == 0 and right_px == 0:
            left_px, right_px = 1, 1

        self.bottom_layout.setStretchFactor(self.path_panel,        left_px)
        self.bottom_layout.setStretchFactor(self.progress_widget,   0)
        self.bottom_layout.setStretchFactor(self.save_folder_label, right_px)

        # elide 再計算（幅が変わるため）
        self._update_path_elision()
        self._update_save_elision()

    def _with_adjust_preserved(self, nav_callable, *args, **kwargs):
        """
        現在の調整モードを保存してから nav_callable を実行し、
        終了後に必要なら調整モードを復元する小さなラッパー。
        nav_callable: インデックス移動や画像切替を行う既存関数
        """
        keep_adjust = bool(getattr(self, "_adjust_mode", False))
        nav_callable(*args, **kwargs)
        if keep_adjust:
            # 画像切替で OFF になっても ON に戻す
            try:
                self.set_adjust_mode(True)
            except Exception:
                pass
    
    def set_path_text(self, text: str) -> None:
        self._path_full_text = text or ""
        self._update_path_elision()

    def _choose_dir_icon(self, path: str) -> str:
        """
        フォルダ/zipルート用のアイコンを返す。
        - 通常フォルダや zip 内フォルダ → 📁
        - zip ファイルのルート        → 🗜️
        """
        icon = "📁"
        if not path:
            return icon
        try:
            if is_zip_uri(path):
                zp, inner = parse_zip_uri(path)
                # zip://...!/ または zip://...! のルートだけ 🗜️
                if inner == "" or inner == "/":
                    icon = "🗜️"
            elif is_archive_file(path):
                # 物理パスが .zip / .cbz など
                icon = "🗜️"
        except Exception:
            pass
        return icon

    def _format_dir_path_text(self, path: str) -> str:
        """左下のパスラベル用：フォルダ/zip のフルパス表示（テキスト部分のみ）。"""
        if not path:
            return ""
        return path

    def _format_image_path_text(self, path: str) -> str:
        """左下のパスラベル用：画像ファイル用（テキスト部分のみ）。"""
        if not path:
            return ""
        return path

    def _set_path_icon(self, icon) -> None:
        """左下パスラベル用のアイコンを更新（None なら非表示）。"""
        label = getattr(self, "path_icon_label", None)
        if not label:
            return

        if icon is not None and hasattr(icon, "isNull") and not icon.isNull():
            pm = icon.pixmap(18, 18)
            label.setPixmap(pm)
            label.show()
        else:
            label.clear()
            label.hide()

    def _update_path_icon_for_folder(self, path: str) -> None:
        """左下パスラベル：フォルダ/zip 用のアイコン更新。"""
        try:
            icon = None
            if is_zip_uri(path):
                zp, inner = parse_zip_uri(path)
                inner = inner.rstrip("/")
                if not inner:
                    # zip://...! ルート → zip ファイル自体のアイコン
                    icon = self._file_icon(zp) or self._folder_icon()
                else:
                    # zip 内フォルダ → 通常フォルダアイコン
                    icon = self._folder_icon()
            elif is_archive_file(path):
                # 実ファイルの .zip / .cbz など
                icon = self._file_icon(path) or self._folder_icon()
            else:
                icon = self._folder_icon()
        except Exception:
            icon = None

        self._set_path_icon(icon)

    def _update_path_icon_for_image(self, path: str) -> None:
        """左下パスラベル：画像ファイル用のアイコン更新。"""
        try:
            # さっき中央ラベル用に追加した _image_icon_for_entry を再利用
            icon = self._image_icon_for_entry(path)
        except Exception:
            icon = None

        self._set_path_icon(icon)

    def _update_path_elision(self) -> None:
        if not getattr(self, "path_label", None):
            return

        text = self._path_full_text or ""
        fm = self.path_label.fontMetrics()

        # ラベルの実効幅（パディング除く）
        panel = getattr(self, "path_panel", None)
        if panel is not None:
            avail_w = max(0, panel.contentsRect().width())
            # アイコンが見えている分だけ幅を減らす
            icon_label = getattr(self, "path_icon_label", None)
            if icon_label is not None and icon_label.isVisible():
                avail_w = max(0, avail_w - icon_label.width())
        else:
            # 念のため従来の fallback も残しておく
            avail_w = max(0, self.path_label.contentsRect().width())

        # 常に「進捗バー手前の余白」を空ける分だけ、使える幅を減らして判定
        safe_w = max(0, avail_w - int(getattr(self, "elide_gap_px", 0)))

        if fm.horizontalAdvance(text) > safe_w:
            shown = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideMiddle, safe_w)
        else:
            shown = text

        self.path_label.setText(shown)
        self.path_label.setToolTip(text)

    def open_options_dialog(self):
        dlg = OptionsDialog(
            self,
            overwrite=self.overwrite_mode,
            thumb_scroll_step=self.thumb_scroll_step,
            hq_zoom=self.hq_zoom,
            # ← ここに prompt_save_on_load は渡さない
        )

        # 生成後にチェック状態を注入（署名を増やさない方法）
        try:
            dlg.chk_prompt_save_on_load.setChecked(self.show_save_dialog_on_load)
        except Exception:
            pass

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.values()
            self.overwrite_mode    = bool(vals["overwrite_mode"])
            self.thumb_scroll_step = int(vals["thumb_scroll_step"])
            self.hq_zoom           = bool(vals["hq_zoom"])
            self.show_save_dialog_on_load = bool(vals["show_save_dialog_on_load"])

            # 保存
            self.settings.setValue("overwrite_mode", self.overwrite_mode)
            self.settings.setValue("thumb_scroll_step", self.thumb_scroll_step)
            self.settings.setValue("hq_zoom", self.hq_zoom)
            self.settings.setValue("show_save_dialog_on_load", self.show_save_dialog_on_load)

            # メニューのチェックも同期
            try:
                self.act_show_save_prompt.setChecked(self.show_save_dialog_on_load)
            except Exception:
                pass

            self._apply_thumb_scroll_step()

    def _on_quick_save_mode_changed(self, is_overwrite: bool):
        """クイック保存モード（連番/上書き）のラジオ変更時に呼ばれる"""
        self.overwrite_mode = bool(is_overwrite)
        try:
            self.settings.setValue("overwrite_mode", self.overwrite_mode)
        except Exception:
            pass

    def _on_quick_browse_dest(self):
        """クイックUIの『参照』ボタンで保存先フォルダを選ぶ"""

        # ダイアログの初期フォルダ候補
        base_dir = (
            getattr(self, "save_custom_dir", "")
            or getattr(self, "save_folder", "")
            or os.path.dirname(getattr(self, "image_path", "") or "")
            or os.path.expanduser("~")
        )

        d = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "保存先フォルダを選択",
            base_dir,
        )
        if not d:
            return  # キャンセル

        # ラジオを「フォルダ指定」に切り替え
        if hasattr(self, "rad_dest_custom_quick"):
            self.rad_dest_custom_quick.setChecked(True)

        # 内部状態と設定に反映
        self.save_dest_mode = "custom"
        self.save_custom_dir = d
        try:
            self.settings.setValue("save_dest_mode", "custom")
            self.settings.setValue("save_custom_dir", d)
        except Exception:
            pass

        # 実際の保存先＆右下ラベルも更新
        try:
            self._apply_save_folder_programmatically(d)
        except Exception:
            pass

    def _update_quick_save_mode_radios(self):
        """self.overwrite_mode をクイックUIのラジオへ反映（シグナル抑止）"""
        rb_seq = getattr(self, "rad_seq_quick", None)
        rb_ow  = getattr(self, "rad_ow_quick", None)
        if not (rb_seq and rb_ow):
            return

        # シグナル一時停止
        rb_seq.blockSignals(True)
        rb_ow.blockSignals(True)
        try:
            is_overwrite = bool(self.overwrite_mode)
            rb_ow.setChecked(is_overwrite)
            rb_seq.setChecked(not is_overwrite)
        finally:
            rb_seq.blockSignals(False)
            rb_ow.blockSignals(False)
    
    def _update_quick_save_dest_radios(self):
        """self.save_dest_mode をクイックUIの保存先ラジオへ反映（シグナル抑止）"""
        rb_same   = getattr(self, "rad_dest_same_quick", None)
        rb_custom = getattr(self, "rad_dest_custom_quick", None)
        btn_browse = getattr(self, "btn_dest_browse_quick", None)

        if not (rb_same and rb_custom):
            return

        is_custom = (getattr(self, "save_dest_mode", "same") == "custom")

        # クイック側の toggled ハンドラを動かさないようにシグナル一時停止
        rb_same.blockSignals(True)
        rb_custom.blockSignals(True)
        try:
            rb_custom.setChecked(is_custom)
            rb_same.setChecked(not is_custom)
        finally:
            rb_same.blockSignals(False)
            rb_custom.blockSignals(False)

        # 「フォルダ指定」のときだけ参照ボタンを有効化
        if btn_browse is not None:
            btn_browse.setEnabled(is_custom)

    def on_toggle_save_prompt(self, checked: bool):
        self.show_save_dialog_on_load = bool(checked)
        self.settings.setValue("show_save_dialog_on_load", self.show_save_dialog_on_load)

    def _apply_thumb_scroll_step(self):
        if hasattr(self, "listview"):
            self.listview.setVerticalScrollMode(
                QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
            )
            sb = self.listview.verticalScrollBar()
            sb.setSingleStep(int(self.thumb_scroll_step))

    def set_save_text(self, text: str) -> None:
        self._save_full_text = text or ""
        self._update_save_elision()

    def _update_save_elision(self) -> None:
        lbl = getattr(self, "save_folder_label", None)
        if not lbl:
            return

        prefix = "保存先："
        try:
            folder = self._effective_save_folder()
        except Exception:
            folder = getattr(self, "save_folder", "") or ""

        if not folder or not os.path.isdir(folder):
            lbl.setText(prefix + "—")
            lbl.setToolTip(prefix + "未設定")
            return

        fm = lbl.fontMetrics()
        avail_w = max(0, lbl.contentsRect().width())
        gap_px  = int(getattr(self, "elide_gap_px", 0))
        prefix_w = fm.horizontalAdvance(prefix)
        safe_w  = max(0, avail_w - gap_px - prefix_w)

        shown_plain = fm.elidedText(folder, QtCore.Qt.TextElideMode.ElideMiddle, safe_w)

        avg = max(1, fm.averageCharWidth())
        pad_count = max(0, gap_px // avg)
        pad_html = "&nbsp;" * pad_count

        url = QtCore.QUrl.fromLocalFile(folder).toString()

        # ★ ホバー状態に応じたリンク色
        link_color = (
            self._save_link_color_hover
            if getattr(self, "_save_link_hovered", False)
            else self._save_link_color_normal
        )

        # プレフィックスは非リンク、パスだけリンク（style で色指定）
        lbl.setText(
            f'{pad_html}{escape(prefix)}'
            f'<a href="{url}" style="color:{link_color}; text-decoration: underline;">'
            f'{escape(shown_plain)}</a>'
        )
        lbl.setToolTip(f'{prefix}{folder}')

    def _open_save_folder_link(self, href: str):
        try:
            path = QtCore.QUrl(href).toLocalFile() or href
            if path and os.path.isdir(path):
                os.startfile(path)  # Windows Explorerで開く
        except Exception as e:
            log_debug(f"[open-folder] failed: {e}")

    def _open_save_folder_link(self, href: str):
        try:
            path = QtCore.QUrl(href).toLocalFile() or href
            if path and os.path.isdir(path):
                os.startfile(path)  # Windows Explorerで開く
        except Exception as e:
            log_debug(f"[open-folder] failed: {e}")

    def _open_explorer_select(self, target: str) -> None:
        """Windows Explorer で target を開く。ファイルなら選択状態にする。"""
        try:
            if not target:
                return

            target = os.path.normpath(target)

            # ★ デバッグ：最初の判定
            is_file = os.path.isfile(target)
            is_dir  = os.path.isdir(target)
            log_debug(
                "[open-explorer-select]",
                "target=", repr(target),
                "is_file=", is_file,
                "is_dir=",  is_dir,
            )

            # ディレクトリならそのまま開く
            if is_dir:
                log_debug("[open-explorer-select] open dir via os.startfile")
                os.startfile(target)
                return

            # ファイルなら /select, でハイライト表示（存在しない場合は親フォルダだけ開く）
            if is_file:
                try:
                    log_debug("[open-explorer-select] explorer /select")
                    subprocess.Popen(["explorer", "/select,", target])
                    return
                except Exception as e:
                    log_debug("[open-explorer-select] explorer /select failed:", e)

            parent = os.path.dirname(target)
            if parent and os.path.isdir(parent):
                log_debug("[open-explorer-select] fallback open parent dir:", repr(parent))
                os.startfile(parent)
        except Exception as e:
            log_debug(f"[open-explorer-select] failed: {e}")

    def _open_current_path_from_label(self) -> None:
        """
        左下パスラベルクリック時：
        - 通常の画像     : 画像ファイルにフォーカスしてエクスプローラを開く
        - zip / zip内画像: zip本体を選択状態で開く
        - フォルダ表示中 : フォルダを開く
        """
        # --- まずは image_path または folder を取得 ---
        path = getattr(self, "image_path", "") or getattr(self, "folder", "") or ""

        # --- 保険①：image_path が空なのに実際には画像が開かれている場合 ---
        # （初回クリックで起きる問題はココ）
        if not getattr(self, "image_path", ""):
            # ★ 現在のサムネ選択から現在画像を推測
            idx = getattr(self, "current_index", -1)
            try:
                if idx >= 0 and hasattr(self, "image_list"):
                    candidate = self.image_list[min(idx, len(self.image_list) - 1)]
                    if candidate and os.path.isfile(candidate):
                        # image_path を復旧
                        self.image_path = candidate
                        path = candidate
            except Exception:
                pass

        # --- 保険②：それでも空なら folder を使う ---
        if not path:
            return

        # --- zip:// の場合は zip 本体に変換 ---
        if is_zip_uri(path):
            try:
                zip_path, inner = parse_zip_uri(path)

                # ★ memzip（zip内zip）なら、一番外側の物理zipまで辿ってそれをターゲットにする
                if isinstance(zip_path, str) and zip_path.startswith("memzip:"):
                    outer = zip_path
                    while isinstance(outer, str) and outer.startswith("memzip:"):
                        meta = _MEM_ZIP_META.get(outer)
                        if not meta:
                            break
                        outer = meta.get("outer") or ""
                    # outer が物理パスならそれを使う。辿れなければ元の zip_path を使う
                    target = outer or zip_path
                else:
                    # 通常の zip://C:/.../foo.zip!/...
                    target = zip_path
            except Exception:
                # 失敗したらとりあえず元のパス
                target = path
        else:
            target = path

        # --- 最終ターゲットを Explorer で開く ---
        self._open_explorer_select(target)

    def eventFilter(self, obj, event):
        # 0) 左下のフルパスラベル
        if obj is getattr(self, "path_label", None):
            if (event.type() == QtCore.QEvent.Type.MouseButtonRelease and
                event.button() == QtCore.Qt.MouseButton.LeftButton):
                try:
                    self._open_current_path_from_label()
                except Exception as e:
                    log_debug(f"[open-path-label] failed: {e}")
                return True  # クリックはここで消費
            return False  # 他のイベントは既定へ

        # 1) 右下の保存先ラベル
        if obj is getattr(self, "save_folder_label", None):
            et = event.type()

            # ホバー開始
            if et in (
                QtCore.QEvent.Type.Enter,
                QtCore.QEvent.Type.HoverEnter,
                QtCore.QEvent.Type.HoverMove,
            ):
                if not self._save_link_hovered:
                    self._save_link_hovered = True
                    self._update_save_elision()
                return False

            # ホバー終了
            if et in (
                QtCore.QEvent.Type.Leave,
                QtCore.QEvent.Type.HoverLeave,
            ):
                if self._save_link_hovered:
                    self._save_link_hovered = False
                    self._update_save_elision()
                return False

            # サイズ変化時は省略表示を更新
            if et == QtCore.QEvent.Type.Resize:
                self._update_save_elision()
                return False

            return False

        # 2) メインプレビュー（self.label）
        if obj is getattr(self, "label", None):

            # ダブルクリック：プレースホルダ表示中なら“表示中のフォルダ/zip”に入る
            if (event.type() == QtCore.QEvent.Type.MouseButtonDblClick and
                event.button() == QtCore.Qt.MouseButton.LeftButton and
                getattr(self, "_placeholder_active", False)):
                base = self._placeholder_path or getattr(self, "folder", "")
                if base and vfs_is_dir(base):
                    # 物理フォルダ / zip ファイル / zip:// 仮想フォルダ すべて対応
                    self.open_folder(base, _src="dblclick_preview")
                else:
                    QtWidgets.QApplication.beep()
                return True

            # ▼ プレースホルダ（フォルダアイコン）表示中の制御
            if getattr(self, "_placeholder_active", False):

                # 右ボタン press は通す（ここで止めると軌跡が始まらない）
                if (event.type() == QtCore.QEvent.Type.MouseButtonPress and
                    event.button() == QtCore.Qt.MouseButton.RightButton):
                    try: self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    except Exception: pass
                    return False  # 通す

                # 右ドラッグ中（move）も通す
                if event.type() == QtCore.QEvent.Type.MouseMove:
                    btns = event.buttons() if hasattr(event, "buttons") else QtCore.Qt.MouseButton.NoButton
                    try:
                        right_drag = (btns & QtCore.Qt.MouseButton.RightButton) == QtCore.Qt.MouseButton.RightButton
                    except TypeError:
                        right_drag = (int(btns) & int(QtCore.Qt.MouseButton.RightButton)) != 0
                    if right_drag:
                        try: self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                        except Exception: pass
                        return False  # 通す（右ドラッグ中）

                # 右ボタン release も通す（ジェスチャ判定＆フェードアウト用）
                if (event.type() == QtCore.QEvent.Type.MouseButtonRelease and
                    event.button() == QtCore.Qt.MouseButton.RightButton):
                    return False  # 通す

                # 左クリック（release）：反転トグルして消費
                if (event.type() == QtCore.QEvent.Type.MouseButtonRelease and
                    event.button() == QtCore.Qt.MouseButton.LeftButton):
                    try:
                        pt = event.position().toPoint()
                    except AttributeError:
                        pt = event.pos()
                    hit = getattr(self, "_placeholder_hit_rect", None)
                    self._placeholder_selected = bool(hit and hit.contains(pt))
                    QtCore.QTimer.singleShot(
                        0, lambda: self._show_folder_placeholder(self._placeholder_path or getattr(self, "folder", ""))
                    )
                    return True  # 消費

                # 左ボタン press は rubberBand 抑止のため消費
                if (event.type() == QtCore.QEvent.Type.MouseButtonPress and
                    event.button() == QtCore.Qt.MouseButton.LeftButton):
                    return True  # 消費

                # ホバー系は矢印固定にして“消費”（CropLabel に渡さない）
                if event.type() in (
                    QtCore.QEvent.Type.Enter, QtCore.QEvent.Type.HoverEnter,
                    QtCore.QEvent.Type.HoverMove, QtCore.QEvent.Type.Leave,
                    QtCore.QEvent.Type.HoverLeave,
                ):
                    try: self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    except Exception: pass
                    return True  # 消費（カーソルを+に戻されないように）

                # リサイズ時は中央維持のため描き直しだけ行い、イベントは通す
                if event.type() == QtCore.QEvent.Type.Resize:
                    QtCore.QTimer.singleShot(
                        0, lambda: self._show_folder_placeholder(self._placeholder_path or getattr(self, "folder", ""))
                    )
                    return False  # 通す

        # 他は既定へ
        return super().eventFilter(obj, event)

    def _first_subfolder(self, base_dir: str) -> str | None:
        if not base_dir or not os.path.isdir(base_dir):
            return None
        try:
            names = os.listdir(base_dir)
        except Exception:
            names = []
        dirs = []
        for n in names:
            p = os.path.join(base_dir, n)
            if os.path.isdir(p):
                dirs.append(os.path.normpath(p))
        try:
            import re
            def nkey(s):
                b = os.path.basename(s)
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", b)]
            dirs.sort(key=nkey)
        except Exception:
            dirs.sort()
        return dirs[0] if dirs else None

    def _make_fixed_crop_handler(self, size_tuple):
        def handler(checked):
            self.fixed_crop_triggered(size_tuple)
        return handler
     
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                path = url.toLocalFile()
                # フォルダなら OK
                if vfs_is_dir(path):
                    event.acceptProposedAction()
                    self.drag_over = True
                    return
                # ファイルなら拡張子チェック
                ext = os.path.splitext(path)[1].lower()
                if ext in self.IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    self.drag_over = True
                    return
        event.ignore()

    def dropEvent(self, event):
        self.drag_over = False
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    # フォルダなら open_folder を呼ぶ
                    if vfs_is_dir(path):
                        self._defer_save_dialog_once = True
                        self.open_folder(path)
                        event.acceptProposedAction()
                        return
                    # 画像ファイルなら従来通り open_image_from_path
                    ext = os.path.splitext(path)[1].lower()
                    if ext in self.IMAGE_EXTENSIONS:
                        self._defer_save_dialog_once = True
                        self.open_image_from_path(path)
                        event.acceptProposedAction()
                        return
        event.ignore()

    def open_image_from_path(self, file_path):
        if not file_path:
            return

        # --- VFSセーフな正規化（ここが肝） ---
        file_path = norm_vpath(file_path)       # ← zip:// を壊さない正規化
        folder    = vfs_parent(file_path)       # ← 親（zip内でもOK）
        log_debug(f"[open] path={file_path}")

        # === 0) 一括クロップ用の回転/反転履歴をリセット ===
        # 画像を開くたびに履歴をクリアしておくと、
        # 以前の画像で行った回転/反転が一括クロップに混ざるのを防げる。
        self._batch_transform_ops = []

        # === 1) 次回だけUI温存フラグ（読み出して即クリア） ===
        preserve = getattr(self, "_preserve_ui_on_next_load", None)
        self._preserve_ui_on_next_load = None
        log_debug(f"[open] preserve={type(preserve).__name__} -> {preserve}")

        # ★ 追加：一発トークンが空でも“粘着”から復元
        if preserve is None:
            preserve = getattr(self, "_nav_chain_state", None)
            if preserve is not None:
                log_debug(f"[open] preserve <- chain = {preserve}")

        # === 2) 保存先ダイアログ抑止（この1回だけ） ===
        skip_save_prompt = False
        try:
            s = getattr(self, "_suppress_save_dialog_paths", None)
            if s:
                n = norm_vpath(file_path)       # ← abspath/normcase ではなく VFS正規化キーで一致判定
                if n in s:
                    skip_save_prompt = True
                    s.discard(n)
        except Exception:
            pass
        log_debug(f"[open] skip_save_prompt={skip_save_prompt}")

        # === 3) 「いまの image_list に含まれているなら」軽量ルート ===
        same_list = (
            getattr(self, "model", None) is not None
            and getattr(self, "image_list", None) is not None
            and any(norm_vpath(p) == file_path for p in getattr(self, "image_list", []))
        )
        log_debug(f"[open] same_list={same_list}")

        # ------- 小ヘルパ：矩形/パネル復元（画像座標の QRect を渡す） -------
        def _restore_from_rect(rect_img: QtCore.QRect | None, is_fixed: bool, preserve_dict: dict | None):
            if not rect_img or rect_img.isNull():
                # 何も無ければ“全体”を矩形に
                img = getattr(self, "image", None)
                if not img:
                    return
                rect_img = QtCore.QRect(0, 0, img.width, img.height)

            x1, y1 = self.label.image_to_label_coords(rect_img.left(),  rect_img.top())
            x2, y2 = self.label.image_to_label_coords(rect_img.right(), rect_img.bottom())
            r_lbl = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)

            if is_fixed:
                self.label.fixed_crop_mode = True
                self.label.fixed_crop_rect_img = QtCore.QRect(rect_img)
                try:
                    self.label._sync_fixed_size_from_rect()
                except Exception:
                    pass
                if hasattr(self, "show_action_panel"):
                    self.show_action_panel(r_lbl, True)
            else:
                self._crop_rect_img = QtCore.QRect(rect_img)
                self.label.drag_rect_img = QtCore.QRect(rect_img)
                self._crop_rect = QtCore.QRect(r_lbl)
                if hasattr(self, "show_action_panel"):
                    self.show_action_panel(r_lbl, False)

            # アスペクト固定
            try:
                if preserve_dict and preserve_dict.get("aspect_lock", False) and hasattr(self, "set_aspect_lock"):
                    self.set_aspect_lock(True, preserve_dict.get("aspect_base"))
            except Exception:
                pass

            # サイズラベル/プレビュー
            try:
                if hasattr(self, "update_crop_size_label"):
                    self.update_crop_size_label(rect_img, img_space=True)
            except Exception:
                pass
            try:
                if hasattr(self, "safe_update_preview"):
                    QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(rect_img))
                elif hasattr(self, "_schedule_preview"):
                    self._schedule_preview(rect_img)
            except Exception:
                pass

            # 調整モード
            try:
                if preserve_dict and preserve_dict.get("adjust", False) and hasattr(self, "set_adjust_mode"):
                    self.set_adjust_mode(True)
                # ★追加：Nudge を必ず起こす（位置合わせは内部でやる）
                if hasattr(self, "ensure_nudge_visibility"):
                    self.ensure_nudge_visibility(True)
            except Exception:
                pass

            # Nudge 復帰
            try:
                if preserve_dict and preserve_dict.get("nudge", False) and hasattr(self, "ensure_nudge_visibility"):
                    self.ensure_nudge_visibility()
            except Exception:
                pass

        # ------- 小ヘルパ：_restore_adjust_state か、だめなら手動で -------
        def _try_restore(preserve_dict, *, fallback_rect_img=None, fallback_fixed=False):
            restored = False
            try:
                # 返り値が None の場合は“失敗扱い”にする（←ココ修正ポイント）
                r = self._restore_adjust_state(preserve_dict)
                restored = bool(r)
                log_debug(f"[open] _restore_adjust_state -> {r} (as {restored})")
            except Exception as e:
                log_debug(f"[open] _restore_adjust_state error: {e}")
                restored = False

            if not restored:
                # 'no_rect' が指定されていれば矩形は復元しない（調整モードだけ戻す用途）
                if preserve_dict and preserve_dict.get('no_rect'):
                    log_debug('[open] skip rect restore due to no_rect')
                    if preserve_dict.get('adjust', False) and hasattr(self, "set_adjust_mode"):
                        self.set_adjust_mode(True)
                    # ★追加：Nudge を必ず起こす
                    if hasattr(self, "ensure_nudge_visibility"):
                        self.ensure_nudge_visibility(True)
                else:
                    log_debug('[open] manual restore fallback')
                    _restore_from_rect(fallback_rect_img, fallback_fixed, preserve_dict)

        if same_list:
            # --- 先に index だけ合わせる ---
            try:
                self.current_index = self.image_list.index(file_path)
            except ValueError:
                self.current_index = max(0, getattr(self, "current_index", 0))
            log_debug(f"[open] light path: index={self.current_index}")

            # --- ★★ preserve が dict の場合は“読み込み前に”現在の矩形をスナップショットする ★★ ---
            prev_fixed = False
            prev_rect_img = None
            if isinstance(preserve, dict):
                # 固定枠が有効ならそれを最優先
                if getattr(self.label, "fixed_crop_mode", False) and getattr(self.label, "fixed_crop_rect_img", None):
                    prev_fixed = True
                    prev_rect_img = QtCore.QRect(self.label.fixed_crop_rect_img)
                else:
                    # 自由矩形
                    tmp = getattr(self, "_crop_rect_img", None)
                    prev_rect_img = _safe_qrect(tmp, fmt="xywh") if tmp is not None else QtCore.QRect()
                    if (prev_rect_img is None or prev_rect_img.isNull()) and getattr(self.label, "drag_rect_img", None):
                        prev_rect_img = _safe_qrect(self.label.drag_rect_img, fmt="xywh")

            # --- preserve が None のときだけ UI をクリア ---
            self._suspend_chain_clear += 1              # （ここから内部リセット扱い）
            try:
                if preserve is None:
                    try:
                        if hasattr(self.label, "clear_rubberBand"):
                            self.label.clear_rubberBand()
                        if hasattr(self.label, "clear_fixed_crop"):
                            self.label.clear_fixed_crop()
                    except Exception:
                        pass
                    try:
                        if hasattr(self, "_hide_action_panel"):
                            self._hide_action_panel()
                    except Exception:
                        pass

                # --- 画像だけ再読込 ---
                self.load_image_by_index(self.current_index) 
            finally:
                self._suspend_chain_clear -= 1          # ← 追加（必ず戻す）

            # --- サムネ選択同期（パス一致で安全に）---
            try:
                self._sync_thumb_selection()
            except Exception:
                pass

            # --- ★ 復元：まず _restore_adjust_state、ダメなら“読み込み前スナップショット”で復元 ★
            if isinstance(preserve, dict):
                _try_restore(preserve, fallback_rect_img=prev_rect_img, fallback_fixed=prev_fixed)
                self._preserve_ui_on_next_load = None
            try:
                self._update_nav_buttons()   # ★ 追加
            except Exception:
                pass
            # ★ 画像表示に切り替わるので、プレースホルダ状態を完全解除
            self._placeholder_active = False
            self._placeholder_selected = False      # ← 追加
            self._placeholder_hit_rect = None       # ← 追加
            self._placeholder_path = ""
            # ★ ActionPanel/Nudge をプレースホルダ都合で隠していた場合のフラグ解除
            self._panel_hidden_by_placeholder = False
            try:
                self.label.setCursor(QtCore.Qt.CursorShape.CrossCursor)  # ← 明示的に十字へ
            except Exception:
                pass
            log_debug("[open] done (light path)")
            return

        # ===== ここから“モデル作り直し”ルート =====
        log_debug("[open] heavy path: rebuild model")
        try:
            if getattr(self, "show_folder_placeholder", False):
                # プレースホルダ表示派：フォルダアイコンを一時表示
                if hasattr(self, "_show_folder_placeholder"):
                    self._show_folder_placeholder(folder)
            else:
                # 何も表示しない派：プレビューを空に
                if hasattr(self, "_clear_main_preview"):
                    self._clear_main_preview()
            # ★ 中央ラベルをフォルダ用表示へ（作ってあるヘルパを呼ぶだけ）
            try:
                self._update_center_label_for_folder(folder)
            except Exception:
                pass
        except Exception:
            pass

        # 1) 画像 + フォルダを再構築
        try:
            import re
            def natural_key(s: str):
                b = os.path.basename(s)
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", b)]

            dirs: list[str] = []
            files: list[str] = []
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                try:
                    if os.path.isdir(p):
                        dirs.append(os.path.normpath(p))
                        continue
                except Exception:
                    pass
                ext = os.path.splitext(p)[1].lower()
                if ext in self.IMAGE_EXTENSIONS:
                    files.append(os.path.normpath(p))

            dirs.sort(key=natural_key)
            files.sort(key=natural_key)

            # 画像ナビ用は従来どおり画像のみ
            self.image_list = files
            self.folder = folder

            # サムネ用は「フォルダ + 画像」
            browser_list = dirs + files

        except Exception:
            # 失敗時は最低限
            self.image_list = [file_path]
            self.folder = folder
            browser_list = self.image_list[:]
            dirs = []

        # 2) index（画像リスト内の位置）と、ビュー上の行(row)を分けて計算
        try:
            self.current_index = self.image_list.index(file_path)
        except ValueError:
            # file_path が image_list に無い場合（削除された/存在しないなど）は、
            # ファイル名の自然順で「次に近い画像」を選ぶ
            try:
                import re as _re
                def _nat_key(s: str):
                    b = os.path.basename(s)
                    return [int(t) if t.isdigit() else t.lower()
                            for t in _re.split(r"(\d+)", b)]

                key_target = _nat_key(file_path)
                L = self.image_list or []
                if not L:
                    self.current_index = 0
                else:
                    pos = 0
                    for i, p in enumerate(L):
                        if _nat_key(p) >= key_target:
                            pos = i
                            break
                    else:
                        # どれよりも後ろなら末尾にする
                        pos = len(L) - 1
                    self.current_index = pos
            except Exception:
                # 何かあっても最悪 0 にして落ちないようにする
                self.current_index = 0

        # フォルダを先頭に並べるので、ビュー上の行は「フォルダ数 + 画像index」
        view_row = (len(dirs) + self.current_index) if self.image_list else -1
        log_debug(f"[open] heavy: index={self.current_index} / count={len(self.image_list)} (view_row={view_row})")


        # 3) モデル張り直し（browser_list を使う！）
        try:
            self.model = ThumbnailListModel(browser_list, thumb_size=(160, 240))
            lv = getattr(self, "listview", None)
            if lv is not None:
                lv.setModel(self.model)

                # --- サムネ選択が変わったらメインに即反映（フォルダならプレースホルダ、画像なら画像） ---
                try:
                    lv.selectionModel().currentChanged.disconnect(self._on_thumb_current_changed)
                except Exception:
                    pass

                def _on_thumb_current_changed(cur: QtCore.QModelIndex, prev: QtCore.QModelIndex):
                    self._preview_from_thumb_index(cur)     # ← 事前に追加した共通ヘルパ

                self._on_thumb_current_changed = _on_thumb_current_changed
                lv.selectionModel().currentChanged.connect(self._on_thumb_current_changed)

                # 初期選択（view_row が有効ならその行）＋即時プレビュー
                if view_row >= 0:
                    idx = self.model.index(view_row, 0)
                    lv.setCurrentIndex(idx)
                    self._preview_from_thumb_index(idx)     # ← ここで“今の選択”を即反映
                else:
                    # 画像が無いときは選択クリア（プレースホルダは別処理で表示）
                    lv.setCurrentIndex(QtCore.QModelIndex())
        except Exception as e:
            log_debug("[open][heavy] setModel error:", e)
        # 4) preserve が None または 'no_rect' 指定のときは UI をクリア
        if (preserve is None) or (isinstance(preserve, dict) and preserve.get('no_rect')):
            try:
                if hasattr(self.label, "clear_rubberBand"):
                    self.label.clear_rubberBand()
                if hasattr(self.label, "clear_fixed_crop"):
                    self.label.clear_fixed_crop()
            except Exception:
                pass
            try:
                if hasattr(self, "_hide_action_panel"):
                    self._hide_action_panel()
            except Exception:
                pass

        # 5) 画像を読み込む
        self.load_image_by_index(self.current_index)

        # --- サムネ選択同期（パス一致で安全に）---
        try:
            self._sync_thumb_selection()
        except Exception:
            pass

        # 7) 保存先ダイアログ（フォルダ切替のときだけ1回）
        if (not skip_save_prompt) and (getattr(self, "_last_prompt_srcdir", None) != folder):
            try:
                self._maybe_prompt_save_on_load(folder)
                self._last_prompt_srcdir = folder
            except Exception:
                pass

        # 8) 復元（dict のときだけ）。矩形は _crop_rect_img / fixed_rect が生きていればそれを使い、無ければ full。
        if isinstance(preserve, dict):
            # heavy では読み込み“前”スナップショットは取れないので、現状から推測して復元
            fallback_fixed = bool(getattr(self.label, "fixed_crop_mode", False) and getattr(self.label, "fixed_crop_rect_img", None))
            fallback_rect = None
            if fallback_fixed:
                fallback_rect = QtCore.QRect(self.label.fixed_crop_rect_img)
            else:
                tmp = getattr(self, "_crop_rect_img", None)
                fallback_rect = _safe_qrect(tmp, fmt="xywh") if tmp is not None else QtCore.QRect()
                if (fallback_rect is None or fallback_rect.isNull()) and getattr(self.label, "drag_rect_img", None):
                    fallback_rect = _safe_qrect(self.label.drag_rect_img, fmt="xywh")
            _try_restore(preserve, fallback_rect_img=fallback_rect, fallback_fixed=fallback_fixed)
            self._preserve_ui_on_next_load = None
        try:
            self._update_nav_buttons()   # ★ ↑/戻る/進むの活性を更新
        except Exception:
            pass
        # === 画像を直接読み込んだケースの履歴を正規化（重複や「進む」側を排除） ===
        try:
            if not hasattr(self, "_nav_history"):
                self._nav_history = []
            if not hasattr(self, "_nav_pos"):
                self._nav_pos = -1

            cur_dir = os.path.normcase(os.path.abspath(self.folder)) if getattr(self, "folder", None) else None
            if cur_dir:
                # 進む側が残っていたら切り捨て
                if 0 <= self._nav_pos < len(self._nav_history) - 1:
                    self._nav_history = self._nav_history[:self._nav_pos + 1]

                # 末尾と同じパスは追加しない（重複防止）
                if not self._nav_history or os.path.normcase(os.path.abspath(self._nav_history[-1])) != cur_dir:
                    self._nav_history.append(cur_dir)

                # 現在位置を必ず末尾に合わせる
                self._nav_pos = len(self._nav_history) - 1

            # ボタン更新（ここで“進む”が必ずグレーアウトになる）
            self._update_nav_buttons()
        except Exception:
            pass
        # ★ 画像表示に切り替わるので、プレースホルダ状態を完全解除
        self._placeholder_active = False
        self._placeholder_selected = False      # ← 追加
        self._placeholder_hit_rect = None       # ← 追加
        self._placeholder_path = ""
        try:
            self.label.setCursor(QtCore.Qt.CursorShape.CrossCursor)  # ← heavy でも十字に統一
        except Exception:
            pass
        log_debug("[open] done (heavy path)")
    
    def _prefetch_neighbors(self):
        """現在の画像の前後を image_list から素直に先読みして LRU に入れる。"""
        try:
            base = int(getattr(self, "current_index", -1))
            L = getattr(self, "image_list", []) or []
            if base < 0 or not L:
                return
            for off in (+1, -1):
                j = base + off
                if 0 <= j < len(L):
                    p = norm_vpath(L[j])           # ← 文字生成せず、リストの値を正規化
                    # 署名取得に失敗しても MISS で返すように（下の _cache_get 修正と対）
                    hit = _cache_get(p)
                    if hit is None:
                        try:
                            im = open_image_any(p) # ← 実際に開いて
                            _cache_put(p, im)      # ← LRU へ
                            log_debug(f"[img-cache] PREFETCH {p}")
                        except Exception as e:
                            log_debug(f"[img-cache] prefetch skip: {p}: {e}")
        except Exception as e:
            log_debug("[img-cache] prefetch error:", e)

    def _move_thumb_focus(self, delta: int):
        lv = getattr(self, "listview", None)
        m  = getattr(self, "model", None)
        if lv is None or m is None:
            return

        cur = lv.currentIndex()
        row = cur.row() if cur.isValid() else (0 if m.rowCount() > 0 else -1)
        if row < 0:
            return

        count = m.rowCount()
        if count <= 0:
            return

        new_row = row + delta
        loop_on = bool(getattr(self, "_thumb_loop_enabled", False))
        if loop_on:
            if new_row < 0:
                new_row = count - 1
            elif new_row >= count:
                new_row = 0
        else:
            new_row = max(0, min(count - 1, new_row))
            if new_row == row:
                return

        idx = m.index(new_row, 0)

        # フォーカスは付ける
        lv.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

        # ★ 先にスクロールして位置を決める（ここがポイント）
        try:
            lv.scrollTo(idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        except Exception:
            pass

        # それから選択を更新（選択が“中央の位置”で初めて描かれるのでワンテンポ感が消える）
        lv.setCurrentIndex(idx)

        # メインプレビューも同期（フォルダならフォルダ絵、画像なら画像）
        try:
            self._preview_from_thumb_index(idx)
        except Exception:
            self.on_thumbnail_clicked(idx)

    def open_folder(self, dir_path: str, *, _from_history: bool = False, _src: str = ""):
        """
        サムネイル欄に「サブフォルダ＋画像」を並べる。
        - 再入防止（_opening_folder）とウォッチドッグログ
        - モデル貼替え中は QListView のシグナルをブロック
        - 初期表示（①②③）の確定
        - 履歴更新とナビボタン更新
        """
        # ====== 0) 再入ガードとウォッチドッグ ======
        if getattr(self, "_opening_folder", False):
            log_debug("[open_folder] re-entry suppressed")
            return
        # ★ 今のフォルダを離れる前に“現在の選択”を記憶
        try:
            self._remember_current_focus()
        except Exception:
            pass

        self._opening_folder = True
        if not hasattr(self, "_open_folder_watch"):
            self._open_folder_watch = QtCore.QTimer(self)
            self._open_folder_watch.setSingleShot(True)
            self._open_folder_watch.timeout.connect(lambda: log_debug("[open_folder][watchdog] still running..."))
        log_debug(f"[open_folder] >>> enter {dir_path}")
        self._open_folder_watch.start(3000)

        # ナビ世代をインクリメント & ログ
        self._nav_epoch += 1
        epoch = self._nav_epoch
        if getattr(self, "_nav_debug", False):
            log_debug(f"[nav] open_folder >>> src={_src} hist={_from_history} epoch={epoch} dir={dir_path}")

        import time
        self._nav_epoch_at = time.monotonic()
        try:
            # ====== 1) パス検証 ======
            if not dir_path:
                log_debug("[open_folder] empty path"); return

            # ★ zip:// URI は絶対に normpath しない（'zip://…' → 'zip:\…' に壊れる）
            if is_zip_uri(dir_path):
                target = dir_path
            else:
                target = os.path.normpath(dir_path)

            if not vfs_is_dir(target):
                log_debug(f"[open_folder] not a dir (vfs): {target}"); return

            # ★ いまのフォルダを退避（パスワード付きアーカイブなどで失敗したときに戻せるように）
            old_folder = getattr(self, "folder", None)
            self.folder = target

            # ====== 2) フォルダ/アーカイブ/画像一覧の取得 ======
            entries = []
            try:
                entries = vfs_listdir(dir_path)
            except PasswordProtectedArchiveError as e:
                # パスワード付きアーカイブはサポート外 → メッセージを出して元のフォルダに戻る
                log_debug('[open_folder] password-protected archive:', e)
                try:
                    self.folder = old_folder
                except Exception:
                    pass
                try:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "パスワード付きアーカイブ",
                        "このアーカイブはパスワード付きのため開けません。\n\n"
                        "パスワード付きアーカイブはサポートしていません。",
                    )
                except Exception:
                    pass
                return
            except Exception as e:
                log_debug('[open_folder] vfs_listdir error:', e); entries = []

            import re
            def natural_key(s: str):
                b = os.path.basename(s)
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', b)]

            dirs = []
            archives = []
            files = []
            for ent in entries:
                uri = ent.get('uri', '')
                name = ent.get('name', uri)
                if ent.get('is_dir') and not is_archive_name(name):
                    dirs.append(uri)
                elif ent.get('is_dir') and is_archive_name(name):
                    archives.append(uri)
                elif (not ent.get('is_dir')) and is_image_name(name):
                    files.append(uri)

            # ====== 3) 並び替え ======
            try:
                dirs.sort(key=natural_key); archives.sort(key=natural_key); files.sort(key=natural_key)
            except Exception:
                dirs.sort(); archives.sort(); files.sort()

            # ====== 4) image_list/ブラウザリスト構築 ======
            self.image_list = files[:]  # 画像だけ
            if not self.image_list:
                # 画像が無いフォルダに入ったときは保存先ラベルも即時更新
                self._update_save_folder_label()

            browser_list = dirs + archives + files  # サムネイル欄に並べる順番

            # ====== 4.5) 『上へ』用のフォーカス候補（子フォルダ）を探す【追加】 ======
            focus_row_up = -1
            if _src == "up":
                # _mark_child_for_up() で保存した“いま居た子フォルダ名”を使う
                target = (getattr(self, "_nav_last_child_basename", "") or "").lower()
                if target:
                    for i, d in enumerate(dirs):           # ← フォルダは先頭側（dirs）に並ぶ前提
                        if os.path.basename(d).lower() == target:
                            focus_row_up = i               # ← 見つけたフォルダ行
                            break

            # --- クリック用タイマーなど後始末 ---
            timer = getattr(self, "_click_timer", None)
            try:
                if isinstance(timer, QtCore.QTimer) and timer.isActive():
                    timer.stop()
            except Exception:
                pass
            if hasattr(self, "_pending_index"):
                self._pending_index = None

            # ====== 5) モデル差し替え or 新規セット ======
            if getattr(self, "model", None) and isinstance(self.model, ThumbnailListModel):
                self.model.reset_items(browser_list, thumb_size=(160, 240))
            else:
                self.model = ThumbnailListModel(browser_list, thumb_size=(160, 240))
                self.listview.setModel(self.model)

            # ====== 6) currentChanged の再配線（1回だけ） ======
            sm = self.listview.selectionModel()
            try:
                sm.currentChanged.disconnect(self._on_thumb_current_changed)
            except Exception:
                pass

            def _on_thumb_current_changed(cur: QtCore.QModelIndex, prev: QtCore.QModelIndex):
                # 最後にフォルダを開いてからの経過時間（秒）
                try:
                    import time
                    elapsed = time.monotonic() - float(getattr(self, "_nav_epoch_at", 0.0))
                except Exception:
                    elapsed = 999.0

                # 1) フォルダ貼り替え中 or 入場直後（揺り戻し期間）はプレビュー更新しない
                if getattr(self, "_opening_folder", False) or elapsed < 0.08:  # ← 80ms デバウンス
                    # ただし「最後に選んだ項目」は覚えておく（復元用）
                    try:
                        info = cur.data(QtCore.Qt.ItemDataRole.UserRole)
                        path = info.get("path") if isinstance(info, dict) else info
                        if path and getattr(self, "folder", None):
                            self._last_focus_by_dir[self._norm_path(self.folder)] = self._norm_path(path)
                    except Exception:
                        pass
                    return

                # 2) 通常時だけメインプレビューへ反映
                self._preview_from_thumb_index(cur)

                # 3) 最後に選んだ項目を記憶（復元用）
                try:
                    info = cur.data(QtCore.Qt.ItemDataRole.UserRole)
                    path = info.get("path") if isinstance(info, dict) else info
                    if path and getattr(self, "folder", None):
                        self._last_focus_by_dir[self._norm_path(self.folder)] = self._norm_path(path)
                except Exception:
                    pass

            self._on_thumb_current_changed = _on_thumb_current_changed
            sm.currentChanged.connect(self._on_thumb_current_changed)

            # ====== 7) 初期選択（①②③） ======
            restored = False
            sel_blocker = QtCore.QSignalBlocker(self.listview)
            try:
                initial_row = -1

                # ★ 7.a 「上へ」から来た場合は、まず『上へ』フォーカス候補を最優先【追加】
                if focus_row_up >= 0:
                    initial_row = focus_row_up
                    log_debug(f"[open_folder] up-focus hit row={initial_row} name={os.path.basename(dirs[initial_row])}")

                # ③ 戻る/進む/上へ → 前回選択の復元（『上へ』候補が無いときだけ使う）
                if initial_row < 0 and _from_history:
                    r = self._preferred_row_in(dir_path, browser_list)
                    if isinstance(r, int) and r >= 0:
                        initial_row = r
                        log_debug(f"[open_folder] history restore row={initial_row} path={browser_list[initial_row]}")

                # ①② 通常入場 or 復元失敗時の既定動作
                if initial_row < 0:
                    if self.image_list:
                        initial_row = len(dirs)  # 先頭画像
                    elif dirs:
                        initial_row = 0          # 先頭サブフォルダ
                    else:
                        initial_row = -1

                if initial_row >= 0:
                    idx = self.model.index(initial_row, 0)
                    sel_flags = (QtCore.QItemSelectionModel.SelectionFlag.Clear
                                | QtCore.QItemSelectionModel.SelectionFlag.Select
                                | QtCore.QItemSelectionModel.SelectionFlag.Current)
                    sm.setCurrentIndex(idx, sel_flags)
                    try:
                        self._preview_from_thumb_index(idx)
                        # 選択がフォルダならプレースホルダを強制再合成（既存）
                        info = idx.data(QtCore.Qt.ItemDataRole.UserRole)
                        path = info.get("path") if isinstance(info, dict) else info
                        if path and vfs_is_dir(path):
                            self._show_folder_placeholder(path, force=True)
                    except Exception:
                        pass

                    # 必要なら current_index を同期（画像のときのみ）
                    try:
                        info = idx.data(QtCore.Qt.ItemDataRole.UserRole)
                        path = info.get("path") if isinstance(info, dict) else info
                        if path in self.image_list:
                            self.current_index = self.image_list.index(path)
                        else:
                            self.current_index = -1
                    except Exception:
                        self.current_index = -1

                    restored = True
                else:
                    sm.clearSelection()
                    self.listview.setCurrentIndex(QtCore.QModelIndex())
                    try:
                        self.img_pixmap = None
                        self.image = None
                        self.label.clear()
                        self._placeholder_active = False
                        self._placeholder_path = ""
                    except Exception:
                        pass
                    self.current_index = -1
            finally:
                del sel_blocker

            # 7.1 フォールバック：
            # ここまでの処理で何も選択されていない場合、
            # フォルダ/zip 内に画像があるなら「先頭画像」を自動選択してプレビューを出す
            try:
                lv = getattr(self, "listview", None)
                m  = getattr(self, "model", None)
                # image_list が空じゃなくて、まだ何も選ばれていないときだけ
                if lv is not None and m is not None and self.image_list:
                    cur_idx = lv.currentIndex()
                    if not cur_idx.isValid():
                        first_img = self.image_list[0]
                        first_row = -1
                        # browser_list 内で先頭画像に対応する行を探す
                        for i, it in enumerate(browser_list):
                            p = it.get("path") if isinstance(it, dict) else it
                            if p == first_img:
                                first_row = i
                                break

                        if first_row >= 0:
                            idx = m.index(first_row, 0)
                            lv.setCurrentIndex(idx)
                            try:
                                # メインプレビューも同期（画像／フォルダ問わず）
                                self._preview_from_thumb_index(idx)
                            except Exception:
                                pass
            except Exception:
                # 保険なので、ここで落ちてもアプリ全体には影響なし
                pass

            # 『上へ』フォーカスマークは使い終わったら必ずクリア【追加】
            self._nav_last_child_basename = ""

            # ====== 8) 履歴の更新（戻る/進む） ======
            if not _from_history:
                # 初期化（保険）
                if not hasattr(self, "_nav_history"):
                    self._nav_history = []
                    self._nav_pos = -1

                # 「進む」側の枝は切る（分岐を捨てる）
                if 0 <= self._nav_pos < len(self._nav_history) - 1:
                    self._nav_history = self._nav_history[:self._nav_pos + 1]

                # 正規化して重複 push を回避（C:\ と C:/ の揺れも吸収）
                cur_norm  = self._norm_path(dir_path)
                last_norm = self._norm_path(self._nav_history[-1]) if self._nav_history else None

                if last_norm != cur_norm:
                    self._nav_history.append(dir_path)
                    self._nav_pos = len(self._nav_history) - 1
                    if getattr(self, "_nav_debug", False):
                        log_debug(f"[nav] push -> pos={self._nav_pos} path={dir_path}")
                else:
                    # 同じ場所なら pos を末尾に寄せるだけ（重複 push しない）
                    self._nav_pos = len(self._nav_history) - 1
                    if getattr(self, "_nav_debug", False):
                        log_debug(f"[nav] skip-dup -> pos={self._nav_pos} path={dir_path}")

            # ====== 9) 表示更新 ======
            if self.image_list:
                # いま何を表示しているかで分岐
                if getattr(self, "current_index", -1) >= 0:
                    # 画像を表示している（またはこれから表示する）→ プレースホルダは解除＆クロスカーソル
                    self._placeholder_active = False
                    self._placeholder_path = ""
                    try:
                        self.label.setCursor(QtCore.Qt.CursorShape.CrossCursor)
                    except Exception:
                        pass
                else:
                    # フォルダを表示している → プレースホルダ状態を維持＆矢印カーソル
                    try:
                        self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    except Exception:
                        pass

                # ★ 復元できなかったときだけ、先頭を遅延オープン（＝画像があるフォルダで初回入りのケース）
                if not restored:
                    def _deferred_open_first(epoch=epoch):
                        if epoch != getattr(self, "_nav_epoch", -1):
                            return
                        if not self.image_list:
                            return
                        try:
                            self.current_index = 0
                            self.open_image_from_path(self.image_list[0])
                            try:
                                self._sync_thumb_selection()
                            except Exception:
                                pass
                        except Exception as e:
                            log_debug("[open_folder][deferred] open first error:", e)
                    QtCore.QTimer.singleShot(0, _deferred_open_first)
            else:
                # 画像が無いフォルダ → プレースホルダ。既に初期選択で描画済み（restored=True）なら上書きしない
                if not restored:
                    QtCore.QTimer.singleShot(0, lambda: self._show_folder_placeholder(dir_path))

            # === 保存先ダイアログ（UIが新フォルダに切り替わってから“1回だけ”） ===
            try:
                epoch_now = self._nav_epoch
                dir_norm = os.path.normcase(os.path.abspath(dir_path or ""))

                def _deferred_prompt(epoch=epoch_now, path=dir_norm):
                    # まだ同じ遷移中か＆フォルダが切り替わっているかを確認
                    if epoch != getattr(self, "_nav_epoch", -1):
                        return
                    cur = os.path.normcase(os.path.abspath(getattr(self, "folder", "") or ""))
                    if cur != path:
                        return
                    # 画像があるフォルダだけに出したい場合（空でも出すならこのifは削除可）
                    if not getattr(self, "image_list", []):
                        return
                    # 同一フォルダ内では二度と出さない
                    last = os.path.normcase(os.path.abspath(getattr(self, "_last_prompt_srcdir", "") or ""))
                    if cur == last:
                        return
                    try:
                        self._maybe_prompt_save_on_load(self.folder)  # ここで初めて出す
                        self._last_prompt_srcdir = self.folder
                    except Exception:
                        pass

                # ★ UI切替が反映された“次のイベントループ”で実行（初期画像の遅延オープンより後に登録すると順序が安定）
                QtCore.QTimer.singleShot(0, _deferred_prompt)
            except Exception:
                pass
            log_debug(f"[open_folder] rows={len(browser_list)} images={len(self.image_list)}")

        except Exception as e:
            import traceback
            log_debug("[open_folder] ERROR:", e)
            log_debug(traceback.format_exc())
        finally:
            # ウォッチ停止＆フラグ解除
            try:
                self._open_folder_watch.stop()
            except Exception:
                pass
            self._opening_folder = False
            log_debug(f"[open_folder] <<< leave {dir_path}")

    # === Shift + ← / → でフォルダ移動 ===
    def _install_folder_shortcuts(self) -> None:
        # ショートカット作成（右ドラッグには一切触れない）
        s_left = QShortcut(QKeySequence("Shift+Left"), self)
        s_right = QShortcut(QKeySequence("Shift+Right"), self)
        # ウィンドウ全体で効かせる
        s_left.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        s_right.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        s_left.activated.connect(self._shortcut_prev_folder)
        s_right.activated.connect(self._shortcut_next_folder)
        # GC対策で保持
        self._shortcut_shift_left = s_left
        self._shortcut_shift_right = s_right

        # 入力欄（LineEdit/TextEdit）で Shift+←/→ を奪わないように、ショートカット抑止フィルタを入れる
        try:
            self._shift_arrow_guard = _ShiftArrowGuard(self)
            QtWidgets.QApplication.instance().installEventFilter(self._shift_arrow_guard)
        except Exception:
            pass

    def _norm_path(self, p: str) -> str:
        try:
            return norm_vpath(p)  # zip://… も含めて大小文字/区切りを吸収
        except Exception:
            return (p or "").lower()

    def _remember_current_focus(self) -> None:
        """今いる self.folder 内で、現在選択中のサムネのパスを記憶"""
        lv = getattr(self, "listview", None)
        m  = getattr(self, "model", None)
        d  = getattr(self, "folder", None)
        if not (lv and m and d):
            return
        idx = lv.currentIndex()
        if not idx.isValid():
            return
        info = idx.data(QtCore.Qt.ItemDataRole.UserRole)
        path = info.get("path") if isinstance(info, dict) else info
        if path:
            self._last_focus_by_dir[self._norm_path(d)] = self._norm_path(path)

    def _preferred_row_in(self, dir_path: str, browser_list: list) -> int:
        """
        dir_path で最後に選択していた項目（画像 or フォルダ）が
        browser_list（サムネモデルの元リスト）内で何行目かを返す。
        見つからなければ -1
        """
        want = self._last_focus_by_dir.get(self._norm_path(dir_path))
        if not want:
            return -1
        want = self._norm_path(want)

        for i, it in enumerate(browser_list):
            path = it.get("path") if isinstance(it, dict) else it
            if path and self._norm_path(path) == want:
                return i
        return -1

    def _shortcut_prev_folder(self) -> None:
        # フォーカスがテキスト入力なら何もしない（念のための二重ガード）
        fw = self.focusWidget()
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return
        # 命名ゆれに両対応（go_* 優先、無ければ show_*）
        prev_folder = getattr(self, "go_prev_folder", getattr(self, "show_prev_folder", None))
        if callable(prev_folder):
            prev_folder()

    def _shortcut_next_folder(self) -> None:
        fw = self.focusWidget()
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return
        next_folder = getattr(self, "go_next_folder", getattr(self, "show_next_folder", None))
        if callable(next_folder):
            next_folder()

    def set_save_folder(self):
        """
        メニュー『ファイル → 保存先を指定』から呼ばれたときの処理。
        ・保存先フォルダを選ぶ
        ・内部状態（save_dest_mode / save_custom_dir）を custom に更新
        ・右下の「保存先：」表示とクイックUIのラジオボタンも連動させる
        """
        # ダイアログの初期フォルダ候補（クイックUIと同じロジック）
        base_dir = (
            getattr(self, "save_custom_dir", "")
            or getattr(self, "save_folder", "")
            or os.path.dirname(getattr(self, "image_path", "") or "")
            or os.path.expanduser("~")
        )

        d = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "保存先フォルダを選択",
            base_dir,
        )
        if not d:
            return  # キャンセル

        # --- 「フォルダ指定」モードとして内部状態を更新 ---
        self.save_dest_mode = "custom"
        self.save_custom_dir = d
        try:
            self.settings.setValue("save_dest_mode", "custom")
            self.settings.setValue("save_custom_dir", d)
        except Exception:
            pass

        # 実際の保存先に反映（右下ラベル用の実効値もここでセット）
        try:
            self._apply_save_folder_programmatically(d)
        except Exception:
            # 念のためのフォールバック
            self.save_folder = d

        # 右下「保存先：」表示を現在の状態で更新
        try:
            self._update_save_folder_label()
        except Exception:
            pass

        # クイックUIの保存先ラジオ（読み込み元/フォルダ指定）にも反映
        try:
            self._update_quick_save_dest_radios()
        except Exception:
            pass


    def open_image(self):
        # ダイアログのフィルタ文字列も動的に生成
        patterns = ' '.join(f'*{ext}' for ext in self.IMAGE_EXTENSIONS)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "画像を開く", "", f"画像ファイル ({patterns})")
        if not file_path:
            return
        self.open_image_from_path(file_path)
        #self.move_progress_widget()
     
    def on_thumbnail_clicked(self, index: QtCore.QModelIndex):
        """サムネ単クリック：フォルダ→プレースホルダ表示、画像→即表示"""
        # open_folder 実行中は何もしない（再入防止）
        if getattr(self, "_opening_folder", False):
            return
        if not index.isValid():
            return

        info = index.data(QtCore.Qt.ItemDataRole.UserRole)
        if not info:
            return

        # 互換：UserRole が dict（新仕様）でも str（旧仕様）でもOK
        if isinstance(info, dict):
            path = info.get("path")
            is_dir = bool(info.get("is_dir"))
        else:
            path = info
            is_dir = vfs_is_dir(path)

        if not path:
            return

        if is_dir:
            # フォルダは単クリックでメインにフォルダのプレースホルダを表示
            try:
                self._show_folder_placeholder(path)
            except Exception:
                pass
            try:
                self.set_path_text(self._format_dir_path_text(path))
                self._update_path_icon_for_folder(path)
            except Exception:
                pass
            return

        # 画像はナビと同じルートで開く（UI温存トークン→open_image_from_path）
        try:
            if hasattr(self, "_prepare_preserve_for_nav"):
                self._prepare_preserve_for_nav()
        except Exception:
            pass

        # image_list 上の位置を同期（冪等）
        try:
            self.current_index = self.image_list.index(path)
        except ValueError:
            norm = norm_vpath(path)
            self.current_index = next(
                (i for i, p in enumerate(self.image_list)
                if norm_vpath(p) == norm),
                -1
            )
            if self.current_index < 0:
                return

        # ★ここが重要：load_image_by_index() は使わない
        self.open_image_from_path(path)

        # プレースホルダ解除 & カーソル戻し（従来の後処理）
        self._placeholder_active = False
        self._placeholder_path = ""
        try:
            self.label.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        except Exception:
            pass

    def keyPressEvent(self, event):
        if (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) and \
        (event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right)):
            self._prepare_preserve_for_nav()
            self._move_thumb_focus(-1 if event.key() == QtCore.Qt.Key.Key_Left else +1)
            event.accept()
            return
        super().keyPressEvent(event)

    def load_image_by_index(self, idx, *, _chain_all_ok: bool = False):
        """
        idx 番目の画像を読み込んで表示する。
        「全てOK」を押した場合は、その時点で image_list に含まれている
        欠損画像（vfs_is_file が False のもの）をすべて一括で除去する。
        """
        # 一旦プレビューを初期状態に
        self._set_preview_placeholder()
        self._preview_base_pixmap = None
        self._preview_src_key = None
        self.label._pan_offset_x = 0
        self.label._pan_offset_y = 0
        self._crop_rect = None
        self._crop_rect_img = None
        self.crop_size_label.setText("0 x 0")

        # === 内部リセット区間：チェーン解放を一時停止 ===
        self._suspend_chain_clear = getattr(self, "_suspend_chain_clear", 0) + 1
        try:
            # 画像切替時に一旦OFF（←内部都合のOFFなのでチェーンは消さない）
            if hasattr(self, "set_adjust_mode"):
                self.set_adjust_mode(False)
            else:
                if hasattr(self.label, "set_adjust_mode"):
                    self.label.set_adjust_mode(False)
                else:
                    setattr(self.label, "adjust_mode", False)
                try:
                    self._adjust_mode = False
                except Exception:
                    pass

            # 矩形類のクリア
            if hasattr(self.label, "clear_rubberBand"):
                self.label.clear_rubberBand()
            if hasattr(self.label, "clear_fixed_crop"):
                self.label.clear_fixed_crop()

            # 微調整オーバーレイやアクションパネルのクローズ
            if hasattr(self, "close_nudge_overlay"):
                self.close_nudge_overlay()
            else:
                w = getattr(self, "_nudge_overlay", None)
                if w is not None:
                    try:
                        w.close()
                    finally:
                        setattr(self, "_nudge_overlay", None)

            if getattr(self, "_action_panel", None):
                self._action_panel.close()
                self._action_panel = None

        finally:
            self._suspend_chain_clear -= 1

        # インデックス範囲外なら終了
        if not (0 <= idx < len(self.image_list)):
            self.set_path_text("")
            self._set_path_icon(None)
            self._set_progress_visible(False)
            return

        file_path = self.image_list[idx]

        # =========================================================
        # ① 欠損画像（物理/zip問わずファイルが存在しない）専用ルート
        # =========================================================
        if not vfs_is_file(file_path):
            # 欠損画像処理中フラグ ON（サムネ currentChanged からの再入を止める）
            prev_missing_flag = getattr(self, "_handling_missing_image", False)
            self._handling_missing_image = True
            try:
                mbox = QtWidgets.QMessageBox(self)
                mbox.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                mbox.setWindowTitle("ファイルが見つかりません")
                mbox.setText(
                    f"画像ファイルが存在しません:\n{file_path}\n\n"
                    "削除された可能性があるため、リストから除外します。"
                )

                # 標準の OK ボタン
                mbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)

                # 「全てOK」ボタン（仕様に合わせて文言変更）
                btn_all_ok = mbox.addButton(
                    "全てOK",
                    QtWidgets.QMessageBox.ButtonRole.AcceptRole,
                )

                mbox.exec()
                clicked_all_ok = (mbox.clickedButton() is btn_all_ok)

                # -------- 「全てOK」：現在の image_list 内の欠損画像を一括削除 --------
                if clicked_all_ok:
                    total = len(self.image_list)
                    if total == 0:
                        self.set_path_text("")
                        self._set_path_icon(None)
                        self._set_progress_visible(False)
                        return

                    # 欠損行・正常行を事前に仕分け（self.image_list = 画像だけのリスト）
                    missing_indices: list[int] = []
                    present_indices: list[int] = []
                    for i, p in enumerate(self.image_list):
                        try:
                            ok = vfs_is_file(p)
                        except Exception:
                            ok = False
                        if ok:
                            present_indices.append(i)
                        else:
                            missing_indices.append(i)

                    if not missing_indices:
                        # もう欠損が無ければ、「全てOK」は単体削除扱いに落とす
                        clicked_all_ok = False
                    else:
                        # 欠損パス一覧（サムネモデル側と同期させるのに使う）
                        missing_set = set(missing_indices)
                        missing_paths = [self.image_list[i] for i in missing_indices]

                        # すべて欠損なら全部消して UI をクリア
                        if not present_indices:
                            # サムネモデルからも欠損エントリを一括削除
                            if getattr(self, "model", None) is not None:
                                try:
                                    self.model.remove_paths(missing_paths)
                                except Exception:
                                    pass

                            # 画像ナビ用リストも空に
                            self.image_list = []

                            self.set_path_text("")
                            self._set_path_icon(None)
                            self._set_progress_visible(False)
                            return

                        # 通常ケース: 正常な画像は一部残る
                        # 削除前の self.image_list を前提にフォーカス位置を決める
                        later = [i for i in present_indices if i > idx]
                        if later:
                            orig_focus = later[0]
                        else:
                            # idx より後ろに正常画像が無ければ、一番最後の正常画像
                            orig_focus = present_indices[-1]

                        # image_list から欠損画像を削除
                        old_image_list = self.image_list
                        self.image_list = [
                            p for i, p in enumerate(old_image_list) if i not in missing_set
                        ]

                        # サムネモデル側からも欠損パスを削除
                        if getattr(self, "model", None) is not None:
                            try:
                                self.model.remove_paths(missing_paths)
                            except Exception:
                                pass

                        # 削除後の new_index = 元インデックス - それより前に削除された数
                        removed_before = sum(1 for r in missing_indices if r < orig_focus)
                        new_index = orig_focus - removed_before

                        if not (0 <= new_index < len(self.image_list)):
                            new_index = max(0, len(self.image_list) - 1)

                        self.current_index = new_index
                        # 次に表示すべき画像を通常ルートで開く
                        return self.load_image_by_index(new_index)

                # -------- ここからは「OK」だけ押されたとき（単体削除）の処理 --------
                try:
                    if 0 <= idx < len(self.image_list):
                        # 削除対象のパスを控えておく
                        target_path = self.image_list[idx]

                        # 画像ナビ用リストから 1 件削除
                        del self.image_list[idx]

                        # サムネモデル側からも該当パスを削除
                        if getattr(self, "model", None) is not None:
                            try:
                                self.model.remove_paths([target_path])
                            except Exception:
                                pass
                except Exception:
                    pass

                if not self.image_list:
                    self.set_path_text("")
                    self._set_path_icon(None)
                    self._set_progress_visible(False)
                    return

                # 消えた画像の「一つ次」にフォーカス
                if idx < len(self.image_list):
                    new_index = idx          # 削除後、元の「次の画像」がここに来る
                else:
                    new_index = len(self.image_list) - 1   # 末尾が消えたら新しい末尾

                self.current_index = new_index
                return self.load_image_by_index(new_index)

            finally:
                # 欠損処理チェーンが完全に終わったタイミングでフラグを戻す
                self._handling_missing_image = prev_missing_flag

        # =========================================================
        # ② ファイルが存在する通常ルート
        # =========================================================
        try:
            im = _cache_get(file_path)
            if im is None:
                # ★ 初回は実際に開く（zip:// は open_image_any がメモリ展開）
                im = open_image_any(file_path)
                # ★ キャッシュに保管（以後は再デコード不要）
                _cache_put(file_path, im)

            # 従来どおり表示用に RGBA 化して保持
            img = im.convert("RGBA") if im.mode in ("P", "PA") else im
            self.image = img.copy()
            self._base_pixmap_dirty = True
            self._scaled_pixmap = None
            self._scaled_key = None

        except Exception as e:
            # RAR 用の外部ツールが見つからないときは専用メッセージ
            msg = str(e)
            if "Cannot find working tool" in msg:
                QtWidgets.QMessageBox.critical(
                    self,
                    "画像読み込みエラー",
                    "RAR を開くには 7-Zip などをインストールし、\n"
                    "7z（または unrar）を PATH に通してください。"
                )
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "画像読み込みエラー",
                    f"{file_path} の読み込み中にエラーが発生しました:\n{msg}"
                )
            return

        self.image_path = file_path
        self.label.clear_rubberBand()

        # zip:// のときでも綺麗な表示名にする
        try:
            base_name = vfs_display_name(file_path, False)
        except Exception:
            base_name = os.path.basename(file_path)

        width, height = self.image.size
        info_text = f"{base_name}  —  {width} x {height}"

        # ★ 通常パスと zip:// の両方に対応した画像用アイコン
        icon = self._image_icon_for_entry(file_path)

        if hasattr(self, "info_banner"):
            self.info_banner.set_content(icon, info_text)

        # 固定枠でない時だけリセット（プリセット✔やカスタムはクリア）
        if not (self.label.fixed_crop_mode and self.label.fixed_crop_size):
            self.label.clear_fixed_crop()
            if hasattr(self, "crop_actions"):
                for act in self.crop_actions.values():
                    act.setChecked(False)
            if hasattr(self, "custom_action") and self.custom_action.isChecked():
                self.custom_action.setChecked(False)

        self.zoom_scale = 1.0  # --- 画像切り替え時はズームリセット ---
        self.show_image()

        # ✅ show_image() 後に固定枠UIを同期
        self._sync_fixed_ui_after_image_change()

        progress_text = f"{self.current_index + 1} / {len(self.image_list)}"
        self.status_label.setText(progress_text)
        self.progress_widget.adjustSize()
        self.update_progress_alignment()

        self.progress.setMaximum(len(self.image_list))
        self.progress.setValue(self.current_index + 1)

        self.set_path_text(self._format_image_path_text(self.image_path))
        self._update_path_icon_for_image(self.image_path)

        self._set_progress_visible(True)

        if self.model:
            self._sync_thumb_selection()

        QtCore.QTimer.singleShot(0, self._prefetch_neighbors)

        self.zoom_label.show_zoom(self.zoom_scale)

        if self.save_folder:
            text = f"保存先: {self.save_folder}"
        elif self.image_path:
            text = f"保存先: {os.path.dirname(self.image_path)}"
        else:
            text = "保存先: "
        self.set_save_text(text)

    def _count_images_on_disk(self, folder: str) -> int:
        """フォルダ直下の画像ファイル数を数える（拡張子は既存の集合を尊重）。"""
        try:
            exts = getattr(self, "IMAGE_EXTENSIONS",
                        {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
            exts = {e.lower() for e in exts}
            cnt = 0
            for n in os.listdir(folder):
                p = os.path.join(folder, n)
                if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                    cnt += 1
            return cnt
        except Exception:
            return 0

    def _update_center_label_for_folder(self, folder_path: str, prefer_built_list: bool = True) -> None:
        """
        中央ラベルを「フォルダ / zip の名前」のみ表示する。
        画像枚数（○枚）は表示しない。
        """
        import os

        try:
            # 表示用の名前（zip:// のときもそれなりに見やすく）
            try:
                # vfs_display_name があればそれを優先して使う
                name = vfs_display_name(folder_path, True)
            except Exception:
                name = os.path.basename(folder_path) or folder_path

            text = name

            # アイコン部分はこれまで通り
            icon = None
            try:
                # zip のルートだけ zip ファイルの OS アイコン、それ以外はフォルダアイコン
                if is_zip_uri(folder_path):
                    zp, inner = parse_zip_uri(folder_path)
                    is_root = (inner == "")
                    if is_root:
                        icon = self._file_icon(zp) or self._folder_icon()
                    else:
                        icon = self._folder_icon()
                else:
                    icon = self._folder_icon()
            except Exception:
                try:
                    icon = self._folder_icon()
                except Exception:
                    icon = None

            if hasattr(self, "info_banner"):
                self.info_banner.set_content(icon, text)

        except Exception:
            # どうしてもダメな場合は、せめてフォルダ名だけは出す
            try:
                name = os.path.basename(os.path.normpath(folder_path)) or folder_path
                icon = None
                try:
                    icon = self._folder_icon()
                except Exception:
                    icon = None
                if hasattr(self, "info_banner"):
                    self.info_banner.set_content(icon, name)
            except Exception:
                pass
     
    def on_progress_jump(self, value):
        if value < 1 or value > len(self.image_list):
            return
        # 調整モードほかUI状態をスナップショット
        snap = self._snapshot_adjust_state()
        # 画像インデックス反映 → 読み込み
        index = value - 1
        self.current_index = index
        self.load_image_by_index(self.current_index)
        # 調整モード/パネルを復元
        self._restore_adjust_state(snap)

    def show_prev_image(self, *args):
        # サムネイル欄の「現在行」を1つ左（前）へ。
        # フォルダにも移動でき、メインプレビューも同期します。
        self._prepare_preserve_for_nav()
        self._move_thumb_focus(-1)

    def show_next_image(self, *args):
        # サムネイル欄の「現在行」を1つ右（次）へ。
        # フォルダにも移動でき、メインプレビューも同期します。
        self._prepare_preserve_for_nav()
        self._move_thumb_focus(+1)

    def _on_nav_prev_clicked(self):
        # 画像移動の直前に復元方針をトークンに積む
        if hasattr(self, "_prepare_preserve_for_nav"):
            self._prepare_preserve_for_nav()
        self._move_thumb_focus(-1)

    def _on_nav_next_clicked(self):
        if hasattr(self, "_prepare_preserve_for_nav"):
            self._prepare_preserve_for_nav()
        self._move_thumb_focus(+1)

    def _prepare_preserve_for_nav(self) -> None:
        """
        連続移動直前に現在のUI状態を保存。
        - 固定枠: スナップショットをそのまま持ち越し
        - 通常矩形: 矩形は毎回消し、adjust(微調整)だけ維持（no_rect 指定）
        - UIが空に見える瞬間は直前のトークン(_nav_chain_state)を持ち越し
        """
        lbl = getattr(self, "label", None)

        def _meaningful(d):
            return bool(d) and (d.get("fixed") or d.get("adjust"))

        # いまの状態をまとめて取得
        try:
            snap = self._snapshot_adjust_state()
        except Exception:
            snap = None

        # ★ここがポイント：スナップが空でも _adjust_mode を見る
        fixed_on  = bool(lbl and getattr(lbl, "fixed_crop_mode", False))
        adjust_on = bool((snap and snap.get("adjust")) or getattr(self, "_adjust_mode", False))

        preserve = None

        # 1) 固定枠はスナップショット優先でそのまま持ち越し
        if fixed_on and _meaningful(snap):
            preserve = dict(snap)
            preserve.pop("no_rect", None)  # 念のため消しておく（固定は枠も復元）
            # ★追加：Nudgeの可視を明示的に持ち越し
            preserve["nudge"] = bool(getattr(self, "_nudge_overlay", None) and self._nudge_overlay.isVisible())

            self._preserve_ui_on_next_load = preserve
            self._nav_chain_state = preserve
            log_debug(f"[nav] preserve-next = (fixed snapshot) {preserve}")
            return

        # 2) 通常矩形：adjustがONなら矩形は毎回消して adjust だけ維持
        if adjust_on and not fixed_on:
            preserve = {
                "adjust": True,
                "no_rect": True,  # ← これで毎回 矩形は復元しない
                "aspect_lock": bool(getattr(lbl, "_aspect_lock", False)),
                "aspect_base": getattr(lbl, "_aspect_base_wh", None),
                "panel_visible": False,
                "nudge": False,
            }
            self._preserve_ui_on_next_load = preserve
            self._nav_chain_state = preserve
            log_debug(f"[nav] preserve-next = (adjust/no_rect) {preserve}")
            return

        # 3) UIが空に見える瞬間：直前の粘着トークンを持ち越し
        prev = getattr(self, "_nav_chain_state", None)
        if _meaningful(prev):
            preserve = prev
            self._preserve_ui_on_next_load = preserve
            # self._nav_chain_state は既に prev なのでそのまま
            log_debug(f"[nav] chain-preserve carry over: {preserve}")
            return

        # 4) 何も保持しない
        self._preserve_ui_on_next_load = None
        self._nav_chain_state = None
        log_debug("[nav] preserve-next = None")

    # === 兄弟フォルダの中で前後に移動（先頭画像を開く） ===
    def _sibling_dirs(self):
        """現在の self.folder と同じ親の配下にあるディレクトリ一覧（natural sort）"""
        folder = getattr(self, "folder", None)
        if not folder:
            return []
        parent = os.path.dirname(folder)
        try:
            dirs = [os.path.normpath(os.path.join(parent, d))
                    for d in os.listdir(parent)
                    if os.path.isdir(os.path.join(parent, d))]
        except Exception:
            return []
        dirs.sort(key=natural_key)
        return dirs

    def _open_first_image_in(self, folder_path: str) -> bool:
        """folder_path 内の最初の画像を開く（成功で True）"""
        try:
            exts = getattr(self, "IMAGE_EXTENSIONS", {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
            files = [os.path.normpath(os.path.join(folder_path, f))
                    for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in exts]
            if not files:
                return False
            files.sort(key=natural_key)
            self.open_image_from_path(files[0])
            return True
        except Exception:
            return False
    def _dir_has_image(self, d: str) -> bool:
        """直下に1枚でも画像があれば True"""
        exts = getattr(self, "IMAGE_EXTENSIONS",
                    {".png",".jpg",".jpeg",".bmp",".gif",".webp",".tif",".tiff"})
        try:
            for n in os.listdir(d):
                p = os.path.join(d, n)
                if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                    return True
        except Exception:
            pass
        return False

    def _natural_key(self, b: str):
        """
        自然順ソート用キー:
        - 連続する 10進数字(0-9 / 全角) は数値として扱う
        - それ以外（丸数字「④」などの isdigit だが isdecimal でないもの）は文字列として扱う
        """
        import re

        if not isinstance(b, str):
            b = str(b)

        parts = re.split(r"(\d+)", b)
        key = []
        for t in parts:
            if not t:
                continue
            # ★ isdigit() ではなく isdecimal() を使うのがポイント
            if t.isdecimal():
                try:
                    key.append(int(t))
                except Exception:
                    # 万一 int 変換できなかったときの保険
                    key.append(t.lower())
            else:
                key.append(t.lower())
        return key

    def _jump_sibling_folder(self, step: int, *, require_images: bool = True) -> bool:
        """
        現在の self.folder から兄弟フォルダへ step(±1)方向にジャンプ。
        require_images=True のときは「画像のないフォルダ」を飛ばして進む。
        成功したら True / 見つからなければ False。
        """
        cur = getattr(self, "folder", None)
        if not cur:
            return False

        parent = os.path.dirname(os.path.normpath(cur))
        if not parent or not os.path.isdir(parent):
            return False

        try:
            names = os.listdir(parent)
        except Exception:
            names = []

        dirs = []
        for n in names:
            p = os.path.join(parent, n)
            if os.path.isdir(p):
                dirs.append(p)

        dirs.sort(key=self._natural_key)

        # 現在インデックス
        try:
            idx = dirs.index(os.path.normpath(cur))
        except ValueError:
            norm_cur = os.path.normcase(os.path.abspath(cur))
            idx = next((i for i, d in enumerate(dirs)
                        if os.path.normcase(os.path.abspath(d)) == norm_cur), -1)
            if idx < 0:
                return False

        i = idx + step
        skipped = 0
        while 0 <= i < len(dirs):
            d = dirs[i]
            if not require_images or self._dir_has_image(d):
                log_debug(f"[nav] jump_sibling step={step} from={cur} to={d} skipped={skipped}")
                self.open_folder(d)  # 履歴やボタン更新は open_folder 側で済む
                return True
            i += step
            skipped += 1

        log_debug(f"[nav] jump_sibling no target (step={step}) from={cur}")
        QtWidgets.QApplication.beep()
        return False

    def go_prev_folder(self):
        # 画像が無いフォルダもスキップせず移動する
        self._jump_sibling_folder(-1)

    def go_next_folder(self):
        # 画像が無いフォルダもスキップせず移動する
        self._jump_sibling_folder(+1)

    def delete_current_image(self):
        """現在の画像ファイルをディスク＆リストから削除し、次/前の画像を表示する"""
        # 画像が選ばれていなければ何もしない
        if not self.image_list or not (0 <= getattr(self, "current_index", -1) < len(self.image_list)):
            return

        idx = self.current_index
        file_path = self.image_list[idx]

        # 圧縮ファイル内（zip://〜）の画像は削除不可
        if is_zip_uri(file_path):
            QMessageBox.information(
                self,
                "削除できません",
                "圧縮ファイル内の画像はこのアプリからは削除できません。\n"
                "元の圧縮ファイルをエクスプローラなどで編集してください。"
            )
            return

        # 確認ダイアログ
        ret = QMessageBox.question(
            self,
            "削除の確認",
            f"以下の画像ファイルを完全に削除します。\n\n{file_path}\n\nよろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        # 削除後に表示したい候補（次があれば次、なければ前）を先に覚えておく
        next_path = None
        if idx + 1 < len(self.image_list):
            next_path = self.image_list[idx + 1]
        elif idx - 1 >= 0:
            next_path = self.image_list[idx - 1]

        # ① ファイル削除
        try:
            os.remove(file_path)
        except Exception as e:
            QMessageBox.warning(self, "削除エラー", f"ファイルの削除に失敗しました:\n{e}")
            return

        # ② フォルダを開き直してモデルをリセット（_gen を進めて遅延ジョブを無効化）
        cur_dir = getattr(self, "folder", os.path.dirname(file_path))
        self.open_folder(cur_dir, _src="delete")

        # ③ 可能なら削除前に決めておいた候補画像にフォーカスを戻す
        if next_path and os.path.exists(next_path):
            try:
                self.current_index = self.image_list.index(next_path)
            except ValueError:
                norm = os.path.normcase(os.path.abspath(next_path))
                self.current_index = next(
                    (i for i, p in enumerate(self.image_list)
                    if os.path.normcase(os.path.abspath(p)) == norm),
                    -1
                )
            if self.current_index >= 0:
                self.load_image_by_index(self.current_index)
                try:
                    self._sync_thumb_selection()
                except Exception:
                    pass
                return

        # ④ 候補が無い/見つからない場合のフォールバック
        if not self.image_list:
            # 画像がもう無ければプレースホルダ
            self._set_preview_placeholder()
            self._set_progress_visible(False)
        else:
            # フォルダ再読込後の最寄りインデックスを安全に表示
            self.current_index = max(0, min(idx, len(self.image_list) - 1))
            self.load_image_by_index(self.current_index)
            try:
                self._sync_thumb_selection()
            except Exception:
                pass

    def _update_nav_buttons(self):
        # 履歴が無ければ初期化
        if not hasattr(self, "_nav_history"):
            self._nav_history = []
        if not hasattr(self, "_nav_pos"):
            self._nav_pos = -1

        hist = self._nav_history
        pos  = int(self._nav_pos)

        # ◀ 戻る / ▶ 進む は「0 <= pos」の下限チェックが重要
        can_back = (0 <= pos - 1)
        can_fwd  = (0 <= pos < len(hist) - 1)

        # 親フォルダがあるか（zip:// を含め VFS 全対応）
        cur_dir = getattr(self, "folder", "") or ""
        if cur_dir:
            parent = vfs_parent(cur_dir) or ""
        else:
            parent = ""
        # _norm_path は内部で norm_vpath を呼ぶので zip:// も含めて比較OK
        can_up = bool(cur_dir and parent and self._norm_path(parent) != self._norm_path(cur_dir))
        try:
            can_reload = bool(cur_dir and vfs_is_dir(cur_dir))
        except Exception:
            can_reload = bool(cur_dir)

        # ボタン反映（存在するものだけ）
        if hasattr(self, "btn_nav_back"): self.btn_nav_back.setEnabled(can_back)
        if hasattr(self, "btn_nav_fwd"):  self.btn_nav_fwd.setEnabled(can_fwd)
        if hasattr(self, "btn_nav_up"):   self.btn_nav_up.setEnabled(can_up)
        if hasattr(self, "btn_nav_reload"): self.btn_nav_reload.setEnabled(can_reload)
        if hasattr(self, "btn_nav_loop"): self.btn_nav_loop.setEnabled(True)

    def _nav_go(self, delta: int):
        new_pos = self._nav_pos + int(delta)
        if 0 <= new_pos < len(self._nav_history):
            self._nav_pos = new_pos
            target = self._nav_history[self._nav_pos]
            src = "back" if int(delta) < 0 else "forward"
            self.open_folder(target, _from_history=True, _src=src)

    def _nav_reload(self):
        cur = getattr(self, "folder", "") or ""
        if not cur:
            return
        try:
            if not vfs_is_dir(cur):
                return
        except Exception:
            return

        # ?????????????????????????????????????
        lv = getattr(self, "listview", None)
        sb_v = lv.verticalScrollBar() if lv else None
        sb_h = lv.horizontalScrollBar() if lv else None
        v_val = sb_v.value() if sb_v else None
        h_val = sb_h.value() if sb_h else None

        # ???????????????????
        self.open_folder(cur, _from_history=True, _src="reload")

        try:
            if sb_v is not None and v_val is not None:
                QtCore.QTimer.singleShot(0, lambda v=v_val, bar=sb_v: bar.setValue(min(v, bar.maximum())))
            if sb_h is not None and h_val is not None:
                QtCore.QTimer.singleShot(0, lambda v=h_val, bar=sb_h: bar.setValue(min(v, bar.maximum())))
        except Exception:
            pass

    def _on_thumb_loop_toggled(self, on: bool):
        self._thumb_loop_enabled = bool(on)
        # 設定に保存（アプリ再起動後もON/OFFを復元）
        try:
            self.settings.setValue("thumb_loop_enabled", self._thumb_loop_enabled)
        except Exception:
            pass

    def _create_thumb_nav_icon(self, kind: str) -> QtGui.QIcon:
        """
        サムネイルナビ用の「再読み込み」「リピート」アイコンを返す。
        まず icons/ 下の PNG を探し、無ければフォールバックで簡易アイコンを描く。
        kind: "reload" / "loop"
        """
        # ---- 1) 外部 PNG の読み込みを試す ----
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            base_dir = "."
        icon_dir = os.path.join(base_dir, "icons")

        name_map = {
            "reload": "reload.png",
            "loop":   "repeat.png",
        }
        fname = name_map.get(kind)
        if fname:
            path = os.path.join(icon_dir, fname)
            if os.path.isfile(path):
                return QtGui.QIcon(path)

        # ---- 2) PNG が無いときのフォールバック ----
        # 文字アイコン（⟳ / 🔁）を描く。とりあえずのしのぎ用。
        size = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_SmallIconSize)
        if size <= 0:
            size = 16

        pix = QtGui.QPixmap(size, size)
        pix.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        font = self.font()
        font.setPointSizeF(size * 0.8)
        p.setFont(font)

        p.setPen(QtGui.QColor(235, 235, 235))
        ch = "⟳" if kind == "reload" else "🔁"
        rect = QtCore.QRectF(0, 0, size, size)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, ch)
        p.end()

        return QtGui.QIcon(pix)

    def _mark_child_for_up(self):
        try:
            cur = getattr(self, "folder", "") or os.path.dirname(getattr(self, "image_path", ""))
            self._nav_last_child_basename = os.path.basename((cur or "").rstrip("\\/"))
            log_debug(f"[nav] mark child='{self._nav_last_child_basename}'")
        except Exception:
            self._nav_last_child_basename = ""

    def _nav_up(self):
        cur = getattr(self, "folder", "") or ""
        if not cur:
            return

        parent = vfs_parent(cur)
        # 親が無い or 自分と同じなら何もしない
        if not parent or self._norm_path(parent) == self._norm_path(cur):
            return

        # --- 履歴に「上へ」を push（進む側を切る & 連続重複を回避） ---
        if not hasattr(self, "_nav_history"):
            self._nav_history = []
            self._nav_pos = -1

        if 0 <= self._nav_pos < len(self._nav_history) - 1:
            self._nav_history = self._nav_history[:self._nav_pos + 1]

        cur_norm  = self._norm_path(parent)
        last_norm = self._norm_path(self._nav_history[-1]) if self._nav_history else None

        if last_norm != cur_norm:
            self._nav_history.append(parent)
            self._nav_pos = len(self._nav_history) - 1
            if getattr(self, "_nav_debug", False):
                log_debug(f"[nav] push(up) -> pos={self._nav_pos} path={parent}")
        else:
            self._nav_pos = len(self._nav_history) - 1
            if getattr(self, "_nav_debug", False):
                log_debug(f"[nav] skip-dup(up) -> pos={self._nav_pos} path={parent}")

        # --- 開くときは“復元モード”で（③の要件：元の選択を復元） ---
        self._mark_child_for_up()
        self.open_folder(parent, _from_history=True, _src="up")

    def _sync_thumb_selection(self):
        """現在の self.image_path がブラウザ一覧の何行目かを探して選択。"""
        if not getattr(self, "image_path", None):
            return
        if not getattr(self, "model", None):
            return

        path_now = norm_vpath(self.image_path)
        model = self.model
        target_row = -1
        for r in range(model.rowCount()):
            info = model.data(model.index(r,0), QtCore.Qt.ItemDataRole.UserRole)
            try:
                p = norm_vpath(info["path"])
            except Exception:
                continue
            if p == path_now:
                target_row = r
                break
        if target_row >= 0:
            self.listview.setCurrentIndex(model.index(target_row, 0))

    def on_thumb_double_clicked(self, index: QtCore.QModelIndex):
        if getattr(self, "_opening_folder", False):
            return
        timer = getattr(self, "_click_timer", None)
        try:
            if isinstance(timer, QtCore.QTimer) and timer.isActive():
                timer.stop()
        except Exception:
            pass
        self._pending_index = None
        if not index.isValid():
            return
        info = index.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        # 互換（dict / str 両対応）
        if isinstance(info, dict):
            path = info.get("path")
            is_dir = bool(info.get("is_dir"))
        else:
            path = info
            is_dir = vfs_is_dir(path)

        if not path:
            return

        # フォルダならその中へ
        if is_dir:
            self.open_folder(path, _src="dblclick_thumb")
            try:
                self._update_nav_buttons()
            except Exception:
                pass
            return

        # 画像ならその画像へ（ナビと同じ preserve ルート）
        norm = norm_vpath(path)
        tgt = next((i for i, p in enumerate(getattr(self, "image_list", []))
                    if norm_vpath(p) == norm), -1)

        if tgt >= 0:
            # 先に“次回だけUI温存”トークンを積む
            try:
                if hasattr(self, "_prepare_preserve_for_nav"):
                    self._prepare_preserve_for_nav()
            except Exception:
                pass

            self.current_index = tgt

            # ★load_image_by_index は呼ばずに open_image_from_path を使う
            self.open_image_from_path(path)

            try:
                self._sync_thumb_selection()
            except Exception:
                pass

    def _preview_from_thumb_index(self, index: QtCore.QModelIndex):
        """サムネ選択中の項目をメインプレビューに反映（画像=画像表示 / フォルダ=プレースホルダ）"""
        # 欠損画像処理中はここからの再入を抑止（行削除で currentChanged が飛んできても無視する）
        if getattr(self, "_handling_missing_image", False):
            return

        if not index.isValid():
            return

        info = index.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        if isinstance(info, dict):
            path = info.get("path")
            is_dir = bool(info.get("is_dir"))
        else:
            path = info
            try:
                is_dir = vfs_is_dir(path)
            except Exception:
                is_dir = False

        if not path:
            return

        # 正規化ヘルパ（_norm_path が無くても動くようフォールバック）
        def _N(p: str) -> str:
            try:
                return self._norm_path(p)
            except Exception:
                try:
                    return os.path.normcase(os.path.abspath(p or ""))
                except Exception:
                    return (p or "").lower()

        if is_dir:
            t0 = _dbg_time(f"preview_dir start: {path}")
            try:
                self._show_folder_placeholder(path, force=True)
            except Exception:
                pass
            _dbg_time(f"preview_dir end:   {path}", t0)

            # ▼ 左下のフルパス（フォルダパス表示＋OSアイコン）
            try:
                self.set_path_text(self._format_dir_path_text(path))
                self._update_path_icon_for_folder(path)
            except Exception:
                pass

            # ▼ 追加：中央ラベルはフォルダ名に（必要なら“枚数”など好みで）
            try:
                self._update_center_label_for_folder(path)
                # 例：枚数も出すなら → self.image_info_label.setText(f"{os.path.basename(path)}  —  {len(self.image_list)} 枚")
            except Exception:
                pass

            try:
                self.current_index = -1  # フォルダなので画像インデックスは無効
            except Exception:
                pass

            # ★ ここで進捗表示をニュートラルにリセット
            try:
                self._clear_progress_display()
            except Exception:
                pass

            return

        # 画像：同一ファイルなら何もしない（重複オープン抑止）
        try:
            cur = getattr(self, "image_path", "")
            if cur and _N(cur) == _N(path):
                return
        except Exception:
            pass

        # image_list 上のインデックスを同期（見つからなければ -1）
        tgt = -1
        try:
            for i, p in enumerate(getattr(self, "image_list", [])):
                if _N(p) == _N(path):
                    tgt = i
                    break
        except Exception:
            pass
        try:
            self.current_index = tgt
        except Exception:
            pass

        # 画像：同一ファイルなら何もしない（重複オープン抑止）
        try:
            cur = getattr(self, "image_path", "")
            if cur and _N(cur) == _N(path):
                return
        except Exception:
            pass

        # image_list 上のインデックスを同期（見つからなければ -1）
        tgt = -1
        try:
            for i, p in enumerate(getattr(self, "image_list", [])):
                if _N(p) == _N(path):
                    tgt = i
                    break
        except Exception:
            pass
        try:
            self.current_index = tgt
        except Exception:
            pass

        # 画像を開く（VFS対応版：zip:// も含めて一律 open_image_from_path に任せる）
        try:
            if hasattr(self, "_prepare_preserve_for_nav"):
                self._prepare_preserve_for_nav()
        except Exception:
            pass

        self._suspend_chain_clear = getattr(self, "_suspend_chain_clear", 0) + 1
        try:
            self.open_image_from_path(path)
        except Exception as e:
            log_debug("[preview] open_image_from_path error:", e)
            # 万一「実はフォルダ扱い」だった場合だけフォールダープレビューにフォールバック
            try:
                if vfs_is_dir(path):
                    self._show_folder_placeholder(path, force=True)
                    self.current_index = -1
                    return
            except Exception:
                pass
        finally:
            self._suspend_chain_clear -= 1

        # サムネ側の選択同期（失敗しても無視）
        try:
            self._sync_thumb_selection()
        except Exception:
            pass

    def navigate_from_gesture(self, direction, modifiers=None) -> None:
        mods = modifiers if modifiers is not None else QtWidgets.QApplication.keyboardModifiers()
        has_shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if has_shift:
            step = 1 if direction > 0 else -1
            if not self._jump_sibling_folder(step, require_images=False):
                QtWidgets.QApplication.beep()
            return
        if hasattr(self, "_prepare_preserve_for_nav"):
            self._prepare_preserve_for_nav()  # ★ 追加
        self._move_thumb_focus(1 if direction > 0 else -1)

    def _make_hq_scaled_pixmap(self, target_w: int, target_h: int) -> QtGui.QPixmap:
        """HQズーム用: PillowのLANCZOS(+軽いアンシャープ)で拡大してQPixmapに返す"""
        try:
            from PIL import ImageFilter, ImageOps
            src = self.image
            if src is None:
                return self.img_pixmap

            # EXIF回転を正しつつRGB(A)化
            img = ImageOps.exif_transpose(src)
            mode = "RGBA" if "A" in img.getbands() else "RGB"
            if img.mode != mode:
                img = img.convert(mode)

            sw, sh = img.width, img.height
            scale = min(target_w / float(sw), target_h / float(sh))
            new_w = max(1, int(round(sw * scale)))
            new_h = max(1, int(round(sh * scale)))

            img = img.resize((new_w, new_h), Image.LANCZOS)
            try:
                img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
            except Exception:
                pass

            qimg = ImageQt(img)
            return QtGui.QPixmap.fromImage(qimg.copy())
        except Exception:
            # フォールバック：Qtのスムーズ拡大
            return self.img_pixmap.scaled(
                target_w, target_h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )

    def show_image(self):
        if self.image is None:
            return

        # --- 1) 元画像→QPixmapは「画像が変わった時だけ」 ---
        if self._base_pixmap_dirty or self.img_pixmap is None:
            from PIL import ImageOps
            # EXIF回転を補正してから、Aチャンネル有無で RGB/RGBA を選択
            img = ImageOps.exif_transpose(self.image)
            mode = "RGBA" if "A" in img.getbands() else "RGB"
            if img.mode != mode:
                img = img.convert(mode)
            qimg = ImageQt(img)
            self.img_pixmap = QtGui.QPixmap.fromImage(qimg.copy())
            self._base_pixmap_dirty = False
            self._scaled_pixmap = None
            self._scaled_key = None

        label_w, label_h = self.label.width(), self.label.height()
        img_w, img_h = self.img_pixmap.width(), self.img_pixmap.height()

        if self.zoom_scale == 1.0 or self.base_display_width is None or self.base_display_height is None:
            # 初期フィット時のみスケール（滑らかでOK、頻度が低い）
            if img_w <= label_w and img_h <= label_h:
                scaled = self.img_pixmap
            else:
                scaled = self.img_pixmap.scaled(
                    label_w, label_h,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            self.base_display_width  = scaled.width()
            self.base_display_height = scaled.height()

            # ★ ズーム=1.0のときの拡大済みキャッシュとして保持
            self._scaled_pixmap = scaled
            self._scaled_key = (int(self.img_pixmap.cacheKey()),
                                self.base_display_width, self.base_display_height,
                                1.0, False)  # ← hqフラグを含める

            # 表示は全域
            self.label._view_rect_scaled = QtCore.QRect(0, 0, scaled.width(), scaled.height())
            display_pixmap = scaled

        else:
            # --- 2) ズーム時：拡大済みキャッシュを使い回す ---
            target_w = int(self.base_display_width  * self.zoom_scale)
            target_h = int(self.base_display_height * self.zoom_scale)
            target_w = max(1, target_w); target_h = max(1, target_h)

            need_rebuild = True
            base_key = int(self.img_pixmap.cacheKey())
            hq = bool(getattr(self, "hq_zoom", False))
            key_now = (base_key, self.base_display_width, self.base_display_height,
                    float(self.zoom_scale), hq)
            if self._scaled_pixmap is not None and self._scaled_key == key_now:
                need_rebuild = False

            if need_rebuild:
                if hq:
                    # HQズーム（Pillow LANCZOS + 軽いアンシャープ）
                    self._scaled_pixmap = self._make_hq_scaled_pixmap(target_w, target_h)
                else:
                    # 従来：Qtのスムーズ拡大
                    self._scaled_pixmap = self.img_pixmap.scaled(
                        target_w, target_h,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    )
            self._scaled_key = key_now
            scaled = self._scaled_pixmap

            # ここからは「表示領域の切り出し」だけ（安い）
            offset_x = getattr(self.label, "_pan_offset_x", 0)
            offset_y = getattr(self.label, "_pan_offset_y", 0)
            center_x = scaled.width() // 2 + offset_x
            center_y = scaled.height() // 2 + offset_y
            lw2, lh2 = label_w // 2, label_h // 2

            crop_rect = QtCore.QRect(center_x - lw2, center_y - lh2, label_w, label_h)
            crop_rect = crop_rect.intersected(scaled.rect())
            self.label._view_rect_scaled = QtCore.QRect(crop_rect)

            # ★ ここでは「copy」だけ。再スケールはしない
            display_pixmap = scaled.copy(crop_rect)

        self.label.setPixmap(display_pixmap)
        # センタリング用のオフセットを更新（既存ロジック流用）
        lw, lh = self.label.width(), self.label.height()
        pw, ph = display_pixmap.width(), display_pixmap.height()
        self.label._init_offset_x = (lw - pw) // 2 if lw > pw else 0
        self.label._init_offset_y = (lh - ph) // 2 if lh > ph else 0

        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.update()
        self.zoom_label.show_zoom(self.zoom_scale)

        # デバッグ
        vr = getattr(self.label, "_view_rect_scaled", None)
        if DEBUG_VIEW_RECT and vr:
            log_debug(f"[VIEW_RECT] x={vr.x()} y={vr.y()} w={vr.width()} h={vr.height()}")
        
    def update_preview(self, crop_rect=None):
        try:
            # ベース確保（画像が切り替わったら自動更新）
            self._ensure_preview_base()
            if (not crop_rect or
                self._preview_base_pixmap is None or
                self._preview_base_pixmap.isNull()):
                self._set_preview_placeholder()
                return

            sx = float(self._preview_sx)
            sy = float(self._preview_sy)

            # 画像座標 → ベース座標（外側丸めで欠け防止）
            x1 = (crop_rect.left()) * sx
            y1 = (crop_rect.top())  * sy
            x2 = (crop_rect.left() + crop_rect.width())  * sx
            y2 = (crop_rect.top()  + crop_rect.height()) * sy
            r  = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)).toAlignedRect()
            r  = r.intersected(self._preview_base_pixmap.rect())
            if r.isEmpty():
                self._set_preview_placeholder()
                return

            # プレビューラベル（正方形）にフィット
            PREVIEW_MAX = max(1, min(512,
                getattr(self.preview_label, "width",  lambda:512)() or 512,
                getattr(self.preview_label, "height", lambda:512)() or 512
            ))
            sub = self._preview_base_pixmap.copy(r)
            fitted = sub.scaled(
                PREVIEW_MAX, PREVIEW_MAX,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )

            canvas = QtGui.QPixmap(PREVIEW_MAX, PREVIEW_MAX)
            canvas.fill(getattr(self, "preview_bg_color", QtGui.QColor(255, 255, 255)))
            p = QtGui.QPainter(canvas)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True) 
            p.drawPixmap((PREVIEW_MAX - fitted.width()) // 2,
                        (PREVIEW_MAX - fitted.height()) // 2,
                        fitted)
            p.end()

            self.preview_label.setPixmap(canvas)

        except Exception:
            self._set_preview_placeholder()
            
    def _ensure_preview_base(self):
        if not getattr(self, "img_pixmap", None):
            self._preview_base_pixmap = None
            self._preview_src_size = None
            self._preview_src_key = None
            return

        src_w = self.img_pixmap.width()
        src_h = self.img_pixmap.height()
        key   = int(self.img_pixmap.cacheKey())  # ←今の pixmap の一意キー

        # ★同じ解像度でも key が違えば作り直す
        if self._preview_base_pixmap is not None and getattr(self, "_preview_src_key", None) == key:
            return

        TARGET = 1600
        scale = min(1.0, TARGET / max(src_w, src_h))
        if scale < 1.0:
            pw = max(1, int(round(src_w * scale)))
            ph = max(1, int(round(src_h * scale)))
            self._preview_base_pixmap = self.img_pixmap.scaled(
                pw, ph,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        else:
            self._preview_base_pixmap = QtGui.QPixmap(self.img_pixmap)

        self._preview_sx = self._preview_base_pixmap.width()  / src_w
        self._preview_sy = self._preview_base_pixmap.height() / src_h
        self._preview_src_size = (src_w, src_h)
        self._preview_src_key  = key            # ←記録

    def _set_preview_placeholder(self):
        side = 384 if getattr(self, "_preview_initial_cap", False) else getattr(self.preview_label, "_base_side", 512)
        pm = QtGui.QPixmap(side, side)
        pm.fill(getattr(self, "preview_bg_color", QtGui.QColor(255, 255, 255)))
        self.preview_label.setPixmap(pm)
    
    def _apply_preview_bg_to_label(self):
        """プレビュー枠（白だった所）を現在選択色で塗る"""
        c = getattr(self, "preview_bg_color", QtGui.QColor("#ffffff"))
        name = c.name(QtGui.QColor.NameFormat.HexArgb) if c.alpha() < 255 else c.name()
        self.preview_label.setStyleSheet(
            f"background: {name}; border: none; border-radius: 0px;"
        )
    
    def _load_custom_colors(self):
        """QColorDialog の Custom colors を INI から復元"""
        for i in range(16):
            val = self.settings.value(f"custom_colors/{i}")
            if isinstance(val, str) and val:
                col = QtGui.QColor(val)
                if col.isValid():
                    QtWidgets.QColorDialog.setCustomColor(i, col.rgb())

    def _save_custom_colors(self):
        for i in range(16):
            val = QtWidgets.QColorDialog.customColor(i)  # int または QColor が返る
            if isinstance(val, int):
                col = QtGui.QColor.fromRgba(val)
            else:
                # PyQt6 環境によっては QColor が返るのでそのまま包む
                col = QtGui.QColor(val)

            if not col.isValid():
                continue

            # 透過も保持したいので HexArgb で保存
            self.settings.setValue(
                f"custom_colors/{i}",
                col.name(QtGui.QColor.NameFormat.HexArgb)
            )
        self.settings.sync()

    def on_pick_preview_bg(self):
        c = QtWidgets.QColorDialog.getColor(self.preview_bg_color, self, "プレビュー領域の背景色",
            options=QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
                    QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid(): self.set_preview_bg_color(c)

    def _apply_view_bg(self):
        """画像表示領域（オレンジ枠の内側）の背景色を反映"""
        c = getattr(self, "view_bg_color", QtGui.QColor("#191919"))
        name = c.name(QtGui.QColor.NameFormat.HexArgb) if c.alpha() < 255 else c.name()
        # 内側だけこの色で塗る（境界線は従来どおり）
        self.label.setStyleSheet(
            f"background: {name}; border: 2px solid #f28524; border-radius: 12px;"
        )

    def on_pick_view_bg(self):
        c = QtWidgets.QColorDialog.getColor(self.view_bg_color, self, "画像表示領域の背景色",
            options=QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel |
                    QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid(): self.set_view_bg_color(c)

    def zoom_in(self):
        # 画像・基準サイズが未確定なら何もしない
        if self.image is None or self.label.pixmap() is None:
            return
        if self.base_display_width is None or self.base_display_height is None:
            # 現在の表示から基準を初期化（初期化前のホイール対策）
            pm = self.label.pixmap()
            self.base_display_width  = pm.width()
            self.base_display_height = pm.height()

        # 現在のズーム値 old → 新しいズーム値 new
        old = self.zoom_scale
        new = 0.25 if abs(old - 0.10) < 1e-6 else min(old + 0.25, 8.0)

        # パン量をズーム倍率比で縮小（相対位置を維持）
        ratio = new / old
        self.label._pan_offset_x = int(self.label._pan_offset_x * ratio)
        self.label._pan_offset_y = int(self.label._pan_offset_y * ratio)

        # ズーム値更新＆再描画
        self.zoom_scale = round(new, 2)
        self.show_image()

    def zoom_out(self):
        # 画像・基準サイズが未確定なら何もしない
        if self.image is None or self.label.pixmap() is None:
            return
        if self.base_display_width is None or self.base_display_height is None:
            pm = self.label.pixmap()
            self.base_display_width  = pm.width()
            self.base_display_height = pm.height()
        
        # 現在のズーム値 old → 新しいズーム値 new
        old = self.zoom_scale
        new = max(old - 0.25, 0.10)

        # パン量をズーム倍率比で縮小（相対位置を維持）
        ratio = new / old
        self.label._pan_offset_x = int(self.label._pan_offset_x * ratio)
        self.label._pan_offset_y = int(self.label._pan_offset_y * ratio)

        # ズーム値更新＆再描画
        self.zoom_scale = round(new, 2)

        # 縮小後の pixmap サイズ
        pm_w = int(self.base_display_width * self.zoom_scale)
        pm_h = int(self.base_display_height * self.zoom_scale)
        # QLabel のサイズ
        lw, lh = self.label.width(), self.label.height()
        # --- オフセットを、必ず有効範囲内にクランプする ---
        max_x = max(0, (pm_w - lw) // 2)
        max_y = max(0, (pm_h - lh) // 2)
        # scaled が大きいときは pan_offset をはみ出さないように
        self.label._pan_offset_x = max(-max_x, min(self.label._pan_offset_x, max_x))
        self.label._pan_offset_y = max(-max_y, min(self.label._pan_offset_y, max_y))
        # scaled が小さいときは自動で真ん中に寄せる
        if pm_w < lw:
            self.label._pan_offset_x = 0
        if pm_h < lh:
            self.label._pan_offset_y = 0

        self.show_image()

    def _record_batch_transform(self, op: str):
        """
        一括切り取り用に、画像変形の履歴を記録する。
        op: 'flip_h' | 'flip_v' | 'rot_left_90' | 'rot_right_90'
        """
        ops = getattr(self, "_batch_transform_ops", None)
        if ops is None:
            ops = []
            self._batch_transform_ops = ops
        ops.append(op)

    def _get_batch_transform_ops(self):
        """記録済みの変形履歴を返す（無ければ空リスト）。"""
        return list(getattr(self, "_batch_transform_ops", []) or [])

    def on_flip_horizontal(self):
        """画像を左右反転。選択中の矩形も一緒に左右反転させる。"""
        if getattr(self, "image", None) is None:
            QtWidgets.QApplication.beep()
            return

        # 1) 画像を左右反転
        try:
            self.image = ImageOps.mirror(self.image)
        except Exception as e:
            log_debug("[flip] horizontal flip failed:", e)
            QtWidgets.QApplication.beep()
            return
        
        self._record_batch_transform("flip_h")

        img_w = self.image.width

        # 画像座標の QRect を左右反転させるヘルパ
        def mirror_rect(rect):
            if rect is None:
                return None
            if hasattr(rect, "isNull") and rect.isNull():
                return None

            x = int(rect.left())
            y = int(rect.top())
            w = max(0, int(rect.width()))
            h = max(0, int(rect.height()))
            if w <= 0 or h <= 0:
                return None

            # 左右反転：新しい left = W - (old_left + width)
            new_left = img_w - (x + w)

            # ★ 画像内クランプはしない（はみ出しを維持）
            return QtCore.QRect(new_left, y, w, h)

        # 2) 画像座標の矩形を左右反転
        fixed_rect = mirror_rect(getattr(self.label, "fixed_crop_rect_img", None))
        drag_rect  = mirror_rect(getattr(self.label, "drag_rect_img", None))
        crop_rect  = mirror_rect(getattr(self, "_crop_rect_img", None))

        if fixed_rect is not None:
            # 固定枠モードの矩形
            self.label.fixed_crop_rect_img = QtCore.QRect(fixed_rect)
            self._crop_rect_img = QtCore.QRect(fixed_rect)
        elif drag_rect is not None:
            # 通常ドラッグの矩形
            self.label.drag_rect_img = QtCore.QRect(drag_rect)
            self._crop_rect_img = QtCore.QRect(drag_rect)
        else:
            # 念のため _crop_rect_img だけ生きているケースも反転
            self._crop_rect_img = QtCore.QRect(crop_rect) if crop_rect is not None else None

        # 3) ピクスマップ再生成 & 再描画
        self._base_pixmap_dirty = True
        self._scaled_pixmap = None
        self._scaled_key = None
        self.show_image()

        # 固定枠モードなら、UIを最新座標に同期
        try:
            self._sync_fixed_ui_after_image_change()
        except Exception:
            pass

        # 4) ラベル座標の矩形（アクションパネル位置など用）も更新
        rect_label = None
        if self.label.fixed_crop_mode and getattr(self.label, "fixed_crop_rect_img", None) is not None:
            rect_label = self.label._fixed_rect_labelcoords()
        elif getattr(self.label, "drag_rect_img", None) is not None:
            rect_label = self.label._drag_rect_labelcoords()

        if rect_label is not None:
            self._crop_rect = QtCore.QRect(rect_label)

        # 5) 解像度ラベル＆プレビュー更新
        if getattr(self, "_crop_rect_img", None) is not None:
            self.update_crop_size_label(self._crop_rect_img, img_space=True)
            self._schedule_preview(self._crop_rect_img)
        else:
            # 範囲が無ければプレースホルダに戻す
            self._set_preview_placeholder()

    def on_flip_vertical(self):
        """画像を上下反転。選択中の矩形も一緒に上下反転させる。"""
        if getattr(self, "image", None) is None:
            QtWidgets.QApplication.beep()
            return

        # 1) 画像を上下反転
        try:
            self.image = ImageOps.flip(self.image)
        except Exception as e:
            log_debug("[flip] vertical flip failed:", e)
            QtWidgets.QApplication.beep()
            return
        
        self._record_batch_transform("flip_v")

        img_h = self.image.height

        # 画像座標の QRect を上下反転させるヘルパ
        def flip_rect(rect):
            if rect is None:
                return None
            if hasattr(rect, "isNull") and rect.isNull():
                return None

            x = int(rect.left())
            y = int(rect.top())
            w = max(0, int(rect.width()))
            h = max(0, int(rect.height()))
            if w <= 0 or h <= 0:
                return None

            # 上下反転：新しい top = H - (old_top + height)
            new_top = img_h - (y + h)

            # ★ 画像内クランプはしない（はみ出しを維持）
            return QtCore.QRect(x, new_top, w, h)

        # 2) 画像座標の矩形を上下反転
        fixed_rect = flip_rect(getattr(self.label, "fixed_crop_rect_img", None))
        drag_rect  = flip_rect(getattr(self.label, "drag_rect_img", None))
        crop_rect  = flip_rect(getattr(self, "_crop_rect_img", None))

        if fixed_rect is not None:
            # 固定枠モードの矩形
            self.label.fixed_crop_rect_img = QtCore.QRect(fixed_rect)
            self._crop_rect_img = QtCore.QRect(fixed_rect)
        elif drag_rect is not None:
            # 通常ドラッグの矩形
            self.label.drag_rect_img = QtCore.QRect(drag_rect)
            self._crop_rect_img = QtCore.QRect(drag_rect)
        else:
            # 念のため _crop_rect_img だけ生きているケースも反転
            self._crop_rect_img = QtCore.QRect(crop_rect) if crop_rect is not None else None

        # 3) ピクスマップ再生成 & 再描画
        self._base_pixmap_dirty = True
        self._scaled_pixmap = None
        self._scaled_key = None
        self.show_image()

        # 固定枠モードなら、UIを最新座標に同期
        try:
            self._sync_fixed_ui_after_image_change()
        except Exception:
            pass

        # 4) ラベル座標の矩形も更新
        rect_label = None
        if self.label.fixed_crop_mode and getattr(self.label, "fixed_crop_rect_img", None) is not None:
            rect_label = self.label._fixed_rect_labelcoords()
        elif getattr(self.label, "drag_rect_img", None) is not None:
            rect_label = self.label._drag_rect_labelcoords()

        if rect_label is not None:
            self._crop_rect = QtCore.QRect(rect_label)
        else:
            self._crop_rect = None

        # 5) 解像度ラベル＆プレビュー更新
        if getattr(self, "_crop_rect_img", None) is not None:
            self.update_crop_size_label(self._crop_rect_img, img_space=True)
            self._schedule_preview(self._crop_rect_img)
        else:
            self._set_preview_placeholder()

    def _rotate_90_common(self, op_const):
        """
        90度回転の共通処理。
        op_const は Image.ROTATE_90（左） / Image.ROTATE_270（右） を想定。
        画像と一緒に選択中の矩形も回転させる。
        """
        if getattr(self, "image", None) is None:
            QtWidgets.QApplication.beep()
            return

        from PIL import Image

        if op_const == Image.ROTATE_90:
            direction = "left"
        elif op_const == Image.ROTATE_270:
            direction = "right"
        else:
            # 想定外
            log_debug("[rotate] unsupported op_const:", op_const)
            return

        try:
            self._record_batch_transform(
                "rot_left_90" if direction == "left" else "rot_right_90"
            )
        except Exception:
            pass

        # --- 元画像サイズ & 元の矩形を取得 ---
        old_w, old_h = self.image.size

        fixed_rect = getattr(self.label, "fixed_crop_rect_img", None)
        drag_rect  = getattr(self.label, "drag_rect_img", None)
        crop_rect  = getattr(self, "_crop_rect_img", None)

        def rot_rect_90(rect):
            """画像座標の QRect を90度回転させる（軸に沿ったまま）。"""
            if rect is None:
                return None
            try:
                r = QtCore.QRect(rect).normalized()
            except Exception:
                return None

            x = int(r.x())
            y = int(r.y())
            w = max(0, int(r.width()))
            h = max(0, int(r.height()))
            if w <= 0 or h <= 0:
                return None

            if direction == "left":
                # 反時計回り90°
                # 新しい左上: x' = y, y' = old_w - (x + w)
                new_left = y
                new_top  = old_w - (x + w)
                new_w, new_h = h, w
            else:
                # 時計回り90°
                # 新しい左上: x' = old_h - (y + h), y' = x
                new_left = old_h - (y + h)
                new_top  = x
                new_w, new_h = h, w

            return QtCore.QRect(new_left, new_top, new_w, new_h)

        new_fixed = rot_rect_90(fixed_rect)
        new_drag  = rot_rect_90(drag_rect)
        new_crop  = rot_rect_90(crop_rect)

        # --- 画像本体を回転 ---
        try:
            self.image = self.image.transpose(op_const)
        except Exception as e:
            log_debug("[rotate] 90deg rotate failed:", e)
            QtWidgets.QApplication.beep()
            return

        # --- 回転後の矩形を書き戻し ---
        if new_fixed is not None:
            self.label.fixed_crop_rect_img = QtCore.QRect(new_fixed)
            self.label.drag_rect_img = None
            self._crop_rect_img = QtCore.QRect(new_fixed)
        elif new_drag is not None:
            self.label.drag_rect_img = QtCore.QRect(new_drag)
            self.label.fixed_crop_rect_img = None
            self._crop_rect_img = QtCore.QRect(new_drag)
        else:
            self.label.fixed_crop_rect_img = None
            self.label.drag_rect_img = None
            self._crop_rect_img = QtCore.QRect(new_crop) if new_crop is not None else None

        # --- ピクスマップ再生成 & 再描画 ---
        self._base_pixmap_dirty = True
        self._scaled_pixmap = None
        self._scaled_key = None

        # ★★ 追加ポイント ★★
        # 90度回転すると画像の縦横が入れ替わるので、
        # ズームの基準サイズも入れ替えておく。
        if self.base_display_width is not None and self.base_display_height is not None:
            self.base_display_width, self.base_display_height = (
                self.base_display_height,
                self.base_display_width,
            )

        self.show_image()

        # 固定枠モードならUI同期
        try:
            self._sync_fixed_ui_after_image_change()
        except Exception:
            pass

        # --- ラベル座標の矩形も更新 ---
        rect_label = None
        if self.label.fixed_crop_mode and getattr(self.label, "fixed_crop_rect_img", None) is not None:
            rect_label = self.label._fixed_rect_labelcoords()
        elif getattr(self.label, "drag_rect_img", None) is not None:
            rect_label = self.label._drag_rect_labelcoords()

        if rect_label is not None:
            self._crop_rect = QtCore.QRect(rect_label)
        else:
            self._crop_rect = None

        # --- 解像度ラベル & プレビュー更新 ---
        if getattr(self, "_crop_rect_img", None) is not None:
            self.update_crop_size_label(self._crop_rect_img, img_space=True)
            self._schedule_preview(self._crop_rect_img)
        else:
            self._set_preview_placeholder()

    def on_rotate_left_90(self):
        """左に90度回転（反時計回り）"""
        from PIL import Image
        self._rotate_90_common(Image.ROTATE_90)

    def on_rotate_right_90(self):
        """右に90度回転（時計回り）"""
        from PIL import Image
        self._rotate_90_common(Image.ROTATE_270)

    def showEvent(self, e):
        super().showEvent(e)
        self._window_shown = True
        if getattr(self, "_first_shown_size", None) is None:
            self._first_shown_size = self.size()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # 表示後に「サイズが変わった」ことを検出した時だけ解除
        if self._preview_initial_cap and self._window_shown:
            if self._first_shown_size and self.size() != self._first_shown_size:
                self.preview_area.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
                self.preview_label.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
                self._preview_initial_cap = False

        # ← サブパネル／カラーパレット行のキャップも解除
        if hasattr(self, "sub_panel"):
            self.sub_panel.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
        if hasattr(self, "color_row_widget"):
            self.color_row_widget.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)

        self.show_image()
        self._sync_fixed_ui_after_image_change()
        self.update_progress_alignment()

        if getattr(self, "_crop_rect_img", None):
            self.update_crop_size_label(self._crop_rect_img, img_space=True)
            QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect_img))

    def on_crop(self, rect, mouse_pos):
        # ★ 新しい通常矩形が確定したので、パネル追従フラグを必ずリセット
        #    （Nudge が出ている/いない、ActionPanel だけ、に関係なく復活させる）
        self._action_panel_detached = False
        self._nudge_detached = False

        # rect はラベル座標（UI 用）。表示位置用に保持
        self._crop_rect = rect
        self.show_action_panel(rect, mouse_pos)
        
        # rect はラベル座標（UI 用）。表示位置用に保持
        self._crop_rect = rect
        self.show_action_panel(rect, mouse_pos)

        # ★ ここが重要：画像座標は round-trip せず drag_rect_img をそのまま使う
        img_rect = getattr(self.label, "drag_rect_img", None)
        if img_rect is not None:
            self._crop_rect_img = img_rect
        else:
            # フォールバック（念のため）
            gx1, gy1 = self.label.label_to_image_coords(rect.left(), rect.top())
            gx2, gy2 = self.label.label_to_image_coords(rect.left() + rect.width(),
                                                        rect.top() + rect.height())
            self._crop_rect_img = QtCore.QRect(min(gx1, gx2), min(gy1, gy2),
                                               abs(gx2 - gx1), abs(gy2 - gy1))

        # 以降は“画像座標”の矩形で統一して使う
        self.update_crop_size_label(self._crop_rect_img, img_space=True)
        self._schedule_preview(self._crop_rect_img)

    def on_fixed_crop_move(self, rect, mouse_pos):
        # ★ 固定矩形をマウスで動かしているので、追従フラグを復活させる
        self._action_panel_detached = False
        self._nudge_detached = False

        # rect はラベル座標（パネル位置などUI用に保持）
        self._crop_rect = rect

        # 画像座標の矩形をソース・オブ・トゥルースとして保持
        img_rect = getattr(self.label, "fixed_crop_rect_img", None)
        if img_rect is None:
            return
        self._crop_rect_img = img_rect

        # パネル表示（位置決めはラベル座標の rect でOK）
        # ★ 中ボタントグルでユーザーが「今はパネル要らない」としている間は勝手に出さない
        if not getattr(self, "_panel_hidden_by_user", False):
            self.show_action_panel(rect, mouse_pos)

        # 解像度ラベルは画像座標でそのまま（丸め往復を避ける）
        self.update_crop_size_label(self._crop_rect_img, img_space=True)

        # プレビューも画像座標の矩形をそのまま渡す
        self._schedule_preview(self._crop_rect_img)
        self._ensure_aspect_base_from_current_rect()

    def show_action_panel(self, rect, mouse_pos):
        if not rect or (hasattr(rect, "isNull") and rect.isNull()):
            return

        is_fixed = (
            getattr(self.label, "fixed_crop_mode", False) and
            getattr(self.label, "fixed_crop_rect_img", None) is not None
        )

        # 既存パネルがあれば閉じる
        if self._action_panel:
            try:
                self._action_panel.close()
            finally:
                self._action_panel = None
                # 埋め込み系は廃止済みなので触らない

        pos = self._compute_action_panel_pos(rect, is_fixed)
        on_pin = self.unfix_fixed_mode if is_fixed else self.pin_current_rect

        self._action_panel = ActionPanel(
            parent=self.label,
            pos=pos,
            on_save=self.do_crop_save,
            on_pin=on_pin,
            #on_cancel=self.cancel_crop,
            on_adjust=self.on_click_adjust,
            on_cancel=self.on_action_cancel,
            is_fixed=is_fixed,
        )

        # 32/64pxモードのときは調整ボタンを無効化
        is_multi = bool(getattr(self, "multiple_lock_enabled", False))
        self._action_panel.enable_adjust(not is_multi)

        # 調整モード表示に同期
        self._action_panel.set_adjusting(bool(getattr(self.label, "adjust_mode", False)))

        # 実寸確定後の最終配置
        self._action_panel.move(self._compute_action_panel_pos(rect, is_fixed))

        # 念のため：ラベルにフォーカスを戻す（矢印キーを確実に拾わせる）
        try:
            self.label.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

        # ▼▼ ここを“統一ロジック”に差し替え ▼▼
        if getattr(self.label, "adjust_mode", False):
            # 左ドラッグ中は出さない（従来仕様）
            if not (QtWidgets.QApplication.mouseButtons() & QtCore.Qt.MouseButton.LeftButton):
                # 位置決めも含めて必ずこの入口を使う
                self.open_nudge_overlay()
        else:
            # 調整OFFなのに出ていたら、位置だけ合わせ直す or 閉じる好みで。
            ov = getattr(self, "_nudge_overlay", None)
            if ov and ov.isVisible():
                # 位置だけ合わせ直したい場合はこれでOK（閉じたい場合は ov.close() に置き換え）
                self._position_nudge_overlay_above_action(
                    ov, gap=getattr(self, "nudge_gap_px", 4)
                )

    def open_nudge_overlay(self):
        """微調整パネルを ActionPanel の少し上に出す。既存があれば再利用してカウンタ維持。"""
        dlg = getattr(self, "_nudge_overlay", None)

        # 既存があればそれを“そのまま”再表示（作り直さない）
        if dlg is not None:
            try:
                gap = getattr(self, "nudge_gap_px", 4)
                # ActionPanel の直上に寄せ直す
                self._position_nudge_overlay_above_action(dlg, gap)
            except Exception:
                pass
            dlg.show()
            dlg.raise_()
            return

        # ここに来るのは「まだ一度も作っていない」場合だけ
        dlg = MovableNudgePanel(self, self.nudge_edge)
        self._nudge_overlay = dlg

        # 比率固定＆基準サイズの同期
        if hasattr(dlg, "set_aspect_button_state"):
            dlg.set_aspect_button_state(bool(getattr(self.label, "_aspect_lock", False)))
        base_wh = getattr(self.label, "_aspect_base_wh", None)
        if hasattr(dlg, "update_aspect_base"):
            dlg.update_aspect_base(base_wh)

        # 閉じられたときだけ参照クリア（hide のときはクリアしない）
        dlg.destroyed.connect(lambda *_: setattr(self, "_nudge_overlay", None))

        # 初回の位置決め→表示
        gap = getattr(self, "nudge_gap_px", 4)
        QtCore.QTimer.singleShot(0, lambda d=dlg, g=gap: self._position_nudge_overlay_above_action(d, g))
        dlg.show(); dlg.raise_()
        def close_nudge_overlay(self):
            for name in ("_nudge_overlay",):
                w = getattr(self, name, None)
                if w is not None:
                    try: w.close()
                    except Exception: pass
                    finally: setattr(self, name, None)

    def _position_nudge_overlay(self, dlg, gap=8):
        panel = getattr(self, "_action_panel", None)
        if panel and panel.isVisible():
            tl = panel.mapToGlobal(QtCore.QPoint(0, 0))
            x = tl.x() + (panel.width() - dlg.width()) // 2
            y = tl.y() - dlg.height() - gap
        else:
            # パネルが無ければ画像ラベル上部に出す
            base = self.label.mapToGlobal(QtCore.QPoint(self.label.width()//2, 40))
            x = base.x() - dlg.width()//2
            y = base.y() - 20

        # 画面内にクランプ
        scr = QtWidgets.QApplication.screenAt(QtCore.QPoint(x, y)) or QtWidgets.QApplication.primaryScreen()
        if scr:
            r = scr.availableGeometry()
            x = max(r.left()+4, min(x, r.right()-dlg.width()-4))
            y = max(r.top()+4,  min(y, r.bottom()-dlg.height()-4))

        dlg.move(x, y)

    # 画面の使用可能領域を取る（親ウィンドウのあるスクリーン優先）
    def _avail_screen_rect(self) -> QtCore.QRect:
        w = self.windowHandle()
        scr = (w.screen() if w else None) or QtGui.QGuiApplication.primaryScreen()
        return scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080)
   

        # === CropperApp のメソッドとして ===
    def set_adjust_mode(self, on: bool) -> None:
        """微調整モードのON/OFF。連続移動の“粘着”状態クリアもここで制御する。"""
        self._adjust_mode = bool(on)

        # Label へ伝播
        try:
            if hasattr(self, "label") and hasattr(self.label, "set_adjust_mode"):
                self.label.set_adjust_mode(self._adjust_mode)
        except Exception:
            pass

        # アクションパネルの表示状態を同期（ボタンの点灯など）
        p = getattr(self, "_action_panel", None)
        if p:
            try:
                p.set_adjusting(self._adjust_mode)
            except Exception:
                pass

        # ==== ★ ON にする時の“粘着プリセット” ====
        if self._adjust_mode:
            try:
                lbl = getattr(self, "label", None)
                fixed_on = bool(lbl and getattr(lbl, "fixed_crop_mode", False))
                if not fixed_on:
                    # 通常矩形は毎回消す(no_rect)／adjustだけ維持
                    self._nav_chain_state = {
                        "adjust": True,
                        "no_rect": True,
                        "aspect_lock": bool(getattr(lbl, "_aspect_lock", False)),
                        "aspect_base": getattr(lbl, "_aspect_base_wh", None),
                        "panel_visible": False,
                        "nudge": False,
                    }
            except Exception:
                pass

        # ---- OFF にする時の後処理 ----
        if not self._adjust_mode:
            # 1) 比率固定はOFFに戻す
            try:
                self.set_aspect_lock(False)
            except Exception:
                try:
                    setattr(self.label, "_aspect_lock", False)
                except Exception:
                    pass

            # 2) 固定枠モードなら、現在の固定矩形からサイズを確定
            try:
                if getattr(self.label, "fixed_crop_mode", False):
                    if hasattr(self.label, "_sync_fixed_size_from_rect"):
                        self.label._sync_fixed_size_from_rect()
                    else:
                        r = getattr(self.label, "fixed_crop_rect_img", None)
                        if r is not None and r.width() > 0 and r.height() > 0:
                            self.label.fixed_crop_size = (int(r.width()), int(r.height()))
            except Exception:
                pass

            # 3) “粘着”トークンを必要なら解放（内部OFF中は解放しない）
            try:
                lbl = getattr(self, "label", None)
                fixed_on = bool(lbl and getattr(lbl, "fixed_crop_mode", False))
                if (not fixed_on) and (getattr(self, "_suspend_chain_clear", 0) <= 0):
                    self._nav_chain_state = None
            except Exception:
                if getattr(self, "_suspend_chain_clear", 0) <= 0:
                    self._nav_chain_state = None

        # UIの最終整合
        try:
            self.ensure_nudge_visibility()
        except Exception:
            pass

        # デバッグログ（任意）
        try:
            placeholder = bool(getattr(self, "_placeholder_active", False) or getattr(self, "current_index", -1) < 0)
            internal = getattr(self, "_suspend_chain_clear", 0) > 0
            src = "USER" if (not internal and not placeholder) else ("PLACEHOLDER" if placeholder else "INTERNAL")
            log_debug(f"[adjust] mode={'ON' if self._adjust_mode else 'OFF'} "
                f"fixed={getattr(self.label, 'fixed_crop_mode', None)} "
                f"chain={'SET' if getattr(self, '_nav_chain_state', None) else 'None'} "
                f"susp={getattr(self, '_suspend_chain_clear', 0)} "
                f"src={src}")
        except Exception:
            pass

    def on_click_adjust(self):
        self.set_adjust_mode(not bool(getattr(self, "_adjust_mode", False)))

    def on_adjust_pressed(self):
        # 互換用ラッパー：内部の統一トグルを呼ぶだけ
        self.on_click_adjust()

    def set_aspect_lock(self, on: bool) -> None:
        on = bool(on)

        # --- ラベル側のフラグ＆基準値 ---
        self.label._aspect_lock = on
        base = None

        if on:
            # いまの矩形（固定→ドラッグ→最後に確定した）から基準を決める
            r = None
            if getattr(self.label, "fixed_crop_mode", False) and getattr(self.label, "fixed_crop_rect_img", None):
                r = self.label.fixed_crop_rect_img
            elif getattr(self.label, "drag_rect_img", None):
                r = self.label.drag_rect_img
            elif getattr(self, "_crop_rect_img", None):
                r = self._crop_rect_img

            if r is not None and r.width() > 0 and r.height() > 0:
                base = (int(r.width()), int(r.height()))
                self.label._aspect_base_wh = base
                try:
                    self.label._aspect_ratio = base[0] / base[1]
                except Exception:
                    self.label._aspect_ratio = None
            else:
                # 矩形が無い/0サイズならクリア
                self.label._aspect_base_wh = None
                self.label._aspect_ratio   = None
        else:
            # OFF時は必ずクリア
            self.label._aspect_base_wh = None
            self.label._aspect_ratio   = None

        # --- NudgePanel へ同期（独立フローティング版だけを更新） ---
        panel = getattr(self, "_nudge_panel", None)
        if panel:
            # ボタンの有効/無効・トグル表示
            if hasattr(panel, "set_aspect_button_state"):
                try:
                    panel.set_aspect_button_state(on)  # ONなら◁▷を無効化＋ボタン表示を同期
                except Exception:
                    pass
            else:
                # 互換: 明示的に◁▷の有効/無効だけ切り替える
                try:
                    panel.set_nudge_enabled(not on)
                except Exception:
                    pass
            # 基準表示の反映（"基準: W×H" / "基準: --"）
            if hasattr(panel, "update_aspect_base"):
                try:
                    panel.update_aspect_base(base if on else None)
                except Exception:
                    pass

        # --- ON時：ドラッグ矩形があるなら固定モードに昇格 ---
        if on and not getattr(self.label, "fixed_crop_mode", False) and getattr(self.label, "drag_rect_img", None):
            r = QtCore.QRect(self.label.drag_rect_img)

            self.label.fixed_crop_mode = True
            self.label.fixed_crop_rect_img = QtCore.QRect(r)
            self.label.drag_rect_img = None
            self._crop_rect_img = QtCore.QRect(r)
            try:
                self.label.fixed_crop_size = (r.width(), r.height())
            except Exception:
                pass

            # パネルを固定モードとして再表示（ラベル座標へ変換）
            try:
                x1, y1 = self.label.image_to_label_coords(r.left(),  r.top())
                x2, y2 = self.label.image_to_label_coords(r.right(), r.bottom())
                rl = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)
                self.show_action_panel(rl, True)  # is_fixed=True
            except Exception:
                pass

            # プレビュー/サイズラベル更新
            try:
                self.update_crop_size_label(self._crop_rect_img, img_space=True)
            except Exception:
                pass
            try:
                if hasattr(self, "safe_update_preview"):
                    QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect_img))
                else:
                    self._schedule_preview(self._crop_rect_img)
            except Exception:
                pass

            # 任意：固定モードのメニュー同期
            try:
                act = getattr(self, "act_fixed_mode", None)
                if act and hasattr(act, "setChecked"):
                    act.setChecked(True)
            except Exception:
                pass

        # --- 見た目リフレッシュ（ハンドル/カーソル等） ---
        try:
            self.label.refresh_edit_ui()
        except Exception:
            try: self.label._resize_handle = None
            except Exception: pass
            try: self.label._hovered_handle = None
            except Exception: pass
            try: self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            except Exception: pass
            self.label.update()

    def reseed_aspect_base_from_current_rect(self):
        """比率固定がONのとき、今見えている矩形サイズを基準(W×H)として取り直し、表示も即更新。"""
        lbl = getattr(self, "label", None)
        if lbl is None:
            return
        if not bool(getattr(lbl, "_aspect_lock", False)):
            return  # OFFなら何もしない

        # いまの矩形（画像座標）を拾う：固定→ドラッグ→最後に保持している画像座標
        r = None
        if getattr(lbl, "fixed_crop_mode", False) and getattr(lbl, "fixed_crop_rect_img", None) is not None:
            r = lbl.fixed_crop_rect_img
        elif getattr(lbl, "drag_rect_img", None) is not None:
            r = lbl.drag_rect_img
        elif getattr(self, "_crop_rect_img", None) is not None:
            r = self._crop_rect_img

        if r is None or r.width() <= 0 or r.height() <= 0:
            return

        base = (int(r.width()), int(r.height()))
        lbl._aspect_base_wh = base
        try:
            lbl._aspect_ratio = base[0] / base[1]
        except Exception:
            lbl._aspect_ratio = None

        # ダイアログ表示も即更新
        dlg = getattr(self, "_nudge_overlay", None)
        if dlg and hasattr(dlg, "set_aspect_base_text"):
            dlg.set_aspect_base_text(base)

        # 見た目の再描画
        try:
            lbl.update()
        except Exception:
            pass
    
    def _ensure_aspect_base_from_current_rect(self) -> None:
        """比率固定ONかつ基準未設定なら、現在の矩形から基準(W,H)を決めてUIに反映"""
        lbl = getattr(self, "label", None)
        if not lbl or not getattr(lbl, "_aspect_lock", False):
            return
        if getattr(lbl, "_aspect_base_wh", None):
            return  # もう決まっている

        # 現在の画像座標の矩形を拾う（優先度：_crop_rect_img → fixed → drag）
        r = getattr(self, "_crop_rect_img", None)
        if r is None:
            if getattr(lbl, "fixed_crop_mode", False) and getattr(lbl, "fixed_crop_rect_img", None):
                r = lbl.fixed_crop_rect_img
            elif getattr(lbl, "drag_rect_img", None):
                r = lbl.drag_rect_img
        if r is None or r.width() <= 0 or r.height() <= 0:
            return

        base = (int(r.width()), int(r.height()))
        lbl._aspect_base_wh = base
        lbl._aspect_ratio   = base[0] / base[1]

        # ダイアログ表示も更新
        dlg = getattr(self, "_nudge_overlay", None)
        if dlg and hasattr(dlg, "set_aspect_base_text"):
            dlg.set_aspect_base_text(base)

    def nudge_edge(self, side: str, delta_px: int) -> int:
        # いまの矩形（画像座標）を取得（固定枠が優先）
        fixed = (
            bool(getattr(self.label, "fixed_crop_mode", False)) and
            getattr(self.label, "fixed_crop_rect_img", None) is not None
        )
        r = (QtCore.QRect(self.label.fixed_crop_rect_img) if fixed
            else QtCore.QRect(getattr(self.label, "drag_rect_img", None) or QtCore.QRect()))
        if r.isNull():
            return 0

        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()

        # ==== 画像端によるクランプは一切しない ====
        # ただし幅・高さが 1px 未満にならないように、反対側との関係だけ制限する
        applied = 0
        if side == "top":
            old = top
            new = min(old + delta_px, bottom - 1)
            top = new
            applied = new - old
            old_edge, new_edge = old, new
        elif side == "bottom":
            old = bottom
            new = max(old + delta_px, top + 1)
            bottom = new
            applied = new - old
            old_edge, new_edge = old, new
        elif side == "left":
            old = left
            new = min(old + delta_px, right - 1)
            left = new
            applied = new - old
            old_edge, new_edge = old, new
        elif side == "right":
            old = right
            new = max(old + delta_px, left + 1)
            right = new
            applied = new - old
            old_edge, new_edge = old, new
        else:
            return 0

        new_rect = QtCore.QRect(QtCore.QPoint(left, top), QtCore.QPoint(right, bottom))

        # ========= カウンタ抑制判定（画像外に出ている間は増減させない） =========
        img_w = img_h = None
        im = getattr(self, "image", None)
        if im is not None:
            try:
                img_w, img_h = im.size  # PIL Image
            except Exception:
                try:
                    img_w = int(getattr(im, "width", None) or 0) or None
                    img_h = int(getattr(im, "height", None) or 0) or None
                except Exception:
                    pass
        if (img_w is None or img_h is None) and hasattr(self.label, "pixmap"):
            pm = self.label.pixmap()
            if pm is not None:
                if img_w is None:
                    img_w = pm.width()
                if img_h is None:
                    img_h = pm.height()

        def out_dir(v: int, vmin: int, vmax: int) -> int:
            """境界の外向き方向：内側=0、上/左=-1、下/右=+1"""
            if v < vmin: return -1
            if v > vmax: return +1
            return 0

        applied_for_counter = applied
        if img_w is not None and img_h is not None:
            if side in ("top", "bottom"):
                vmin, vmax = 0, img_h - 1
            else:
                vmin, vmax = 0, img_w - 1

            was = out_dir(old_edge, vmin, vmax)
            now = out_dir(new_edge, vmin, vmax)
            # すでに外側で、さらに外側へ進む時はカウンタだけ止める
            if was != 0 and now == was:
                applied_for_counter = 0

        # 画像外OKのまま反映（パネルは動かさない）
        self._crop_rect_img = QtCore.QRect(new_rect)
        if fixed:
            self.label.fixed_crop_rect_img = new_rect
        else:
            self.label.drag_rect_img = new_rect

        # ラベル座標を内部保持（描画用）。外へは移動通知を飛ばさない＝パネル据え置き
        x1, y1 = self.label.image_to_label_coords(new_rect.left(),  new_rect.top())
        x2, y2 = self.label.image_to_label_coords(new_rect.right(), new_rect.bottom())
        self._crop_rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)

        # サイズ表示
        if hasattr(self, "update_crop_size_label"):
            self.update_crop_size_label(self._crop_rect_img, img_space=True)

        # プレビュー更新（画像内にクリップして渡す）
        def _update_preview_clipped():
            iw, ih = img_w, img_h
            if iw and ih:
                img_rect = QtCore.QRect(0, 0, int(iw), int(ih))
                clip = self._crop_rect_img.intersected(img_rect)
            else:
                clip = self._crop_rect_img
            if hasattr(self, "safe_update_preview"):
                self.safe_update_preview(clip)
            elif hasattr(self, "_schedule_preview"):
                self._schedule_preview(clip)

        QtCore.QTimer.singleShot(0, _update_preview_clipped)

        # 再描画
        self.label.update()
        return int(applied_for_counter)

    def _compute_action_panel_pos(self, rect: QtCore.QRect, is_fixed: bool) -> QtCore.QPoint:
        # パネル実寸（未生成なら見積もり）
        panel = getattr(self, "_action_panel", None)
        if panel and panel.size().isValid():
            pw, ph = panel.width(), panel.height()
        else:
            pw = 180 if is_fixed else 255
            ph = 28

        lw, lh = self.label.width(), self.label.height()

        # 矩形の外側に出すギャップ
        gap_x, gap_y = 0, 0

        # ★「外側の左上」を望ましい位置にする（＝右下に出す）
        x_desired = rect.x() + rect.width()  + gap_x
        y_desired = rect.y() + rect.height() + gap_y

        # マージン（下端は0で跳ね返り抑止）
        ml, mt, mr, mb = 8, 8, 8, 0

        # クランプ
        x = max(ml, min(x_desired, lw - pw - mr))
        y = max(mt, min(y_desired, lh - ph - mb))

        # ★右端クランプが発動していたら、右余白だけ“寄せ切り”で上書き（固定/ドラッグ共通）
        if x_desired >= (lw - pw - mr):
            edge_margin_right_on_clamp = 0   # 0〜4でお好み
            x = max(ml, lw - pw - edge_margin_right_on_clamp)

        return QtCore.QPoint(int(x), int(y))

    def _constrain_to_screen(self, x: int, y: int, w: int, h: int, margin: int = 8) -> tuple[int, int]:
        """(x,y,w,h) をスクリーンの availableGeometry 内に収める"""
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        nx = max(avail.left() + margin, min(x, avail.right()  - w - margin))
        ny = max(avail.top()  + margin, min(y, avail.bottom() - h - margin))
        return nx, ny

    def _position_nudge_overlay_above_action(self, dlg: QtWidgets.QWidget, gap: int = 12, margin: int = 8) -> None:
        """
        独立Nudgeパネルを ActionPanel の少し上に出す。
        上に置けない場合は下に回す。左右/上下とも画面からはみ出さないようクランプ。
        """
        panel = getattr(self, "_action_panel", None)

        # まずダイアログのサイズを確定
        try:
            dlg.adjustSize()
        except Exception:
            pass
        dw, dh = dlg.size().width(), dlg.size().height()
        if dw <= 0 or dh <= 0:
            s = dlg.sizeHint()
            dw, dh = max(1, s.width()), max(1, s.height())

        if not panel or not panel.isVisible():
            # フォールバック：画像ラベル上部に出す
            try:
                base = self.label.mapToGlobal(QtCore.QPoint(self.label.width() // 2, 0))
                x = base.x() - dw // 2
                y = base.y() + gap
            except Exception:
                x, y = 100, 100  # 最終フォールバック
            x, y = self._constrain_to_screen(x, y, dw, dh, margin)
            dlg.move(x, y)
            return

        # ActionPanel のグローバル矩形
        p_tl = panel.mapToGlobal(QtCore.QPoint(0, 0))
        pr = QtCore.QRect(p_tl, panel.size())

        # “ちょい上”に中央合わせで置く
        x = pr.center().x() - dw // 2
        y = pr.top() - gap - dh

        # 上に置けないなら下に回す
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        if y < avail.top() + margin:
            y = pr.bottom() + max(gap - 4, 0) 

        # 画面内にクランプ
        x, y = self._constrain_to_screen(x, y, dw, dh, margin)
        dlg.move(x, y)

    def _suspend_nudge_overlay(self, hide: bool):
        """True=一時的に隠す / False=再表示しつつ保存パネルの近くに寄せ直す"""
        ov = getattr(self, "_nudge_overlay", None)
        if not ov:
            return
        try:
            if hide:
                ov.hide()
            else:
                # 調整モード中＆アクションパネルが出ているなら寄せ直して再表示
                if getattr(self.label, "adjust_mode", False):
                    ap = getattr(self, "_action_panel", None)
                    if ap and ap.isVisible():
                        self._position_nudge_overlay(ov, gap=getattr(self, "nudge_gap_px", 8))
                ov.show()
        except Exception:
            pass
    
    def _current_fixed_label_rect(self):
        return self.label._fixed_rect_labelcoords()
    
    def _sync_fixed_ui_after_image_change(self):
        """画像の描画（show_image）後に、固定枠のUIを最新の座標に同期する。"""
        if not (self.label.fixed_crop_mode and
                getattr(self.label, "fixed_crop_rect_img", None) is not None):
            return

        # 画像座標 → ラベル座標に変換して最新化
        rect_now = self._current_fixed_label_rect()
        if not rect_now:
            return

        self._crop_rect = rect_now
        self._crop_rect_img = QtCore.QRect(self.label.fixed_crop_rect_img)

        # パネル位置を再スナップ
        if self._action_panel:
            self._action_panel.move(self._compute_action_panel_pos(rect_now, True))
            self._action_panel.raise_()
        else:
            self.show_action_panel(rect_now, self._compute_action_panel_pos(rect_now, True))

        # ラベル＆プレビューも更新
        self.update_crop_size_label(self._crop_rect_img, img_space=True)
        QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect_img))

    def do_crop_save(self):
        rect = self._crop_rect
        ok, info = self.save_cropped(rect)

        # --- Success の表示位置は「グローバル座標」で計算する ---
        if self._action_panel and hasattr(self._action_panel, "btn_save"):
            btn = self._action_panel.btn_save
            pos_g = btn.mapToGlobal(btn.rect().center()) + QtCore.QPoint(0, -40)
        else:
            pos_g = self.mapToGlobal(QtCore.QPoint(100, 100))

        # --- Success トースト（毎回 flash を叩く統一版）---
        msg = "✔ Success" if ok else "✖ Failed"
        lab = getattr(self, "_success_label", None)
        if not isinstance(lab, SuccessLabel):
            # 初回だけ生成（位置・文言は渡さない）
            self._success_label = lab = SuccessLabel(parent=self)

        # 毎回 flash 経由で表示＆タイマー開始（最前面化も flash 側で対応）
        lab.flash(pos=pos_g, message=msg, ok=ok, timeout=1500)

        # 失敗時はここで終了（保存系の後処理はしない）
        if not ok:
            return

        # ===== ここから保存成功後の後処理 =====
        if self.label.fixed_crop_mode:
            rect_now = self._current_fixed_label_rect()
            if rect_now:
                self._crop_rect = rect_now
                self._crop_rect_img = QtCore.QRect(self.label.fixed_crop_rect_img)
                if self._action_panel:
                    pos1 = self._compute_action_panel_pos(rect_now, is_fixed=True)
                    self._action_panel.move(pos1)
                    self._action_panel.raise_()
                    QtCore.QTimer.singleShot(0, lambda:
                        self._action_panel.move(self._compute_action_panel_pos(self._crop_rect, True))
                    )
            self.update_crop_size_label(self._crop_rect_img, img_space=True)
            self._schedule_preview(self._crop_rect_img)
        else:
            if self._crop_rect is not None and self._action_panel:
                self._action_panel.move(self._compute_action_panel_pos(self._crop_rect, is_fixed=False))
                self._action_panel.raise_()
            if getattr(self, "_crop_rect_img", None) is not None:
                self.update_crop_size_label(self._crop_rect_img, img_space=True)
                self._schedule_preview(self._crop_rect_img)

    def _ensure_jpeg_compatible(self, img, save_ext: str):
        """
        JPEG保存に備えて Pillow Image を安全化する。
        - αを含む場合は白背景に合成してRGB化
        - JPEGが受け付けないモードはRGBへ
        """
        try:
            save_ext = (save_ext or "").lower()
            if save_ext in ("jpg", "jpeg"):
                if "A" in img.getbands():
                    # 透過は白で合成
                    from PIL import Image
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
                elif img.mode not in ("RGB", "L", "CMYK"):
                    img = img.convert("RGB")
        except Exception:
            # ここで落とすと保存全体が死ぬので黙ってフォールバック
            pass
        return img

    def save_cropped(self, rect):
        if self.image is None:
            return False, "画像が読み込まれていません"

        # 元画像サイズ
        img_w, img_h = self.image.width, self.image.height

        # ---- 画像座標の切り出し矩形を決定（fixed優先 → _crop_rect_img → rect から変換）----
        img_rect = getattr(self.label, "fixed_crop_rect_img", None)
        if img_rect is not None:
            x = int(img_rect.left());  y = int(img_rect.top())
            w = int(img_rect.width()); h = int(img_rect.height())
        else:
            if getattr(self, "_crop_rect_img", None) is not None:
                x = int(self._crop_rect_img.left());  y = int(self._crop_rect_img.top())
                w = int(self._crop_rect_img.width()); h = int(self._crop_rect_img.height())
            else:
                if rect is None:
                    print("[SAVE ERROR] 切り出し範囲が指定されていません")
                    return False, "切り出し範囲が指定されていません"
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.left() + rect.width(), rect.top() + rect.height()
                gx1, gy1 = self.label.label_to_image_coords(x1, y1)
                gx2, gy2 = self.label.label_to_image_coords(x2, y2)
                x = min(gx1, gx2); y = min(gy1, gy2)
                w = abs(gx2 - gx1); h = abs(gy2 - gy1)

        # 画像境界にクランプ
        left   = max(0, x)
        top    = max(0, y)
        right  = min(img_w, x + w)
        bottom = min(img_h, y + h)

        if right - left <= 0 or bottom - top <= 0:
            print("[SAVE ERROR] 切り出し範囲が画像外です")
            return False, "切り出し範囲が画像外です"

        # “画像全体を覆っているか” 判定（後でコピー保存に使う）
        is_full_cover = (left == 0 and top == 0 and right == img_w and bottom == img_h)

        try:
            box = (left, top, right, bottom)
            cropped = self.image.crop(box)

            # 保存先ルート（Path で保持）
            if self.save_folder:
                # 「フォルダ指定」などで固定されている場合はこちら
                save_root = Path(self.save_folder)
            else:
                # 「読込み元と同一」モード → 画像の元フォルダ
                # （zip:// のときは zip が置いてあるフォルダ）
                save_root = Path(self._get_image_source_dir(self.image_path))

            # ホーム(~)展開など軽い正規化
            save_root = save_root.expanduser()
            save_root.mkdir(parents=True, exist_ok=True)

            # 文字列パスとしても使いたい場面があるので、従来の folder も残しておく
            folder = str(save_root)

            # 既存画像パスから「出力に使うファイル名だけ」を決定
            output_name = self._output_name_from_image_path()
            # そのファイル名から拡張子を抜き出して保存形式を決める
            base, ext0 = os.path.splitext(output_name)
            orig_ext = ext0.lower().lstrip(".")
            # 出力拡張子（未知はpng）
            save_ext = orig_ext if orig_ext in ("jpg","jpeg","png","bmp","gif","tif","tiff","webp") else "png"

            # ===== メタデータ（EXIF/ICC）を拾う =====
            src_info = {}
            try:
                if hasattr(self.image, "info"):
                    src_info = self.image.info or {}
            except Exception:
                pass
            exif = src_info.get("exif")
            icc  = src_info.get("icc_profile")

            # ===== 形式別の保存キーワード =====
            save_kw = {}
            if save_ext in ("jpg", "jpeg"):
                save_kw.update(dict(quality=95, subsampling=0, optimize=True))
            elif save_ext == "webp":
                save_kw.update(dict(quality=90, method=6))
            elif save_ext == "png":
                save_kw.update(dict(compress_level=6))
            if exif:
                save_kw["exif"] = exif
            if icc:
                save_kw["icc_profile"] = icc

            # ========== 上書き保存モード ==========
            if getattr(self, "overwrite_mode", False):
                import shutil

                if self.save_folder:
                    save_root = Path(self.save_folder).expanduser()
                    save_root.mkdir(parents=True, exist_ok=True)
                    folder = str(save_root)

                dst_path = str(save_root / output_name)

                # ---- “全体覆い”かつロッシー形式はコピーで無劣化 ----
                if is_full_cover and orig_ext in ("jpg", "jpeg", "webp"):
                    try:
                        if os.path.normcase(os.path.abspath(self.image_path)) != os.path.normcase(os.path.abspath(dst_path)):
                            shutil.copy2(self.image_path, dst_path)
                        print("上書き保存(コピーで無劣化):", dst_path)
                    except Exception as e:
                        print("[SAVE WARNING] copy2失敗 -> エンコード保存にフォールバック:", e)
                        self._ensure_jpeg_compatible(cropped, save_ext).save(dst_path, **save_kw)
                        print("上書き保存(エンコード):", dst_path)
                else:
                    self._ensure_jpeg_compatible(cropped, save_ext).save(dst_path, **save_kw)
                    print("上書き保存:", dst_path)

                # mtime を確実に動かす（低粒度FS対策）
                try:
                    os.utime(dst_path, None)
                except Exception:
                    pass

                # ★ 上書き時の「開き直し」条件分岐
                #   1) 保存先が読込元と同一 → 開き直して即反映
                #   2) 保存先が別           → 開き直しなし
                try:
                    # 読み込み元の“基準フォルダ”
                    # （zip:// 等も考慮したいなら _get_image_source_dir を使うのが安全）
                    src_dir = os.path.normcase(os.path.abspath(self._get_image_source_dir(self.image_path)))

                    # 保存先フォルダ（save_root は overwrite ブロックで確定済み）
                    dst_dir = os.path.normcase(os.path.abspath(str(save_root)))

                    if src_dir == dst_dir:
                        # 可能ならUI状態を温存してから開き直す
                        try:
                            state = self._snapshot_adjust_state()
                            if not state:
                                state = {"rect": "full"}
                        except Exception:
                            state = {"rect": "full"}
                        self._preserve_ui_on_next_load = state

                        # 次の読み込み1回だけ保存先ダイアログ抑止（持っているなら）
                        try:
                            s = getattr(self, "_suppress_save_dialog_paths", None)
                            if s is None:
                                s = set()
                                self._suppress_save_dialog_paths = s
                            s.add(os.path.normcase(os.path.abspath(dst_path)))
                        except Exception:
                            pass

                        # 同一フォルダ上書きは “保存した実体” を開き直す
                        try:
                            self.open_image_from_path(dst_path)
                        except Exception:
                            pass
                    else:
                        # 保存先が別フォルダなら、見た目リセットを避けるため開き直さない
                        pass

                except Exception:
                    pass
                
                try:
                    if getattr(self, "model", None):
                        print("[thumb] invalidate request:", dst_path)
                        self.model.invalidate_path(dst_path)
                except Exception as e:
                    print("[thumb] invalidate request failed:", e)

                return True, dst_path

            # ========== 通常（連番保存） ==========
            # ① α有のときJPEGは避け、pngに自動切替
            has_alpha = ("A" in cropped.getbands())
            if save_ext in ("jpg", "jpeg") and has_alpha:
                save_ext = "png"

            # ② 連番を探す
            i = 1
            candidate = None
            while i < 1000:
                name = f"{base}_cropped_{i:03d}.{save_ext}"
                candidate = save_root / name
                if not candidate.exists():
                    break
                i += 1
            if i >= 1000:
                raise RuntimeError("保存先に連番の空きがありません (001-999).")

            candidate = str(candidate)  # この先は文字列として扱う

            # ③ “全体覆い”かつ ロッシー形式で拡張子も同じなら コピーで無劣化
            if is_full_cover and save_ext in ("jpg", "jpeg", "webp") and save_ext == orig_ext:
                try:
                    import shutil
                    shutil.copy2(self.image_path, candidate)
                    print("保存(コピーで無劣化):", candidate)
                except Exception as e:
                    print("[SAVE WARNING] copy2失敗 -> エンコード保存にフォールバック:", e)
                    self._ensure_jpeg_compatible(cropped, save_ext).save(candidate, **save_kw)
                    print("保存(エンコード):", candidate)
            else:
                self._ensure_jpeg_compatible(cropped, save_ext).save(candidate, **save_kw)
                print("保存:", candidate)

            return True, candidate

        except Exception as e:
            print("[SAVE ERROR]", e)
            return False, str(e)

    # ------------------------------
    # 一括切り取り（現在フォルダ内の画像全部）
    # ------------------------------
    def on_batch_crop_clicked(self):
        """
        現在の切り出し矩形をテンプレにして、同じフォルダ内の全画像を一括で切り取り保存する。
        - 解像度が違う画像は「元画像に対する割合」で矩形を再計算する
        - ★ ただし割合化の前に「基準画像の範囲内へ矩形をクリップ」する
        - ★ 解像度/アスペクト比が混在していそうなら警告して
          「近似モードで続行」「中止」を選ばせる
        - 保存先は現在の設定（同じフォルダ / 別フォルダ）に従う
        """
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        import os, traceback

        # ---- ロガーがあれば使う ----
        def _log(*args):
            try:
                log_debug(*args)
            except Exception:
                pass

        # ---- ベースとなる画像＆矩形が無い場合は中止 ----
        if self.image is None or self._crop_rect_img is None:
            _log("[BATCH] base image/rect missing",
                 "image?", self.image is not None,
                 "rect?", self._crop_rect_img is not None)
            QMessageBox.warning(self, "一括切り取り", "一括切り取りの元になる切り出し範囲がありません。")
            return

        base_img = self.image
        try:
            base_w, base_h = base_img.size
        except Exception:
            _log("[BATCH] base_img.size failed\n" + traceback.format_exc())
            QMessageBox.warning(self, "一括切り取り", "現在の画像サイズを取得できません。")
            return

        rect = QtCore.QRect(self._crop_rect_img).normalized()
        _log("[BATCH] base size:", (base_w, base_h))
        _log("[BATCH] base rect img:", rect)

        if rect.width() <= 0 or rect.height() <= 0:
            _log("[BATCH] invalid rect size:", rect.width(), rect.height())
            QMessageBox.warning(self, "一括切り取り", "切り出し範囲が正しくありません。")
            return

        # =========================================================
        # ★ 1) 基準画像側で一回クリップしてから比率化
        # =========================================================
        base_bounds = QtCore.QRect(0, 0, int(base_w), int(base_h))
        clipped = rect.intersected(base_bounds)

        # QRect.intersected は交差が無いと width/height 0 になりがちなので保険
        if clipped.isNull() or clipped.width() <= 0 or clipped.height() <= 0:
            _log("[BATCH] rect outside base after clip:", clipped)
            QMessageBox.warning(
                self, "一括切り取り",
                "切り出し範囲が基準画像の外にはみ出しています。\n"
                "基準画像内に収まる範囲で矩形を作り直してください。"
            )
            return

        if clipped != rect:
            _log("[BATCH] base rect clipped:",
                 "from", rect, "to", clipped)
        rect = clipped

        # 画像座標 → 割合に変換（クリップ後）
        fx = rect.left() / base_w
        fy = rect.top() / base_h
        fw = rect.width() / base_w
        fh = rect.height() / base_h

        _log("[BATCH] base ratios:", "fx=", fx, "fy=", fy, "fw=", fw, "fh=", fh)

        # ---- 対象となる画像一覧（現状の image_list を受ける） ----
        raw_paths = list(getattr(self, "image_list", []) or [])
        _log("[BATCH] raw image_list len =", len(raw_paths))

        # デバッグのため、まず中身をざっくり表示
        for p in raw_paths[:50]:
            try:
                s = str(p)
                ext = os.path.splitext(s)[1].lower()
                _log("[BATCH] raw item:", repr(s), "ext=", ext, "zip_uri=", is_zip_uri(s))
            except Exception:
                _log("[BATCH] raw item inspect failed:", repr(p))

        if not raw_paths:
            QMessageBox.information(self, "一括切り取り", "一括切り取りの対象となる画像がありません。")
            return

        # ここで「画像名だけ」にフィルタ
        paths = []
        for p in raw_paths:
            try:
                s = str(p)
                if is_image_name(s) or is_image_name(os.path.basename(s)):
                    paths.append(s)
            except Exception:
                pass

        _log("[BATCH] filtered paths len =", len(paths))
        for p in paths[:50]:
            _log("[BATCH] use:", repr(p))

        if not paths:
            _log("[BATCH] no image-like entries after filter")
            QMessageBox.information(self, "一括切り取り", "対象になりそうな画像が見つかりません。")
            return

        # =========================================================
        # ★ 2) 解像度・アスペクト比のばらつきを軽く集計して警告
        #    - “回転履歴”を考慮した見た目サイズで判定して
        #      いらんタイミングの警告を抑止する
        # =========================================================

        # ---- 現在の回転/反転履歴を取得（警告判定と実処理で共通に使う）----
        ops = []
        try:
            if hasattr(self, "_get_batch_transform_ops"):
                ops = self._get_batch_transform_ops() or []
        except Exception:
            ops = []

        # ---- ★ 回転履歴だけをサイズに反映するライト関数 ----
        #      （警告用：画像を実際に開いて回す必要はない）
        def _apply_ops_to_wh(w0: int, h0: int):
            w = int(w0); h = int(h0)
            for op in ops:
                if op in ("rot_left_90", "rot_right_90"):
                    w, h = h, w
            return w, h

        # ---- ★ 警告に使う“基準サイズ”の扱い ----
        # base_w/base_h は「現在表示中の self.image」由来で
        # 既に回転が反映されている場合がある。
        # ここでさらに ops を当てると 90°回転が二重適用され、
        # 不要な“解像度/比率混在”警告の原因になる。
        #
        # なので警告用の基準は「元ファイルサイズ」→ ops で算出する。
        raw_base_w, raw_base_h = base_w, base_h
        try:
            if getattr(self, "image_path", None):
                _raw_img = open_image_any(self.image_path)
                raw_base_w, raw_base_h = _raw_img.size
        except Exception:
            pass

        warn_base_w, warn_base_h = _apply_ops_to_wh(raw_base_w, raw_base_h)
        # デバッグしたいなら一時的にON
        _log("[BATCH] warn base raw:", (raw_base_w, raw_base_h), "ops:", ops, "->", (warn_base_w, warn_base_h))

        def _collect_variance_info(sample_paths, max_probe=120):
            sizes = []
            ars = []
            probed = 0
            failed = 0

            for sp in sample_paths[:max_probe]:
                try:
                    img = open_image_any(sp)
                    w, h = img.size

                    # ★ 回転履歴を考慮した“見た目サイズ”で集計
                    w, h = _apply_ops_to_wh(w, h)

                    sizes.append((int(w), int(h)))
                    if h:
                        ars.append(float(w) / float(h))
                    probed += 1
                except Exception:
                    failed += 1

            uniq_sizes = sorted(set(sizes))

            # ★ 基準ARも「回転履歴を反映した基準サイズ」で計算
            base_ar = (float(warn_base_w) / float(warn_base_h)) if warn_base_h else 0.0

            max_ar_delta = 0.0
            min_ar = None
            max_ar = None
            if ars:
                min_ar = min(ars)
                max_ar = max(ars)
                max_ar_delta = max(abs(min_ar - base_ar), abs(max_ar - base_ar))

            return {
                "probed": probed,
                "failed": failed,
                "uniq_sizes": uniq_sizes,
                "uniq_size_count": len(uniq_sizes),

                # ★ 表示用の基準サイズ/ARも警告専用の値に
                "base_size": (int(warn_base_w), int(warn_base_h)),
                "base_ar": base_ar,

                "min_ar": min_ar,
                "max_ar": max_ar,
                "max_ar_delta": max_ar_delta,
            }

        info = _collect_variance_info(paths)

        # ★ ここが最重要ログ
        _log("[BATCH] WARN info:", info)

        # しきい値は「実用でウザくなりにくい」程度に緩め
        # - サイズが2種類以上
        # - あるいは基準ARとの差がそこそこ大きい
        need_warn = (
            info["uniq_size_count"] >= 2
            or (info["max_ar_delta"] is not None and info["max_ar_delta"] >= 0.08)
        )

        if need_warn:
            # 表示用テキスト
            uniq_preview = info["uniq_sizes"][:8]
            uniq_txt = ", ".join([f"{w}×{h}" for (w, h) in uniq_preview])
            if info["uniq_size_count"] > len(uniq_preview):
                uniq_txt += f" ...(+{info['uniq_size_count']-len(uniq_preview)})"

            ar_txt = "—"
            if info["min_ar"] is not None and info["max_ar"] is not None:
                ar_txt = f"{info['min_ar']:.3f} ～ {info['max_ar']:.3f}"

            warn_msg = (
                "フォルダ内に、解像度やアスペクト比が異なる画像が混在している可能性があります。\n\n"
                f"基準画像: {info['base_size'][0]}×{info['base_size'][1]}  (AR={info['base_ar']:.3f})\n"
                f"検査サンプル: {info['probed']} 枚  / 失敗: {info['failed']} 枚\n"
                f"検出された解像度の種類: {info['uniq_size_count']}\n"
                f"例: {uniq_txt}\n"
                f"サンプルAR範囲: {ar_txt}\n\n"
                "このまま続行すると、画像によっては意図と違う位置/範囲が切り取られるかもしれません。\n"
                "（基準矩形を割合に換算して近似クロップします）"
            )

            mb = QMessageBox(self)
            mb.setIcon(QMessageBox.Icon.Warning)
            mb.setWindowTitle("一括切り取り")
            mb.setText("解像度/比率の混在を検出しました")
            mb.setInformativeText(warn_msg)

            btn_continue = mb.addButton("近似モードで続行", QMessageBox.ButtonRole.AcceptRole)
            btn_cancel = mb.addButton("中止", QMessageBox.ButtonRole.RejectRole)
            mb.setDefaultButton(btn_cancel)

            mb.exec()
            if mb.clickedButton() != btn_continue:
                _log("[BATCH] canceled at variance warning")
                return

        # =========================================================
        # 3) 通常の最終確認
        # =========================================================
        msg = (
            f"現在の切り出し範囲を元に、このフォルダ内の画像 {len(paths)} 枚を\n"
            f"一括で切り取り＆保存します。\n\n"
            f"・保存先: 現在の設定に従います\n"
            f"・上書きモード: {'上書き保存' if getattr(self, 'overwrite_mode', False) else '連番保存'}\n\n"
            f"よろしいですか？"
        )
        ret = QMessageBox.question(
            self, "一括切り取り", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            _log("[BATCH] canceled by user")
            return

        # ---- 進捗ダイアログ ----
        progress = QProgressDialog("一括切り取り中.", "キャンセル", 0, len(paths), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)

        errors = 0
        processed = 0
        first_error_text = None

        # ★ 上書き＆同一フォルダ時の即時反映用
        overwrite = bool(getattr(self, "overwrite_mode", False))
        changed_paths = []
        reload_current_after = False

        # =========================================================
        # 4) 実処理
        # =========================================================
        for i, src in enumerate(paths, start=1):
            progress.setValue(i - 1)
            progress.setLabelText(f"一括切り取り中.\n{src}")
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                _log("[BATCH] canceled during progress")
                break

            try:
                from PIL import Image, ImageOps

                # 画像を開く
                img = open_image_any(src)

                # ★ 今の画像に至る変形履歴を、対象画像にも同じ順序で適用
                for op in ops:
                    if op == "flip_h":
                        img = ImageOps.mirror(img)
                    elif op == "flip_v":
                        img = ImageOps.flip(img)
                    elif op == "rot_left_90":
                        img = img.transpose(Image.ROTATE_90)
                    elif op == "rot_right_90":
                        img = img.transpose(Image.ROTATE_270)

                # 変形後サイズで割合クロップする
                w, h = img.size

                # 割合からこの画像の矩形を計算
                x = int(round(fx * w))
                y = int(round(fy * h))
                cw = int(round(fw * w))
                ch = int(round(fh * h))

                _log("[BATCH] opened size:", (w, h))
                _log("[BATCH] calc rect:", (x, y, cw, ch))

                # 画面外に出ないようにクリップ
                if cw <= 0 or ch <= 0:
                    raise ValueError("切り出し範囲が画像外になりました（cw/ch<=0）。")

                if x < 0:
                    cw += x
                    x = 0
                if y < 0:
                    ch += y
                    y = 0
                if x + cw > w:
                    cw = w - x
                if y + ch > h:
                    ch = h - y

                _log("[BATCH] clipped rect:", (x, y, cw, ch))

                if cw <= 0 or ch <= 0:
                    raise ValueError("切り出し範囲が画像外になりました（after clip）。")

                cropped = img.crop((x, y, x + cw, y + ch))

                # 保存先フォルダとファイル名を決定して保存
                save_root = self._resolve_batch_save_root(src)
                save_root.mkdir(parents=True, exist_ok=True)
                dst_path = self._build_batch_output_path(save_root, src)

                # 拡張子判定
                dst_ext = dst_path.suffix.lower()
                if dst_ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
                    dst_ext = ".png"
                    dst_path = dst_path.with_suffix(dst_ext)

                save_kwargs: dict = {}
                if dst_ext in (".jpg", ".jpeg"):
                    # 既存実装の引数違いに両対応
                    try:
                        cropped = self._ensure_jpeg_compatible(cropped, dst_ext.lstrip("."))
                    except TypeError:
                        cropped = self._ensure_jpeg_compatible(cropped)
                    except Exception:
                        cropped = self._ensure_jpeg_compatible(cropped)

                    save_kwargs["quality"] = 95
                    save_kwargs["subsampling"] = 0

                cropped.save(str(dst_path), **save_kwargs)
                processed += 1
                _log("[BATCH] saved OK", str(dst_path))

                # ★ 上書き時は“変更されたパス”を記録してサムネ更新に使う
                if overwrite:
                    try:
                        changed_paths.append(str(dst_path))
                    except Exception:
                        pass

                    # 可能なら即 invalidate（サムネ欄の即時反映）
                    try:
                        if getattr(self, "model", None):
                            self.model.invalidate_path(os.path.normpath(str(dst_path)))
                    except Exception:
                        pass

                    # ★ 現在表示中の画像が今回上書きされた可能性があるなら、
                    #    ループ後に1回だけ開き直すフラグを立てる
                    try:
                        if getattr(self, "image_path", None):
                            cur = os.path.normcase(os.path.abspath(self.image_path))
                            sp  = os.path.normcase(os.path.abspath(str(src)))
                            dp  = os.path.normcase(os.path.abspath(str(dst_path)))
                            if cur == sp and sp == dp:
                                reload_current_after = True
                    except Exception:
                        pass

            except Exception as e:
                errors += 1
                tb = traceback.format_exc()
                _log("[BATCH] ERROR:", repr(e))
                _log(tb)
                if first_error_text is None:
                    first_error_text = f"{src}\n{repr(e)}\n{tb}"

        progress.setValue(len(paths))

        # =========================================================
        # ★ 4.5) 上書き＆同一フォルダ系の“最終即時反映”
        #        - 現在表示中の画像が対象なら最後に1回だけ開き直す
        # =========================================================
        if overwrite and reload_current_after:
            try:
                # UI状態を温存して開き直し
                try:
                    state = self._snapshot_adjust_state()
                    if not state:
                        state = {"rect": "full"}
                except Exception:
                    state = {"rect": "full"}
                self._preserve_ui_on_next_load = state

                self.open_image_from_path(self.image_path)
            except Exception:
                pass

        # ---- 結果表示 ----
        msg = f"一括切り取りが完了しました。\n\n成功: {processed} 件"
        if errors:
            msg += f"\nエラー: {errors} 件"
        QMessageBox.information(self, "一括切り取り", msg)

        if first_error_text:
            _log("[BATCH] first error detail:\n" + first_error_text)

        _log("[BATCH] ===== batch crop end =====\n")

    def _resolve_batch_save_root(self, src_path: str) -> Path:
        """
        一括切り取り時の保存先ルートフォルダを決定する。
        - 「別フォルダ指定」が有効ならそのフォルダ
        - それ以外は元画像と同じ場所（zip 内画像は zip ファイルのあるフォルダ）
        """
        # まずユーザー指定の保存先（既に save_folder に入れている想定）
        save_folder = getattr(self, "save_folder", "") or ""
        save_dest_mode = getattr(self, "save_dest_mode", "same")

        if save_dest_mode != "same" and save_folder:
            return Path(save_folder).expanduser()

        # zip:// の場合は zip 本体のフォルダを使う
        if is_zip_uri(src_path):
            zp, _inner = parse_zip_uri(src_path)
            return Path(os.path.dirname(zp) or ".").expanduser()

        # 通常ファイル
        return Path(os.path.dirname(src_path) or ".").expanduser()

    def _build_batch_output_path(self, save_root: Path, src_path: str) -> Path:
        """
        元画像の表示名から保存ファイル名を決定する。
        - 上書きモード ON  : 同名
        - 上書きモード OFF : name_cropped.png, name_cropped_002.png ... の連番
        """
        name = vfs_display_name(src_path, is_dir=False)
        base, ext = os.path.splitext(name)
        ext = ext.lower()

        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
            ext = ".png"

        overwrite = getattr(self, "overwrite_mode", False)

        if overwrite:
            return save_root / f"{base}{ext}"

        # 連番: xxx_cropped.png, xxx_cropped_002.png ...
        idx = 1
        while True:
            if idx == 1:
                cand = save_root / f"{base}_cropped{ext}"
            else:
                cand = save_root / f"{base}_cropped_{idx:03d}{ext}"
            if not cand.exists():
                return cand
            idx += 1
            if idx > 9999:
                # さすがに異常なので最後の名前を返して諦める
                return cand

    def on_crop_rect_moved(self, rect):
        self._crop_rect = rect

        img_rect = getattr(self.label, "drag_rect_img", None)
        if img_rect is None:
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.left() + rect.width(), rect.top() + rect.height()
            gx1, gy1 = self.label.label_to_image_coords(x1, y1)
            gx2, gy2 = self.label.label_to_image_coords(x2, y2)
            img_rect = QtCore.QRect(min(gx1, gx2), min(gy1, gy2), abs(gx2 - gx1), abs(gy2 - gy1))
        self._crop_rect_img = img_rect

        if self._action_panel:
            is_fixed = getattr(self.label, "fixed_crop_mode", False) and \
                      getattr(self.label, "fixed_crop_rect_img", None) is not None
            self._action_panel.move(self._compute_action_panel_pos(rect, is_fixed))

        self._schedule_preview(self._crop_rect_img)
        self.update_crop_size_label(self._crop_rect_img, img_space=True)


    def pin_current_rect(self):
        """
        現在の選択矩形（画像座標）を固定枠に昇格させ、
        パン/ズームや画像移動時にも維持させる。
        """
        # すでに固定枠なら何もしない
        if self.label.fixed_crop_mode and getattr(self.label, "fixed_crop_rect_img", None) is not None:
            return

        # 1) 画像座標の矩形を取得（優先: drag_rect_img → _crop_rect_img）
        img_rect = getattr(self.label, "drag_rect_img", None)
        if img_rect is None:
            img_rect = getattr(self, "_crop_rect_img", None)
        if img_rect is None:
            # 何も選ばれていなければ終了
            QMessageBox.information(self, "固定化", "固定化できる選択範囲がありません。")
            return

        # 2) 固定枠モードへ移行（画像座標サイズを“ソース・オブ・トゥルース”として保持）
        w, h = max(0, img_rect.width()), max(0, img_rect.height())
        if w == 0 or h == 0:
            QMessageBox.information(self, "固定化", "矩形のサイズが0です。")
            return

        self.label.fixed_crop_mode = True
        self.label.fixed_crop_size = (w, h)
        self.label.fixed_crop_rect_img = QtCore.QRect(img_rect)  # 画像座標で保持

        # --- 固定化サイズをカスタムサイズに同期し、トグルONにする ---
        self.custom_size = (w, h)
        if hasattr(self, "update_custom_edit_action_text"):
            self.update_custom_edit_action_text()

        # トグルON（ハンドラは走らせない：既に固定枠は作ってあるため）
        if hasattr(self, "custom_toggle_action"):
            blocker = QtCore.QSignalBlocker(self.custom_toggle_action)
            self.custom_toggle_action.setChecked(True)
            del blocker

        # プリセット側の✔は外して整合性を保つ
        if hasattr(self, "crop_actions"):
            for act in self.crop_actions.values():
                act.setChecked(False)

        # 3) UI用（ラベル座標矩形）を作ってプレビュー等を更新
        x1, y1 = self.label.image_to_label_coords(img_rect.left(), img_rect.top())
        x2, y2 = self.label.image_to_label_coords(img_rect.left() + img_rect.width(),
                                                  img_rect.top()  + img_rect.height())
        label_rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        self._crop_rect = label_rect
        self._crop_rect_img = QtCore.QRect(img_rect)  # 今後も画像座標で使う

        # プレビュー＆サイズ表示を画像座標で更新
        self.update_crop_size_label(self._crop_rect_img, img_space=True)
        QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect_img))

        # 既存パネルがあれば閉じて作り直し（真ん中に「固定化」ボタン含む）
        if self._action_panel:
            self._action_panel.close()
            self._action_panel = None

        self.show_action_panel(label_rect, True)

        # 再描画（固定枠は paintEvent 内で毎回“画像座標→ラベル座標”換算して描画されます）
        self.label.update()

        try:
            self.label.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

    def unfix_fixed_mode(self) -> None:
        """固定モードを解除して通常ドラッグに戻す（矩形は引き継ぐ）"""

        # 戻る押下時は微調整オーバーレイを閉じる（参照も捨てる）
        try:
            if hasattr(self, "close_nudge_overlay"):
                self.close_nudge_overlay()
        except Exception:
            pass
    
        lbl = self.label
        if getattr(lbl, "fixed_crop_mode", False):
            # 矩形を通常ドラッグ側へ引き継ぎ
            if getattr(lbl, "fixed_crop_rect_img", None):
                lbl.drag_rect_img = QtCore.QRect(lbl.fixed_crop_rect_img)
                self._crop_rect_img = QtCore.QRect(lbl.drag_rect_img)
            # 固定モード解除
            lbl.fixed_crop_mode = False
            lbl.fixed_crop_rect_img = None

            # パネルを作り直し（通常モードとして）
            try:
                x1, y1 = lbl.image_to_label_coords(self._crop_rect_img.left(),  self._crop_rect_img.top())
                x2, y2 = lbl.image_to_label_coords(self._crop_rect_img.right(), self._crop_rect_img.bottom())
                rl = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)
                self.show_action_panel(rl, False)
            except Exception:
                pass

            # サイズラベル／プレビュー更新
            try: self.update_crop_size_label(self._crop_rect_img, img_space=True)
            except Exception: pass
            try:
                if hasattr(self, "safe_update_preview"):
                    QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect_img))
                else:
                    self._schedule_preview(self._crop_rect_img)
            except Exception: pass

            # 見た目同期
            try: lbl.refresh_edit_ui()
            except Exception:
                try: lbl.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                except Exception: pass
                lbl.update()

            try:
                self.label.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            except Exception:
                pass
            
            # --- メニュー「カスタムサイズを使う」を UIだけ OFF（INIは触らない） ---
            act = getattr(self, "custom_toggle_action", None)
            if act and act.isChecked():
                with QSignalBlocker(act):   # シグナルループ防止
                    act.setChecked(False)
            
    def _hide_action_panel(self):
        """通常選択が消えた/左ドラッグを始めた等で、アクションパネルを隠す。
        固定枠モード中は選択は残すので何もしない。
        Nudge は埋め込み廃止に伴い、非固定時のみ“一時的に”隠す。
        """
        # ① Nudge（オーバーレイ版）は、非固定のときだけ一時非表示
        if not getattr(self.label, "fixed_crop_mode", False):
            if hasattr(self, "_suspend_nudge_overlay"):
                try:
                    self._suspend_nudge_overlay(True)  # 位置は保持して隠す
                except Exception:
                    pass

        # ② 固定枠モード中は何もしない（ActionPanel/Nudgeとも残す）
        if getattr(self.label, "fixed_crop_mode", False):
            return

        # ③ 非固定：ActionPanel を閉じ、内部状態とサイズラベルをリセット
        if getattr(self, "_action_panel", None):
            try:
                self._action_panel.close()
            except Exception:
                pass
            self._action_panel = None

        self._crop_rect = None
        self._crop_rect_img = None
        try:
            self.update_crop_size_label(None)
        except Exception:
            pass
    
    def toggle_adjust_mode(self):
        self.set_adjust_mode(not bool(getattr(self, "_adjust_mode", False)))

    def update_custom_edit_action_text(self):
        """編集メニューのラベルに現在のカスタムサイズを表示（例: '... (1024×768)' ）"""
        if hasattr(self, "custom_edit_action"):
            if self.custom_size and all(isinstance(v, int) and v > 0 for v in self.custom_size):
                w, h = self.custom_size
                self.custom_edit_action.setText(f"カスタムサイズを編集... ({w}×{h})")
            else:
                self.custom_edit_action.setText("カスタムサイズを編集...")

    def _unset_fixed_crop_ui(self):
        # プリセット（QActionGroup）を全解除
        try:
            excl = self.crop_action_group.isExclusive()
            self.crop_action_group.setExclusive(False)
            for act in self.crop_action_group.actions():
                act.setChecked(False)
            self.crop_action_group.setExclusive(excl)
        except Exception:
            pass
        # カスタムサイズのトグルもOFF
        try:
            self.custom_toggle_action.setChecked(False)
        except Exception:
            pass

    def on_custom_toggle(self, checked: bool):
        if checked:
            if not self.custom_size:
                # まだサイズなし → 編集ダイアログへ。適用は on_custom_edit 側で行う
                self._custom_edit_in_toggle = True
                self.on_custom_edit()
                self._custom_edit_in_toggle = False

                if not self.custom_size:
                    # 入力されなかった → ON を取り消す
                    blocker = QtCore.QSignalBlocker(self.custom_toggle_action)
                    self.custom_toggle_action.setChecked(False)
                    del blocker
                    return

                # 入力があった場合は on_custom_edit が適用まで実施済み
                return

            # 既にサイズあり → ここで適用
            if hasattr(self, "crop_actions"):
                for act in self.crop_actions.values():
                    act.setChecked(False)
            self.fixed_crop_triggered(self.custom_size, True)
        else:
            # OFF：固定枠解除＆パネルを隠す
            if hasattr(self.label, "clear_fixed_crop"):
                self.label.clear_fixed_crop()
            if hasattr(self, "_hide_action_panel"):
                self._hide_action_panel()
            self.update_crop_size_label(None)
            self._set_preview_placeholder()

    def on_custom_edit(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("カスタムサイズを編集")
        layout = QtWidgets.QVBoxLayout(dialog)

        # このダイアログだけダークに
        dialog.setStyleSheet("""
        QDialog { background: #121212; color: #e0e0e0; }
        QLabel  { color: #e0e0e0; }

        /* 入力欄 */
        QLineEdit {
            background: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            selection-background-color: #2d6cdf;  /* 選択時のハイライト */
        }
        QLineEdit:focus { border: 1px solid #5a9bff; }
        /* プレースホルダ（Qtが対応していれば有効） */
        QLineEdit::placeholder { color: #9aa0a6; }

        /* OK / Cancel ボタン */
        QDialogButtonBox QPushButton {
            background: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            padding: 4px 10px;
        }
        QDialogButtonBox QPushButton:hover  { background: #333333; }
        QDialogButtonBox QPushButton:pressed{ background: #222222; }
        """)

        # タイトルバーもダーク化：
        QtCore.QTimer.singleShot(0, lambda: _enable_dark_titlebar(dialog))

        width_edit = QtWidgets.QLineEdit()
        height_edit = QtWidgets.QLineEdit()
        width_edit.setPlaceholderText("幅 (例: 1024)")
        height_edit.setPlaceholderText("高さ (例: 768)")

        if self.custom_size:
            try:
                w, h = self.custom_size
                width_edit.setText(str(w)); height_edit.setText(str(h))
            except Exception:
                pass

        layout.addWidget(QtWidgets.QLabel("幅:"));    layout.addWidget(width_edit)
        layout.addWidget(QtWidgets.QLabel("高さ:"));  layout.addWidget(height_edit)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(btn_box)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                w = int(width_edit.text()); h = int(height_edit.text())
                if w > 0 and h > 0:
                    self.custom_size = (w, h)
                    self.update_custom_edit_action_text()

                    if self.custom_toggle_action.isChecked():
                        # すでにON：ここでサイズを即反映（編集メニューから or トグル経由）
                        if hasattr(self, "crop_actions"):
                            for act in self.crop_actions.values():
                                act.setChecked(False)
                        self.fixed_crop_triggered(self.custom_size, True)
                    else:
                        # OFF → ON に切り替え（適用は on_custom_toggle 側で実施）
                        self.custom_toggle_action.setChecked(True)
            except Exception:
                pass
        # Cancel：何もしない
      
    def cancel_crop(self):
        # --- 1) 独立Nudge(オーバーレイ)は閉じる ---
        try:
            self.close_nudge_overlay()   # 参照も中で None になる実装にしてあるはず
        except Exception:
            pass

        # --- 2) 比率固定はOFFにして状態をクリア ---
        try:
            self.set_aspect_lock(False)
        except Exception:
            # 念のため直接フラグを落としておく
            try:
                self.label._aspect_lock = False
                self.label._aspect_base_wh = None
                self.label._aspect_ratio = None
            except Exception:
                pass

        # --- 3) 調整モードは終了（ラベル側にも反映）---
        try:
            self.label.set_adjust_mode(False)
        except Exception:
            try:
                setattr(self.label, "adjust_mode", False)
            except Exception:
                pass

        # --- 4) アクションパネルを閉じる ---
        if getattr(self, "_action_panel", None):
            try:
                self._action_panel.close()
            except Exception:
                pass
            self._action_panel = None

        # --- 5) 選択の可視状態をクリア（固定/通常どちらも）---
        try:
            self.label.clear_rubberBand()   # 既存メソッド
        except Exception:
            pass
        try:
            self.label.clear_fixed_crop()
        except Exception:
            # 直接フィールドを掃除（保険）
            try:
                self.label.fixed_crop_mode = False
                self.label.fixed_crop_rect_img = None
            except Exception:
                pass

        # --- 6) プリセットやカスタムのUIをリセット ---
        if hasattr(self, "crop_actions"):
            for act in self.crop_actions.values():
                try:
                    act.setChecked(False)
                except Exception:
                    pass
        if getattr(self, "custom_action", None):
            try:
                if self.custom_action.isChecked():
                    self.custom_action.setChecked(False)
            except Exception:
                pass

        # --- 7) 内部保持・表示のリセット ---
        self._crop_rect = None
        self._crop_rect_img = None
        try:
            self.update_crop_size_label(None)   # "0 x 0" に戻す
        except Exception:
            pass
        try:
            self._set_preview_placeholder()     # プレビュー初期化
        except Exception:
            pass

        # --- 8) 見た目/フォーカスの整え ---
        try:
            self.label.refresh_edit_ui()
        except Exception:
            try:
                self.label.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            except Exception:
                pass
            self.label.update()

        # 矢印キー操作をすぐ再開できるようにフォーカスを戻す
        try:
            self.label.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

    def fixed_crop_triggered(self, size_tuple, from_toggle: bool = False):
        current = self.label.fixed_crop_mode and self.label.fixed_crop_size == size_tuple
        if current:
            self.label.clear_fixed_crop()
            self._crop_rect = None
            if self._action_panel:
                self._action_panel.close()
                self._action_panel = None
            for act in self.crop_actions.values():
                act.setChecked(False)
            if hasattr(self, "custom_action"):
                self.custom_action.setChecked(False)
            return

        # --- それ以外は普通にON ---
        if size_tuple not in self.crop_actions:
            self.custom_action.setChecked(True)
            for act in self.crop_actions.values():
                act.setChecked(False)
        else:
            # 通常のプリセット
            self.custom_action.setChecked(False)
            for k, act in self.crop_actions.items():
                act.setChecked(k == size_tuple)
        # 枠を有効化
        if self.label.pixmap():
            self.label.start_fixed_crop(size_tuple)
            rect = self.label.fixed_crop_rect
            if rect:
                # 追加: 画像座標の矩形も保持して即更新
                if self.label.fixed_crop_rect_img is not None:
                    self._crop_rect_img = QtCore.QRect(self.label.fixed_crop_rect_img)
                    self.update_crop_size_label(self._crop_rect_img, img_space=True)
                    self._schedule_preview(self._crop_rect_img)

                pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                pos.setX(min(pos.x(), self.label.width() - 60))
                pos.setY(min(pos.y(), self.label.height() - 40))
                self._crop_rect = rect
                self.show_action_panel(rect, pos)

    def safe_update_preview(self, crop_rect):
        try:
            if not self or not hasattr(self, "image") or self.image is None \
               or not hasattr(self, "label") or self.label is None or self.label.pixmap() is None:
                if hasattr(self, "preview_label"):
                    self._set_preview_placeholder()
                return
            self.update_preview(crop_rect)
        except Exception as e:
            log_debug("[SAFE PREVIEW ERROR]", e)
            import traceback
            log_debug(traceback.format_exc())
            if hasattr(self, "preview_label"):
                self._set_preview_placeholder()

    def toggle_action_panel(self):
        """
        アクションパネルの表示/非表示をトグル。
        それに連動してオーバーレイの微調整パネルも一緒に隠す／再表示する。
        """
        is_fixed = bool(getattr(self.label, "fixed_crop_mode", False) and
                        getattr(self.label, "fixed_crop_rect_img", None) is not None)
        if is_fixed:
            if hasattr(self, "_current_fixed_label_rect"):
                rect = self._current_fixed_label_rect()
            else:
                rimg = self.label.fixed_crop_rect_img
                x1, y1 = self.label.image_to_label_coords(rimg.left(),  rimg.top())
                x2, y2 = self.label.image_to_label_coords(rimg.right(), rimg.bottom())
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2-x1)+1, abs(y2-y1)+1)
        else:
            rect = getattr(self, "_crop_rect", None)

        panel = getattr(self, "_action_panel", None)

        # まだパネルが無い → ユーザー操作で「出す」ので、フラグはリセットして表示
        if panel is None:
            if rect:
                # ユーザーが明示的に再表示したのでフラグは下げる
                self._panel_hidden_by_user = False

                pos = self._compute_action_panel_pos(rect, is_fixed)
                self.show_action_panel(rect, pos)
                if bool(getattr(self.label, "adjust_mode", False)):
                    # 初回は通常ルートで生成
                    self.open_nudge_overlay()
            return

        overlay = getattr(self, "_nudge_overlay", None)

        if panel.isVisible():
            # ==== 非表示側（中ボタンで隠す） ====
            panel.hide()
            if overlay:
                overlay.hide()

            # ★ 中ボタントグルなどで「今はパネル要らない」状態になった
            self._panel_hidden_by_user = True

            # ★ このタイミングで「隠したときの矩形（画像座標）」を記録しておく
            try:
                r_img = getattr(self, "_crop_rect_img", None)
                if isinstance(r_img, QtCore.QRect):
                    self._panel_hide_rect_img = QtCore.QRect(r_img)
                else:
                    self._panel_hide_rect_img = None
            except Exception:
                self._panel_hide_rect_img = None

        else:
            # ==== 再表示側 ====

            # アクションパネル自体は、ユーザーが最後に動かした位置を優先
            panel.show()
            panel.raise_()

            # ユーザーが再度トグルしたので「要らない」状態を解除
            self._panel_hidden_by_user = False

            # --- Nudge の扱い ---
            if bool(getattr(self.label, "adjust_mode", False)):
                if overlay:
                    # ★ 矩形が「隠している間に動いたかどうか」を判定
                    moved = False
                    try:
                        cur_img = getattr(self, "_crop_rect_img", None)
                        prev_img = getattr(self, "_panel_hide_rect_img", None)
                        if isinstance(cur_img, QtCore.QRect) and isinstance(prev_img, QtCore.QRect):
                            moved = (cur_img != prev_img)
                    except Exception:
                        moved = False

                    # 矩形が動いていたら → ActionPanel の近くにスナップし直す
                    # 動いていなければ → 位置はいじらず、そのまま再表示
                    if moved:
                        try:
                            gap = getattr(self, "nudge_gap_px", 4)
                            if hasattr(self, "_position_nudge_overlay_above_action"):
                                self._position_nudge_overlay_above_action(overlay, gap)
                        except Exception:
                            pass

                    overlay.show()
                    overlay.raise_()
                else:
                    # まだ一度も作っていなければ新規作成（このときだけ位置決めする）
                    self.open_nudge_overlay()

    def _snapshot_adjust_state(self) -> dict:
        """調整/パネル/固定枠/比率固定 の簡易スナップショット"""
        return {
            "adjust": bool(getattr(self, "_adjust_mode", False) or getattr(self.label, "adjust_mode", False)),
            "panel_visible": bool(self._action_panel and self._action_panel.isVisible()),
            "fixed": bool(self.label.fixed_crop_mode and (self.label.fixed_crop_rect_img is not None)),
            # 追加：比率固定の保持
            "aspect_lock": bool(getattr(self.label, "_aspect_lock", False)),
            "aspect_base": getattr(self.label, "_aspect_base_wh", None),
            "nudge": bool(getattr(self, "_nudge_overlay", None) and self._nudge_overlay.isVisible()),
        }

    def _restore_adjust_state(self, snap: dict | None) -> None:
        """画像切替後にUI状態を復元（比率固定→調整→パネルの順）"""
        if not snap:
            return

        # ① 比率固定の復元（先にやる：後続UIの同期に必要）
        try:
            aspect_on = bool(snap.get("aspect_lock", False))
            # set_aspect_lock は内部で NudgePanel 側のUIも同期してくれる実装
            # （基準矩形が無ければ "基準: --" になるが、ロックONのまま維持される）
            if hasattr(self, "set_aspect_lock"):
                self.set_aspect_lock(aspect_on)

            # 基準 (W×H) を覚えていた場合は可能な範囲で戻す
            base = snap.get("aspect_base", None)
            if aspect_on and base and isinstance(base, (tuple, list)) and len(base) == 2:
                self.label._aspect_base_wh = (int(base[0]), int(base[1]))
                try:
                    self.label._aspect_ratio = self.label._aspect_base_wh[0] / self.label._aspect_base_wh[1]
                except Exception:
                    self.label._aspect_ratio = None
                # NudgePanel の表示（"基準: W×H"）も更新
                panel = getattr(self, "_nudge_panel", None)
                if panel and hasattr(panel, "update_aspect_base"):
                    panel.update_aspect_base(self.label._aspect_base_wh)
        except Exception:
            pass

        # ② 調整モード
        try:
            self.set_adjust_mode(bool(snap.get("adjust", False)))
        except Exception:
            pass

        # ③ パネル復帰（現在の画像に矩形がある場合のみ）
        try:
            is_fixed = bool(self.label.fixed_crop_mode and (self.label.fixed_crop_rect_img is not None))
            r = self._current_fixed_label_rect() if is_fixed else getattr(self, "_crop_rect", None)
            if r and snap.get("panel_visible", False):
                pos = self._compute_action_panel_pos(r, is_fixed)
                self.show_action_panel(r, pos)
        except Exception:
            pass

    def move_progress_widget(self):
        # QStatusBarを使う場合は手動で位置を動かす必要なし
        pass
    
    def update_crop_size_label(self, rect=None, img_space=False):
        """解像度ラベル更新。
        rect が渡されたらそれを優先。
        - img_space=True なら rect は画像ピクセル座標の矩形
        - False なら rect はラベル座標なので画像座標に変換
        何も来なければ fixed_crop_rect_img → drag_rect_img の順で拾う。
        """
        if not getattr(self, "image", None):
            self.crop_size_label.setText("0 x 0")
            return

        img_w, img_h = self.image.width, self.image.height

        # 1) 画像座標の矩形を決定
        img_rect = None
        if rect is not None:
            if img_space:
                img_rect = rect
            else:
                # ラベル座標 → 画像座標に変換（両隅）
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.left() + rect.width(), rect.top() + rect.height()
                gx1, gy1 = self.label.label_to_image_coords(x1, y1)
                gx2, gy2 = self.label.label_to_image_coords(x2, y2)
                img_rect = QtCore.QRect(min(gx1, gx2), min(gy1, gy2), abs(gx2 - gx1), abs(gy2 - gy1))
        elif getattr(self.label, "fixed_crop_rect_img", None) is not None:
            img_rect = self.label.fixed_crop_rect_img
        elif getattr(self.label, "drag_rect_img", None) is not None:
            img_rect = self.label.drag_rect_img
        else:
            self.crop_size_label.setText("0 x 0")
            return

        # 2) 保存処理と同じ正規化＆クリップ
        x = int(img_rect.left())
        y = int(img_rect.top())
        w = int(img_rect.width())
        h = int(img_rect.height())
        x1, y1, x2, y2 = x, y, x + w, y + h

        left   = max(0, min(x1, x2))
        top    = max(0, min(y1, y2))
        right  = min(img_w, max(x1, x2))
        bottom = min(img_h, max(y1, y2))

        crop_w = max(0, right - left)
        crop_h = max(0, bottom - top)

        # ★ どちらかが 0 なら完全に非選択とみなし、見た目を揃える
        if crop_w == 0 or crop_h == 0:
            self.crop_size_label.setText("0 x 0")
            return

        # --- それ以外 ---
        self.crop_size_label.setText(f"{crop_w} x {crop_h}")

    def pan_image(self, dx, dy):
        # ラベルのオフセット値を変更
        if not hasattr(self.label, "_pan_offset_x"):
            self.label._pan_offset_x = 0
        if not hasattr(self.label, "_pan_offset_y"):
            self.label._pan_offset_y = 0

        # --- 拡大してる場合のみパンを有効 ---
        if self.zoom_scale > 1.0:
            self.label._pan_offset_x += dx
            self.label._pan_offset_y += dy

            # 画像が見切れすぎないように範囲制限
            max_x = int((self.base_display_width * self.zoom_scale - self.label.width()) / 2)
            max_y = int((self.base_display_height * self.zoom_scale - self.label.height()) / 2)
            self.label._pan_offset_x = max(-max_x, min(self.label._pan_offset_x, max_x))
            self.label._pan_offset_y = max(-max_y, min(self.label._pan_offset_y, max_y))

            self._request_repaint()

    def _has_selection(self) -> bool:
        return (
            getattr(self.label, "fixed_crop_rect_img", None) is not None or
            getattr(self.label, "drag_rect_img", None) is not None or
            getattr(self, "_crop_rect_img", None) is not None
        )

    def _shortcut_save(self):
        if self._has_selection():
            self.do_crop_save()
        else:
            QtWidgets.QApplication.beep()  # 何も選んでない時は無視
    def _request_repaint(self):
        # 例: 60fps ≒ 16ms
        interval = max(1, int(round(1000 / max(1, int(getattr(self, "_repaint_hz", 60))))))
        if not self._repaint_timer.isActive():
            self._repaint_timer.start(interval)

    def _repaint_now(self):
        self.show_image()

    def _schedule_preview(self, img_rect):
        self._pending_preview_rect = QtCore.QRect(img_rect) if img_rect else None
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _preview_now(self):
        if self._pending_preview_rect is not None:
            self.safe_update_preview(self._pending_preview_rect)
            self._pending_preview_rect = None
        else:
            # 直近の更新が無ければ止めて待機（無駄に回さない）
            self._preview_timer.stop()

    def _read_color_setting(self, key: str, default_hex: str) -> QtGui.QColor:
        s = self.settings.value(key, type=str)
        if isinstance(s, str) and s:
            c = QtGui.QColor(s)
            if c.isValid():
                return c
        return QtGui.QColor(default_hex)
 
    def _load_color(self, key: str, default_hex: str) -> QtGui.QColor:
        s = self.settings.value(key, type=str)
        c = QtGui.QColor(s) if s else QtGui.QColor(default_hex)
        return c if c.isValid() else QtGui.QColor(default_hex)

    def _save_color(self, key: str, c: QtGui.QColor):
        self.settings.setValue(key, c.name(QtGui.QColor.NameFormat.HexArgb))
        self.settings.sync()

    def set_view_bg_color(self, c: QtGui.QColor):
        """画像表示領域の背景だけを更新＆保存（チップは触らない）"""
        self.view_bg_color = QtGui.QColor(c)
        self._apply_view_bg()
        # 設定保存＋カスタム色の保存
        self.settings.setValue(
            "view_bg_color",
            self.view_bg_color.name(QtGui.QColor.NameFormat.HexArgb)
        )
        self.settings.sync()
        self._save_custom_colors()

    def set_preview_bg_color(self, c: QtGui.QColor):
        """プレビュー領域の背景だけを更新＆保存（チップは触らない）"""
        self.preview_bg_color = QtGui.QColor(c)
        self._apply_preview_bg_to_label()

        # 選択範囲ありなら再描画、なければプレースホルダを再生成
        if getattr(self, "_crop_rect_img", None):
            self.safe_update_preview(self._crop_rect_img)
        else:
            self._set_preview_placeholder()

        self.settings.setValue(
            "preview_bg_color",
            self.preview_bg_color.name(QtGui.QColor.NameFormat.HexArgb)
        )
        self.settings.sync()
        self._save_custom_colors()

    # === 背景色チップ関連のヘルパ ===

    def _on_view_chip_clicked(self, chip, color: QtGui.QColor):
        """画像表示領域の色チップが左クリックされたとき"""
        self.set_view_bg_color(color)

        chips = (self.view_chip_cur, self.view_chip_q1, self.view_chip_q2)
        selected_index = 0
        for i, w in enumerate(chips):
            is_sel = (w is chip)
            w.set_selected(is_sel)
            if is_sel:
                selected_index = i

        # ★ 選択中インデックスを設定に保存
        try:
            self.settings.setValue("view_chip_selected", selected_index)
        except Exception:
            pass

    def _on_preview_chip_clicked(self, chip, color: QtGui.QColor):
        """プレビュー領域の色チップが左クリックされたとき"""
        self.set_preview_bg_color(color)

        chips = (self.preview_chip_cur, self.preview_chip_q1, self.preview_chip_q2)
        selected_index = 0
        for i, w in enumerate(chips):
            is_sel = (w is chip)
            w.set_selected(is_sel)
            if is_sel:
                selected_index = i

        # ★ 選択中インデックスを設定に保存
        try:
            self.settings.setValue("preview_chip_selected", selected_index)
        except Exception:
            pass

    def _nudge_should_be_visible(self) -> bool:
        # 調整モードON かつ 「自由矩形 or 固定枠」のどちらかがあるときだけ表示
        if not bool(getattr(self, "_adjust_mode", False)):
            return False
        has_free_rect = getattr(self, "_crop_rect_img", None) is not None
        has_fixed_rect = bool(getattr(self.label, "fixed_crop_rect_img", None))
        return has_free_rect or has_fixed_rect

    def ensure_nudge_visibility(self, desired: bool | None = None):
        want = self._nudge_should_be_visible() if desired is None else bool(desired)
        ov = getattr(self, "_nudge_overlay", None)
        if want:
            # ここでだけ開く。ON以外では絶対に開かない
            self.open_nudge_overlay()
        else:
            self.close_nudge_overlay()

    def open_nudge_overlay(self):
        # ← ガード：条件を満たさなければ絶対に出さない
        if not self._nudge_should_be_visible():
            self.close_nudge_overlay()
            return
        ov = getattr(self, "_nudge_overlay", None)
        if ov is None:
            # nudge_cb はあなたの実装に合わせて
            cb = getattr(self, "nudge_edge", None) or (lambda s, d: None)
            ov = MovableNudgePanel(self, cb)
            self._nudge_overlay = ov
        ov.show()
        # 位置合わせ（あるやつを使う）
        try:
            gap = getattr(self, "nudge_gap_px", 4)
            if hasattr(self, "_position_nudge_overlay_above_action"):
                self._position_nudge_overlay_above_action(ov, gap)
        except Exception:
            pass

    def close_nudge_overlay(self):
        ov = getattr(self, "_nudge_overlay", None)
        if ov:
            try:
                ov.close()
            except Exception:
                pass
        self._nudge_overlay = None

    def _suspend_nudge_overlay(self, suspend: bool):
        """ドラッグ中など一時的に隠す/戻す。復帰はゲート経由に統一。"""
        ov = getattr(self, "_nudge_overlay", None)
        if suspend:
            if ov: ov.hide()
        else:
            self.ensure_nudge_visibility()  # ← ここが肝：勝手に出さない

    def on_action_cancel(self):

        # 取消直前の状態を保持（固定モードだったか？）
        was_fixed = bool(getattr(self.label, "fixed_crop_mode", False))

        # 1) 矩形消去（固定/自由の両方）
        try: self.label.clear_rubberBand()
        except Exception: pass
        try: self.label.clear_fixed_crop()
        except Exception: pass
        self._crop_rect_img = None
        self._crop_rect     = None

        # 2) 調整モードは確実にOFF
        try: self.set_adjust_mode(False)
        except Exception: pass
        try: self.label.set_adjust_mode(False)
        except Exception: pass

        # 2.5) 固定モードだった場合はメニュー側のチェックも必ず解除
        if was_fixed:
            try: self._unset_fixed_crop_ui()
            except Exception: pass

        # 3) アクションパネルを閉じる
        try: self._hide_action_panel()
        except Exception: pass

        # 4) 微調整パネルは “完全に閉じる”
        try: self.ensure_nudge_visibility(False)  # ← 後述のゲート関数
        except Exception:
            try: self.close_nudge_overlay()
            except Exception: pass

    def _apply_save_folder_programmatically(self, folder: str):
        """ダイアログ無しで保存先を適用し、表示ラベルも更新する"""
        self.save_folder = folder
        try:
            self.set_save_text(f"保存先: {folder}")
        except Exception:
            pass

    def _maybe_prompt_save_on_load(self, src_dir: str):
        # DnD直後は Explorer が DoDragDrop 中 → 次フレームで開く
        if getattr(self, "_defer_save_dialog_once", False):
            self._defer_save_dialog_once = False
            QtCore.QTimer.singleShot(0, lambda: self._maybe_prompt_save_on_load(src_dir))
            return
        """画像/フォルダ読み込み直後に保存先ダイアログを出す"""

        # —— ① 上書き直後の同一パスなら 1回だけスキップ ——
        try:
            cur_path = os.path.normcase(os.path.abspath(getattr(self, "image_path", "") or ""))
            paths = getattr(self, "_suppress_save_dialog_paths", None)
            if paths is None:
                self._suppress_save_dialog_paths = set()
                paths = self._suppress_save_dialog_paths
            if cur_path and cur_path in paths:
                paths.discard(cur_path)  # 一度だけ有効
                return
        except Exception:
            pass

        if not bool(getattr(self, "show_save_dialog_on_load", False)):
            # ダイアログを出さない設定でも、表示だけは実効値に合わせて更新
            self._update_save_folder_label()
            return

        dlg = SaveDestinationDialog(
            self,
            src_dir=src_dir,
            dest_mode=self.save_dest_mode,
            custom_dir=self.save_custom_dir,
            overwrite=bool(self.overwrite_mode),
            show_again=bool(self.show_save_dialog_on_load),
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            # キャンセル時も、表示だけは実効値（=読み込み元等）で更新
            self._update_save_folder_label()
            return

        vals = dlg.values()

        # 保存方法（連番/上書き）
        self.overwrite_mode = bool(vals.get("overwrite", False))
        self.settings.setValue("overwrite_mode", self.overwrite_mode)

        # クイックUIに反映
        self._update_quick_save_mode_radios()

        # 保存先（同じ/別）
        self.save_dest_mode = "custom" if vals.get("dest_mode") == "custom" else "same"
        self.settings.setValue("save_dest_mode", self.save_dest_mode)

        # 実フォルダを適用
        if self.save_dest_mode == "custom":
            # フォルダ指定：ここで固定先を決める
            self.save_custom_dir = vals.get("custom_dir", "") or src_dir
            self.settings.setValue("save_custom_dir", self.save_custom_dir)
            self._apply_save_folder_programmatically(self.save_custom_dir)
        else:
            # 読み込み元と同じ：固定保存先はクリアしておく
            # → 実際の保存時 / 表示は毎回「今開いているフォルダ」から解決
            self.save_custom_dir = ""
            self.save_folder = ""
            self.settings.setValue("save_custom_dir", "")

        # ★ クイックUIの保存先ラジオにも反映
        self._update_quick_save_dest_radios()
        # ★ 右下表示も今の状態で更新
        self._update_save_folder_label()

        # 「今後表示しない」
        self.show_save_dialog_on_load = bool(vals.get("show_again", True))
        self.settings.setValue("show_save_dialog_on_load", self.show_save_dialog_on_load)
        try:
            self.act_show_save_prompt.setChecked(self.show_save_dialog_on_load)
        except Exception:
            pass

    def _get_image_source_dir(self, path: str) -> str:
        """
        保存先=読込み元のときに使う「元フォルダ」。
        通常画像 : 画像があるフォルダ
        zip://～ : zip ファイルが置いてあるフォルダ
        memzip://（zip内zip）: 一番外側の物理zipが置いてあるフォルダ
        """
        if not path:
            return ""

        if is_zip_uri(path):
            try:
                zip_path, inner = parse_zip_uri(path)

                # --- memzip（zip内zip）なら、外側の物理zipまで辿る ---
                if isinstance(zip_path, str) and zip_path.startswith("memzip:"):
                    outer = zip_path
                    # memzip → その outer → さらに memzip なら再帰的に辿る
                    while isinstance(outer, str) and outer.startswith("memzip:"):
                        meta = _MEM_ZIP_META.get(outer)
                        if not meta:
                            break
                        outer = meta.get("outer") or ""
                    if outer:
                        return os.path.dirname(outer)
                    # うまく辿れなかったら空文字（最終的には _effective_save_folder 側でフォールバック）
                    return ""

                # 通常の zip://C:/.../foo.zip!/...
                return os.path.dirname(zip_path)
            except Exception:
                # 失敗したらとりあえず元の dirname にフォールバック
                return os.path.dirname(path)

        # 通常のファイルパス
        return os.path.dirname(path)

    def _output_name_from_image_path(self) -> str:
        path = getattr(self, "image_path", "") or ""
        if not path:
            return "cropped"

        try:
            if is_zip_uri(path):
                zip_path, inner = parse_zip_uri(path)
                inner = inner or ""

                inner_name = Path(inner).name if inner else ""
                log_debug(f"[output_name] zip path={path!r} inner={inner!r} -> inner_name={inner_name!r}")

                if inner_name:
                    return inner_name

                if isinstance(zip_path, str):
                    outer_name = Path(zip_path).name
                    log_debug(f"[output_name] fallback outer_name={outer_name!r}")
                    return outer_name

                outer_name = Path(str(zip_path)).name
                log_debug(f"[output_name] fallback outer_name(str)={outer_name!r}")
                return outer_name

            name = Path(path).name
            log_debug(f"[output_name] normal path={path!r} -> name={name!r}")
            return name

        except Exception as e:
            log_debug(f"[output_name] ERROR: {e} path={path!r}")
            try:
                name = os.path.basename(path) or "cropped"
                log_debug(f"[output_name] fallback basename -> {name!r}")
                return name
            except Exception:
                return "cropped"

    def _effective_save_folder(self) -> str:
        """実効保存先（表示用）。save_folder未指定なら読み込み元 or 現在フォルダを返す。"""
        sf = getattr(self, "save_folder", "") or ""
        if sf:
            try:
                return os.path.normpath(sf)
            except Exception:
                return sf

        # 未指定 → 画像のあるフォルダ（zip:// のときは zip 親フォルダ）、無ければ現在ブラウズ中フォルダ
        if getattr(self, "image_path", None):
            base_dir = self._get_image_source_dir(self.image_path)
            try:
                return os.path.normpath(base_dir)
            except Exception:
                return base_dir or ""

        return getattr(self, "folder", "") or ""

    def _update_save_folder_label(self) -> None:
        """右下の“保存先:”表示を実効値で更新（stateは変えない）"""
        path = self._effective_save_folder()
        try:
            # あなたの既存の表示関数に合わせる
            self.set_save_text(f"保存先: {path if path else '―'}")
        except Exception:
            # 直接ラベルにsetTextしている実装ならそちらで
            if hasattr(self, "save_folder_label"):
                self.save_folder_label.setText(f"保存先: {path if path else '―'}")

    def _folder_icon(self) -> QtGui.QIcon:
        """Qt5/Qt6 両対応でフォルダアイコンを返す"""
        style = QtWidgets.QApplication.style()

        # Qt5: QStyle.SP_DirIcon / Qt6: QStyle.StandardPixmap.SP_DirIcon
        sp = getattr(QtWidgets.QStyle, "SP_DirIcon", None)
        if sp is None and hasattr(QtWidgets.QStyle, "StandardPixmap"):
            sp = getattr(QtWidgets.QStyle.StandardPixmap, "SP_DirIcon", None)

        if sp is not None:
            try:
                return style.standardIcon(sp)
            except Exception:
                pass

        # フォールバック: QFileIconProvider（どの版でも使える）
        try:
            prov = QtWidgets.QFileIconProvider()
            return prov.icon(QtWidgets.QFileIconProvider.IconType.Folder)
        except Exception:
            pass

        # 最終フォールバック（空アイコン）
        return QtGui.QIcon()

    def _file_icon(self, path: str) -> QtGui.QIcon:
        """関連付けに応じたファイルアイコン（zipなど）を返す"""
        try:
            prov = QtWidgets.QFileIconProvider()
            return prov.icon(QtCore.QFileInfo(path))
        except Exception:
            return QtGui.QIcon()

    def _image_icon_for_entry(self, path: str) -> QtGui.QIcon:
        """
        画像用のアイコンを返す。
        - 通常パス: そのまま _file_icon(path)
        - zip:// の中身: inner の拡張子からダミーパスを作って _file_icon(dummy) に渡す
        """
        try:
            if is_zip_uri(path):
                zp, inner = parse_zip_uri(path)
                inner = inner.rstrip("/")

                # inner が空 = zip のルート（ここに来るのは稀だけど一応）
                if not inner:
                    return self._file_icon(zp)

                # inner の拡張子を使ってダミーパス生成
                _, ext = os.path.splitext(inner)
                if ext:
                    dummy = "dummy" + ext  # 例: dummy.png / dummy.jpg
                    return self._file_icon(dummy)

                # 拡張子がない場合は zip 自体 or フォルダアイコンにフォールバック
                return self._file_icon(zp) or self._folder_icon()
            else:
                return self._file_icon(path)
        except Exception:
            try:
                return self._folder_icon()
            except Exception:
                return QtGui.QIcon()

    def _show_folder_placeholder(self, folder_path: str = "", *, force: bool = False):
        """
        メインプレビューにフォルダアイコンを中央表示（画像が無いとき用）
        - オーバーレイ画像は「フォルダ直下」から1枚だけ（サブフォルダは見ない）
        - 直下候補は自然順で安定化。キャッシュして再選択時の揺れを抑える
        - force=True なら候補キャッシュを無視して取り直す
        - ★ 遅延せず“同期”で合成する（確実に表示）
        """
        import os, re

        # 対象フォルダ / zip
        base_dir = folder_path or getattr(self, "folder", "")
        if not base_dir:
            return

        # ★ プレースホルダ表示中は ActionPanel / Nudge を“一時的に”隠す
        #   - ここでは状態(固定枠/矩形)は壊さない
        #   - ユーザーが中ボタン等で隠した flag とは別に扱う
        try:
            self._panel_hidden_by_placeholder = True
            ap = getattr(self, "_action_panel", None)
            if ap and ap.isVisible():
                ap.hide()

            # Nudge はオーバーレイ版なら suspend を優先
            if hasattr(self, "_suspend_nudge_overlay"):
                try:
                    self._suspend_nudge_overlay(True)
                except Exception:
                    pass
            else:
                ov = getattr(self, "_nudge_overlay", None)
                if ov:
                    try:
                        ov.hide()
                    except Exception:
                        pass
        except Exception:
            pass

        # 物理フォルダか zip(ルート/サブフォルダ)か判定
        is_phys_dir = os.path.isdir(base_dir)
        is_zip_root = False            # zip 自体（ルート）を選んでいるか
        is_zip_virtual_dir = False     # zip の中のフォルダを選んでいるか
        zip_file_path = ""

        try:
            if is_zip_uri(base_dir):
                zp, inner = parse_zip_uri(base_dir)
                zip_file_path = zp
                if inner == "" or inner == "/":
                    # zip のルート (zip://...! )
                    is_zip_root = True
                else:
                    # zip 内のサブフォルダ
                    is_zip_virtual_dir = True
            elif is_archive_file(base_dir):
                # 物理 zip ファイルパスをフォルダ扱いしている場合
                zip_file_path = base_dir
                is_zip_root = True
        except Exception:
            pass

        if not is_phys_dir and not is_zip_root and not is_zip_virtual_dir:
            # どれでもなければ従来通り終了
            return

        # ★ トータル計測スタート
        t0_total = _dbg_time(f"[placeholder] {base_dir} TOTAL start")

        # 画像状態クリア（フォルダモードへ）
        self.image = None
        self.image_path = ""
        self.current_index = -1

        # ラベルサイズ
        if not hasattr(self, "label") or not hasattr(self.label, "size"):
            w, h = 320, 240
        else:
            sz = self.label.size()
            w, h = max(32, int(sz.width())), max(32, int(sz.height()))

        # 中身プレビュー用の 1 枚を探す（直下 → 子フォルダを深さ制限付きで再帰）
        cand = None

        if not hasattr(self, "_folder_overlay_cache"):
            self._folder_overlay_cache = {}

        # 物理フォルダと VFS(zip:// や zip ファイルパス) でキーを分ける
        if is_phys_dir:
            cache_key = os.path.normcase(os.path.abspath(base_dir))
        else:
            cache_key = f"VFS::{base_dir}"

        if force:
            self._folder_overlay_cache.pop(cache_key, None)

        # ★ 候補画像探索の計測開始
        t0_find = _dbg_time(f"[placeholder] {base_dir} find_candidate start")

        cand = self._folder_overlay_cache.get(cache_key)
        # 物理パスの場合だけ実ファイルが消えていないかチェック
        if cand and is_phys_dir and not os.path.isfile(cand):
            cand = None

        def first_image_recursive(dir_uri: str, depth: int = 0, max_depth: int = 3) -> str | None:
            """dir_uri 以下から、最初に見つかった画像 1 枚だけを返す（深さ制限付き）"""
            if depth > max_depth:
                return None
            try:
                entries = vfs_listdir(dir_uri)
            except Exception:
                return None

            # 1) まず直下の画像
            files = [
                e["uri"]
                for e in entries
                if not e.get("is_dir") and is_image_name(e.get("uri", ""))
            ]
            if files:
                try:
                    files.sort(key=natural_key)
                except Exception:
                    files.sort()
                return files[0]

            # 2) 無ければサブフォルダを自然順で再帰
            dirs = [e["uri"] for e in entries if e.get("is_dir")]
            if not dirs or depth >= max_depth:
                return None
            try:
                dirs.sort(key=natural_key)
            except Exception:
                dirs.sort()

            for sub in dirs:
                pick = first_image_recursive(sub, depth + 1, max_depth)
                if pick:
                    return pick
            return None

        if not cand:
            cand = first_image_recursive(base_dir, 0, 4)  # ★ 深さ 4 まで
            if cand:
                self._folder_overlay_cache[cache_key] = cand
            else:
                self._folder_overlay_cache.pop(cache_key, None)

        # ★ 候補探索終了
        _dbg_time(f"[placeholder] {base_dir} find_candidate end (cand={cand})", t0_find)

        # キャンバスとレイアウト
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(QtGui.QColor(35, 35, 35))

        base = max(64, min(256, int(min(w, h) * 0.5)))
        icon_w = icon_h = base
        target = QtCore.QRect((w - icon_w) // 2, (h - icon_h) // 2 - 10, icon_w, icon_h)

        # フォルダ / zip のアイコン
        if is_zip_root and zip_file_path:
            icon = self._file_icon(zip_file_path) or self._folder_icon()
        else:
            icon = self._folder_icon()
        folder_pm = icon.pixmap(icon_w, icon_h) if (icon and not icon.isNull()) else QtGui.QPixmap(icon_w, icon_h)
        if folder_pm.isNull():
            folder_pm = QtGui.QPixmap(icon_w, icon_h)
            folder_pm.fill(QtGui.QColor(90, 90, 90))

        # 名前表示用
        name = ""
        try:
            if is_zip_root and zip_file_path:
                name = os.path.basename(zip_file_path)
            else:
                name = os.path.basename(base_dir.rstrip("\\/")) or base_dir
        except Exception:
            name = os.path.basename(str(base_dir))

        if name:
            text_rect = QtCore.QRect(
                0,
                target.bottom() + 8,
                w,
                24,
            )
        else:
            text_rect = QtCore.QRect()

        # ヒット領域（選択枠のために少し広め）
        frame = target.adjusted(-12, -12, 12, 12)
        sel_rect = QtCore.QRect(frame); sel_rect.adjust(-6, -6, 6, 28)
        self._placeholder_hit_rect = QtCore.QRect(sel_rect)
        selected = bool(getattr(self, "_placeholder_selected", False))

        pal = self.palette() if hasattr(self, "palette") else None
        hl = pal.color(QtGui.QPalette.ColorRole.Highlight) if pal else QtGui.QColor("#2d7dff")
        hl_text = pal.color(QtGui.QPalette.ColorRole.HighlightedText) if pal else QtGui.QColor("#ffffff")

        # ★ 合成処理の計測開始
        t0_comp = _dbg_time(f"[placeholder] {base_dir} compose start")

        # ===== 同期で合成 =====
        p = QtGui.QPainter(canvas)
        try:
            rh = getattr(QtGui.QPainter, "Antialiasing", None) \
                or getattr(QtGui.QPainter.RenderHint, "Antialiasing", None)
            if rh is not None:
                p.setRenderHint(rh, True)

            # 反転ハイライト
            if selected:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(hl.red(), hl.green(), hl.blue(), 80))
                p.drawRoundedRect(sel_rect, 12, 12)
                pen = QtGui.QPen(hl); pen.setWidth(2)
                p.setPen(pen); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(sel_rect, 12, 12)

            # フォルダ本体
            p.drawPixmap(target, folder_pm, folder_pm.rect())

            # オーバーレイ（中身プレビュー用の1枚）
            if cand:
                # ★ 画像読み込み計測開始
                t0_img = _dbg_time(f"[placeholder] {base_dir} load_overlay start")

                overlay_img = QtGui.QImage()

                try:
                    # まずはできるだけ軽量に QImage に直接読ませる
                    if is_zip_uri(cand) or (not os.path.isabs(cand) or not os.path.exists(cand)):
                        # zip:// や memzip:// など仮想パス
                        try:
                            data = open_bytes_any(cand)  # zip 内でも temp 展開なしで bytes 取得
                            overlay_img = QtGui.QImage.fromData(data)
                        except Exception:
                            overlay_img = QtGui.QImage()

                        # Qt 側で読めなかったフォーマットだけ、従来の Pillow 経由にフォールバック
                        if overlay_img.isNull():
                            from io import BytesIO
                            try:
                                im = open_image_any(cand).convert("RGBA")
                                bio = BytesIO()
                                im.save(bio, format="PNG")
                                overlay_img = QtGui.QImage.fromData(bio.getvalue(), "PNG")
                            except Exception:
                                overlay_img = QtGui.QImage()
                    else:
                        # 普通のファイルパスは従来通り
                        overlay_img = QtGui.QImage(cand)
                except Exception:
                    overlay_img = QtGui.QImage()

                # ★ 画像読み込み計測終了
                _dbg_time(f"[placeholder] {base_dir} load_overlay end", t0_img)


                if not overlay_img.isNull():
                    side = icon_w

                    # サムネと同じルールの「オーバーレイ専用枠」
                    margin_x      = int(side * 0.12)
                    margin_top    = int(side * 0.32)
                    margin_bottom = int(side * 0.22)

                    overlay_rect = QtCore.QRect(
                        target.left() + margin_x,
                        target.top()  + margin_top,
                        side - margin_x * 2,
                        max(1, side - margin_top - margin_bottom),
                    )

                    iw, ih = overlay_img.width(), overlay_img.height()
                    if iw > 0 and ih > 0 and overlay_rect.width() > 0 and overlay_rect.height() > 0:
                        s = min(overlay_rect.width() / iw, overlay_rect.height() / ih)
                        dw, dh = max(1, int(iw * s)), max(1, int(ih * s))

                        dst = QtCore.QRect(
                            overlay_rect.center().x() - dw // 2,
                            overlay_rect.center().y() - dh // 2,
                            dw, dh,
                        )

                        clip = QtGui.QPainterPath()
                        r = max(2, int(min(dw, dh) * 0.06))
                        clip.addRoundedRect(QtCore.QRectF(dst), r, r)
                        p.save()
                        p.setClipPath(clip)
                        p.drawImage(dst, overlay_img)
                        p.restore()

            # テキスト
            if name and h >= 64:
                f = QtGui.QFont(); f.setPointSize(10); p.setFont(f)
                p.setPen(hl_text if selected else QtGui.QColor(200, 200, 200, 210))
                p.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignHCenter, name)
        finally:
            p.end()

        # ★ 合成処理終了
        _dbg_time(f"[placeholder] {base_dir} compose end", t0_comp)

        # 反映
        if hasattr(self.label, "setPixmap"):
            self.label.setPixmap(canvas)

        # プレースホルダ状態ON（矢印カーソル）
        self._placeholder_active = True
        self._placeholder_path = base_dir
        try:
            self.label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        except Exception:
            pass

        # ★ トータル計測終了
        _dbg_time(f"[placeholder] {base_dir} TOTAL end", t0_total)

class _ShiftArrowGuard(QtCore.QObject):
    """テキスト入力中は Shift+←/→ を通常の選択操作に譲るためのガード"""
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.ShortcutOverride:
            try:
                if (ev.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right) and
                    (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)):
                    fw = QtWidgets.QApplication.focusWidget()
                    if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
                        # ここで受理すると QShortcut には渡らない（＝テキスト選択が優先される）
                        ev.accept()
                        return True
            except Exception:
                pass
        return super().eventFilter(obj, ev)

class ClickableProgressBar(QtWidgets.QProgressBar):
    clickedValueChanged = QtCore.pyqtSignal(int)  # 新しい値がクリックで決まった時にemit

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            x = event.position().x() 
            width = self.width()
            minval = self.minimum()
            maxval = self.maximum()
            if maxval > minval:
                percent = min(max(x / width, 0), 1)
                value = int(round(percent * (maxval - minval) + minval))
                self.setValue(value)
                self.clickedValueChanged.emit(value)
        super().mousePressEvent(event)

class ProgressWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.count_label = QtWidgets.QLabel("55 / 57")
        layout.addWidget(self.count_label)
        self.progress_bar = ClickableProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(57)
        self.progress_bar.setValue(55)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar, stretch=1)
        self.percent_label = QtWidgets.QLabel("96%")
        layout.addWidget(self.percent_label)

        self.progress_bar.clickedValueChanged.connect(self.on_jump)

    def set_progress(self, current, total):
        self.count_label.setText(f"{current} / {total}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        percent = int(round(current / total * 100)) if total else 0
        self.percent_label.setText(f"{percent}%")

    def on_jump(self, value):
        self.set_progress(value, self.progress_bar.maximum())

class _TipPopup(QtWidgets.QFrame):
    """控えめで柔らかいチップツール（フェード/タイマーなし・はみ出しゼロ）"""
    def __init__(self, parent=None):
        super().__init__(parent,
            QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)

        # 親QSSの影響は遮断
        self.setObjectName("SoftTip")
        self.setStyleSheet(
            "#SoftTip { border: none; background: transparent; }"
            "#SoftTip QLabel { border: none; background: transparent; }"
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # クリック等を奪わない・アクティブにしない・透過背景
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # ラベル
        self._lab = QtWidgets.QLabel(self)
        self._lab.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._lab.setWordWrap(False)  # 折り返し無しでスッキリ
        # 少し小さめ＆軽め
        f = self._lab.font()
        try: f.setPointSizeF(max(9.0, f.pointSizeF() - 1))
        except Exception: f.setPointSize(max(9, f.pointSize() - 1))
        self._lab.setFont(f)
        self._lab.setStyleSheet("QLabel{color:#f1f1f1; background:transparent;}")

        # 余白は控えめ
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.addWidget(self._lab)

        # 見た目パラメータ（柔らかめ）
        self._radius = 10
        self._bg = QtGui.QColor(18, 18, 18, 190)       # 黒に近いグレー、やや透過
        self._border = QtGui.QColor(255, 255, 255, 22) # ごく薄い縁
        self._border_w = 1
        self._max_w = 360

        # 点滅防止のための前回状態
        self._last_text = ""
        self._last_pos = QtCore.QPoint(-9999, -9999)

    # 角のはみ出し防止
    def _rounded_path(self) -> QtGui.QPainterPath:
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        return path

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        region = QtGui.QRegion(self._rounded_path().toFillPolygon().toPolygon())
        self.setMask(region)

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        # ほのかな内側シャドウ（控えめ）
        for i, a in enumerate((28, 16)):
            r = rect.adjusted(2+i, 2+i, -(2+i), -(2+i))
            path = QtGui.QPainterPath(); path.addRoundedRect(r, self._radius-1, self._radius-1)
            p.fillPath(path, QtGui.QColor(0, 0, 0, a))

        # 背景
        path = self._rounded_path()
        p.fillPath(path, self._bg)

        # 薄い縁
        pen = QtGui.QPen(self._border, self._border_w); pen.setCosmetic(True)
        p.setPen(pen); p.drawPath(path)

    def show_text(self, text: str, global_pos: QtCore.QPoint):
        """フェード/タイマーなし。位置追従のみ。"""
        if not text:
            self.hide()
            return

        # サイズ反映
        self._lab.setMaximumWidth(self._max_w)
        self._lab.setText(text)
        self.adjustSize()
        if self.width() < 1 or self.height() < 1:
            return

        # 画面内クランプ＋整数座標
        scr = QtWidgets.QApplication.screenAt(global_pos) or QtWidgets.QApplication.primaryScreen()
        geo = scr.availableGeometry() if scr else QtCore.QRect(0,0,1920,1080)
        x = min(max(global_pos.x() + 12, geo.left()),  geo.right()  - self.width())
        y = min(max(global_pos.y() + 16, geo.top()),   geo.bottom() - self.height())
        new_pos = QtCore.QPoint(int(round(x)), int(round(y)))

        # 既に表示中で「文面同じ＆位置がほぼ同じ」なら位置だけ更新（点滅防止）
        if self.isVisible() and text == self._last_text and \
           (new_pos - self._last_pos).manhattanLength() <= 6:
            self.move(new_pos)
        else:
            self.move(new_pos)
            if not self.isVisible():
                self.show()
                self.raise_()

        self._last_text = text
        self._last_pos = new_pos

    def hide(self):
        super().hide()
        # 次回を“新規表示扱い”に
        self._last_text = ""
        self._last_pos = QtCore.QPoint(-9999, -9999)

class _SuccessToast(QtWidgets.QFrame):
    """最前面の小さな Success トースト（常に前面）"""
    def __init__(self, parent=None):
        super().__init__(parent,
            QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._lab = QtWidgets.QLabel(self)
        self._lab.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lab.setStyleSheet("QLabel{color:#ffffff; background:transparent; font-weight:bold;}")

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addWidget(self._lab)

        self._radius = 10
        self._bg = QtGui.QColor(34, 139, 34, 220)   # やや透過のグリーン
        self._border = QtGui.QColor(255, 255, 255, 30)

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QtGui.QPainterPath(); path.addRoundedRect(r, self._radius, self._radius)
        p.fillPath(path, self._bg)
        pen = QtGui.QPen(self._border, 1); pen.setCosmetic(True); p.setPen(pen); p.drawPath(path)

    def show_text(self, text: str, anchor_global: QtCore.QPoint):
        self._lab.setText(text or "Success")
        self.adjustSize()
        # ちょい右下に出す＆画面内クランプ
        scr = QtWidgets.QApplication.screenAt(anchor_global) or QtWidgets.QApplication.primaryScreen()
        geo = scr.availableGeometry() if scr else QtCore.QRect(0,0,1920,1080)
        x = min(max(anchor_global.x() + 12, geo.left()),  geo.right()  - self.width())
        y = min(max(anchor_global.y() + 12, geo.top()),   geo.bottom() - self.height())
        self.move(int(x), int(y))
        self.show()
        self.raise_()
 
if __name__ == "__main__":
    # ★ コマンドライン引数に --debug-log があればログ有効化
    if "--debug-log" in sys.argv:
        LOG_ENABLED = True
        sys.argv.remove("--debug-log")  # Qt に渡さないように消しておく

    app = QtWidgets.QApplication(sys.argv)
    win = CropperApp()
    win.show()
    sys.exit(app.exec())
