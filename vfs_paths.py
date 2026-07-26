"""Pure path and URI helpers for the virtual filesystem."""


# 物理フォルダで「フォルダ扱い」したい拡張子
# ※ .cbr も実質 rar ベースなので一緒に対応しておく
ARCHIVE_FILE_EXTS = {".zip", ".cbz", ".rar", ".cbr", ".7z", ".7zip"}

# zip 内 zip (memzip) として特別扱いする拡張子（従来どおり zip/cbz のみ）
ARCHIVE_EMBED_EXTS = {".zip", ".cbz"}


def _ext(s: str) -> str:
    s = (s or "").lower()
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if s.endswith(ext):
            return ext
    import os
    return os.path.splitext(s)[1]


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
