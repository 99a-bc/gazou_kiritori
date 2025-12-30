# background_removal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Type
import gc
import os
from pathlib import Path

from PIL import Image

# ============================================================
# Debug print helper
# ============================================================

_BG_DEBUG = (
    os.environ.get("GAZOU_BG_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    or os.environ.get("GAZOU_KIRITORI_BG_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
)

def set_bg_debug(flag: bool) -> None:
    """このモジュール内のデバッグprintをON/OFFする。"""
    global _BG_DEBUG
    _BG_DEBUG = bool(flag)

def _dbg(*args, **kwargs) -> None:
    """debug時だけprintする。"""
    if _BG_DEBUG:
        print(*args, **kwargs)

def _dbg_cuda_mem(tag: str) -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        rsv = torch.cuda.memory_reserved() / (1024 ** 2)
        maxa = torch.cuda.max_memory_allocated() / (1024 ** 2)
        _dbg(f"[BG][CUDA] {tag} alloc={alloc:.1f}MiB reserved={rsv:.1f}MiB max_alloc={maxa:.1f}MiB")
    except Exception as e:
        _dbg(f"[BG][CUDA] mem stat failed: {e}")

# ============================================================
# Model Info
# ============================================================

@dataclass(frozen=True)
class BgModelInfo:
    key: str
    label: str
    kind: str  # backend kind id (e.g. "bria_rmbg")
    params: Dict[str, Any] | None = None

    # 既存コードが dict っぽく扱っても壊れないように互換メソッド
    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "params": dict(self.params or {}),
        }


# ★ ここに「モデル定義」を足すだけでUIに追加できる想定
_SUPPORTED_MODELS: Dict[str, BgModelInfo] = {
    "bria_rmbg_1_4": BgModelInfo(
        key="bria_rmbg_1_4",
        label="BRIA RMBG v1.4",
        kind="bria_rmbg",
        params={
            "repo": "briaai/RMBG-1.4",
            "pipeline": "v1",
            "local_files_only": True,  # ★一度DLしたら固定（ネットに見に行かない）
        },
    ),
    "bria_rmbg_2_0": BgModelInfo(
        key="bria_rmbg_2_0",
        label="BRIA RMBG v2.0（要アクセス許可/初回DL）",
        kind="bria_rmbg",
        params={
            "repo": "briaai/RMBG-2.0",   # オフライン運用するならここをローカルパスにしてもOK
            "pipeline": "v2",
            "local_files_only": True,  # ★一度DLしたら固定（ネットに見に行かない）
        },
    ),
}

def get_available_bg_models() -> Dict[str, BgModelInfo]:
    return dict(_SUPPORTED_MODELS)

# ============================================================
# Offline availability helpers (no network)
# ============================================================

def _dedup_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        k = str(rp)
        if k in seen:
            continue
        seen.add(k)
        out.append(rp)
    return out


def _candidate_hf_cache_dirs() -> list[Path]:
    """
    Hugging Face Hub のキャッシュ候補ディレクトリを列挙（存在しなくても返す）。
    このアプリでは、HF_HOME が設定されている場合は
    そちらの hub ディレクトリだけを見るようにして、
    ユーザー全体のグローバルキャッシュは無視する。
    """
    cand: list[Path] = []

    # まず、このアプリ用に設定している HF_HOME を最優先で見る
    v = os.environ.get("HF_HOME")
    if v:
        cand.append(Path(v) / "hub")
        cand = _dedup_paths(cand)

        # デバッグ用ログ
        _dbg("[BGDBG] HF cache candidates (HF_HOME only):")
        for base in cand:
            _dbg("  ", base)

        return cand

    # HF_HOME が無い場合だけ、従来どおりの推測ロジックを使う
    # 最優先：明示指定
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(key)
        if v:
            cand.append(Path(v))

    # TRANSFORMERS_CACHE が指定されてるケース（環境によって hub 直や transformers 直がある）
    v = os.environ.get("TRANSFORMERS_CACHE")
    if v:
        p = Path(v)
        cand.append(p)
        cand.append(p / "hub")

    # デフォルト（Windows でもここになりがち）
    cand.append(Path.home() / ".cache" / "huggingface" / "hub")

    cand = _dedup_paths(cand)

    _dbg("[BGDBG] HF cache candidates (fallback):")
    for base in cand:
        _dbg("  ", base)

    return cand

def _repo_id_to_cache_dirname(repo_id: str) -> str:
    """
    "briaai/RMBG-2.0" -> "models--briaai--RMBG-2.0"
    """
    repo_id = repo_id.strip().replace("\\", "/")
    if "/" in repo_id:
        org, name = repo_id.split("/", 1)
        return f"models--{org}--{name}"
    # 変な値が来た時の保険（ローカルパス等は別処理で弾く）
    return "models--" + repo_id.replace("/", "--")


def _has_any_snapshot(model_dir: Path) -> bool:
    snap = model_dir / "snapshots"
    if not snap.is_dir():
        return False
    try:
        for x in snap.iterdir():
            if x.is_dir():
                return True
    except Exception:
        return False
    return False


def is_hf_repo_cached(repo_or_path: str) -> bool:
    """
    repo_id もしくはローカルパスについて、
    “オフラインでロードできそうな実体があるか” を判定。
    """
    if not repo_or_path:
        return False

    p = Path(repo_or_path)

    # ローカルパス扱い（絶対・相対どちらも）
    # 例: repo="models/RMBG-2.0" のようにしている場合
    if p.exists():
        # ディレクトリならそのままOK / ファイルでもOK（ケースによるが“存在する”を優先）
        return True

    # repo_id 扱い（Hugging Face Hub のキャッシュを探す）
    dirname = _repo_id_to_cache_dirname(repo_or_path)
    for base in _candidate_hf_cache_dirs():
        model_dir = base / dirname
        if _has_any_snapshot(model_dir):
            return True

    return False


def is_model_cached(model_key: str) -> bool:
    info = _SUPPORTED_MODELS.get(model_key)
    if not info or not info.params:
        return False
    repo = str(info.params.get("repo", "")).strip()
    if not repo:
        return False

    _dbg("[BGDBG] is_model_cached:", model_key, "repo=", repo)

    ok = is_hf_repo_cached(repo)
    _dbg("[BGDBG]   ->", ok)
    return ok

def list_cached_models() -> list[str]:
    cached: list[str] = []
    for k in _SUPPORTED_MODELS.keys():
        try:
            if is_model_cached(k):
                cached.append(k)
        except Exception:
            pass
    return cached

# ============================================================
# Backend Base
# ============================================================

class _BackendBase:
    def __init__(self, device: str = "cpu", *, debug: bool = False, **_kwargs):
        self.device = device
        self.debug = debug

    def load(self) -> None:
        raise NotImplementedError

    def remove(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """重いリソース（GPUテンソル等）を解放するためのフック"""
        return

def _auto_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ============================================================
# Backend: BRIA RMBG (v1.4 / v2.0 共通)
# ============================================================

class _BriaRmbgBackend(_BackendBase):
    """
    params:
      repo: str  (e.g. "briaai/RMBG-2.0" or local path "models/RMBG-2.0")
      pipeline: "v1" or "v2"
      local_files_only: bool (optional)  # オフライン強制したい時用
    """
    def __init__(
        self,
        device: str = "cpu",
        *,
        repo: str,
        pipeline: str = "v2",
        local_files_only: bool = False,
        debug: bool = False,
        **_kwargs,
    ):
        super().__init__(device, debug=debug)
        self.repo = repo
        self.pipeline = pipeline
        self.local_files_only = local_files_only

        self.model = None
        self._torch = None
        self._np = None

    def load(self) -> None:
        try:
            import torch
            import numpy as np
            from transformers import AutoModelForImageSegmentation
        except Exception as e:
            raise RuntimeError(
                "[BG] import 失敗: torch / transformers / numpy が必要です。"
            ) from e

        # ★ RMBG-2.0(v2) が要求しがちな追加依存を先にチェック（原因を分かりやすく）
        if str(self.pipeline).lower() == "v2":
            try:
                import timm  # noqa: F401
                import kornia  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "[BG] RMBG-2.0 の実行には追加ライブラリが必要です: timm / kornia\n"
                    "以下を venv で実行してください:\n"
                    "  python -m pip install -U timm kornia\n"
                ) from e

        self._torch = torch
        self._np = np

        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.repo,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            # ★ 元の例外内容も表示して原因追跡できるようにする
            raise RuntimeError(
                f"[BG] {self.repo} のロードに失敗しました。\n"
                "- RMBG-2.0 は Hugging Face 側でアクセス許可/ログインが必要な場合があります\n"
                "- オフライン運用なら repo をローカルフォルダにして local_files_only=True を使ってください\n"
                f"\n[元のエラー]\n{type(e).__name__}: {e}"
            ) from e

        model.to(self.device)
        model.eval()
        self.model = model

    def _unwrap_pred(self, obj):
        """
        出力が list/tuple/dict/ModelOutput などで来ても Tensor に寄せる。

        重要:
        - RMBG v1.4 (pipeline="v1") は「先頭」を取らないと外れを引くことがある
        - RMBG v2 系 (pipeline="v2") は「最後」の方が高解像度寄りなことがある
        """
        pipeline = getattr(self, "pipeline", "v1")
        prefer_last = (pipeline == "v2")

        # ModelOutputっぽい属性
        for attr in ("logits", "preds", "pred", "mask", "masks"):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
                break

        # list/tuple の入れ子
        while isinstance(obj, (list, tuple)) and len(obj) > 0:
            obj = obj[-1] if prefer_last else obj[0]
            for attr in ("logits", "preds", "pred", "mask", "masks"):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                    break

        # dict
        if isinstance(obj, dict) and len(obj) > 0:
            obj = next(iter(obj.values()))
        return obj

    def _preprocess_v1(self, image: Image.Image):
        torch = self._torch
        np = self._np

        img = image.convert("RGB")
        img = img.resize((1024, 1024), Image.BILINEAR)

        np_img = np.array(img).astype(np.float32) / 255.0  # (H,W,C)
        im_tensor = torch.tensor(np_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        return im_tensor.to(self.device)


    # ---------- v2 pipeline (RMBG-2.0 推奨) ----------
    def _preprocess_v2(self, image: Image.Image):
        try:
            from torchvision import transforms
        except Exception as e:
            raise RuntimeError("[BG] torchvision が必要です（RMBG-2.0 推奨パイプライン）。") from e

        img = image.convert("RGB")
        tfm = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        return tfm(img).unsqueeze(0).to(self.device)

    def _squeeze_pred_to_hw(self, pred):
        torch = self._torch

        pred = self._unwrap_pred(pred)
        if pred is None:
            raise RuntimeError("[BG] モデル出力が None でした（予期しない出力形式）")

        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)

        pred = pred.detach().float().cpu()

        # (H,W) に落とす
        if pred.ndim == 4:
            # (B,C,H,W) -> (H,W)
            pred = pred[0, 0]
        elif pred.ndim == 3:
            # (B,H,W) or (C,H,W) 想定
            if pred.shape[0] in (1, 3, 4):
                pred = pred[0]
            else:
                pred = pred[0]
        elif pred.ndim == 2:
            pass
        else:
            raise ValueError(f"[BG] Unexpected pred shape: {tuple(pred.shape)}")

        return pred

    def _pred_to_mask_L(self, pred, orig_size):
        torch = self._torch
        np = self._np

        pred_hw = self._squeeze_pred_to_hw(pred)

        # 値域で「logitsか確率か」を判定して安定化
        vmin = float(pred_hw.min().item())
        vmax = float(pred_hw.max().item())

        if self.debug:
            _dbg(f"[BG] pred range: min={vmin:.6f}, max={vmax:.6f}, shape={tuple(pred_hw.shape)}")

        if vmax > 1.5 or vmin < -0.5:
            prob = torch.sigmoid(pred_hw)
        else:
            prob = pred_hw.clamp(0.0, 1.0)

        p = prob.numpy().astype(np.float32)
        mask = Image.fromarray((p * 255.0).astype(np.uint8), mode="L")
        mask = mask.resize(orig_size, Image.BILINEAR)
        return mask

    def _infer_v1_mask(self, image: Image.Image) -> Image.Image:
        torch = self._torch
        inp = self._preprocess_v1(image)

        with torch.no_grad():
            out = self.model(inp)

        return self._pred_to_mask_L(out, image.size)

    def _infer_v2_mask(self, image: Image.Image) -> Image.Image:
        torch = self._torch
        inputs = self._preprocess_v2(image)  # ← 今の実装だと Tensor が返る想定

        with torch.no_grad():
            # inputs が dict の場合にも、Tensor の場合にも対応
            if isinstance(inputs, dict):
                out = self.model(**inputs)
            else:
                try:
                    out = self.model(inputs)
                except TypeError:
                    # モデルによっては pixel_values を要求することがある
                    out = self.model(pixel_values=inputs)

        return self._pred_to_mask_L(out, image.size)

    def remove(self, image: Image.Image) -> Image.Image:
        if self.model is None:
            self.load()

        if self.pipeline == "v2":
            mask = self._infer_v2_mask(image)
        elif self.pipeline == "v1":
            mask = self._infer_v1_mask(image)
        else:
            raise ValueError(f"[BG] Unknown pipeline: {self.pipeline}")

        out = image.convert("RGBA")
        out.putalpha(mask)
        return out

    def close(self) -> None:
        try:
            _dbg_cuda_mem("before backend close")
            if hasattr(self, "model") and self.model is not None:
                try:
                    # 念のためCPUへ逃がしてから参照を切る
                    self.model.to("cpu")
                except Exception:
                    pass
                self.model = None
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            _dbg_cuda_mem("after backend close")
        except Exception as e:
            _dbg(f"[BG] backend close failed: {e}")

# ============================================================
# Backend Registry (登録制)
# ============================================================

_BACKEND_REGISTRY: Dict[str, Type[_BackendBase]] = {
    "bria_rmbg": _BriaRmbgBackend,
    # ここに追加していく： "u2net": _U2NetBackend, など
}


# ============================================================
# Manager
# ============================================================

class BackgroundRemovalManager:
    """
    - set_model(key) でモデル切替
    - remove_background(pil) で RGBA を返す
    """
    def __init__(
        self,
        model_key: str = "bria_rmbg_1_4",
        device: Optional[str] = None,
        *,
        debug: bool = False,
    ):
        self._model_key = model_key
        self._device = device
        self._debug = debug
        set_bg_debug(debug)

        self._backend: Optional[_BackendBase] = None
        self._last_error: Optional[Exception] = None

    def set_model(self, model_key: str) -> None:
        if model_key not in _SUPPORTED_MODELS:
            raise KeyError(f"Unknown bg model key: {model_key}")

        # 同じなら何もしない
        if model_key == self._model_key:
            return

        old = self._model_key

        # ★ ここが重要：旧モデル(backend)を確実に破棄してVRAMを空ける
        self._dispose_backend(reason=f"switch {old} -> {model_key}")

        # 新モデルに切り替え（実体のロードは遅延初期化のまま）
        self._model_key = model_key
        self._backend = None
        self._last_error = None

    def get_current_model_info(self) -> BgModelInfo:
        return _SUPPORTED_MODELS.get(
            self._model_key,
            BgModelInfo(self._model_key, self._model_key, "unknown", {}),
        )

    def get_last_error(self) -> Optional[Exception]:
        return self._last_error

    def _ensure_backend(self) -> _BackendBase:
        if self._backend is not None:
            return self._backend

        info = self.get_current_model_info()
        cls = _BACKEND_REGISTRY.get(info.kind)
        if cls is None:
            raise RuntimeError(f"Unsupported bg backend kind: {info.kind}")

        params = dict(info.params or {})
        dev = _auto_device(self._device)

        try:
            self._backend = cls(device=dev, debug=self._debug, **params)
        except Exception as e:
            self._last_error = e
            raise

        return self._backend

    def remove_background(self, image: Image.Image) -> Image.Image:
        try:
            backend = self._ensure_backend()
            return backend.remove(image)
        except Exception as e:
            self._last_error = e
            raise

    # 互換用
    def remove(self, image: Image.Image) -> Image.Image:
        return self.remove_background(image)

    def _dispose_backend(self, reason: str = "") -> None:
        if self._backend is None:
            _dbg(f"[BG] dispose skipped (backend None): key={self._model_key} reason={reason}")
            return

        _dbg(f"[BG] dispose backend: key={self._model_key} reason={reason} backend={type(self._backend)}")
        try:
            import torch
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated() / (1024**2)
                r = torch.cuda.memory_reserved() / (1024**2)
                _dbg(f"[BG] cuda mem before: alloc={a:.1f}MiB reserved={r:.1f}MiB")
        except Exception:
            pass

        try:
            # close() があれば呼ぶ
            close = getattr(self._backend, "close", None)
            if callable(close):
                close()
        except Exception as e:
            _dbg(f"[BG] backend.close() failed: {e!r}")

        self._backend = None

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                a = torch.cuda.memory_allocated() / (1024**2)
                r = torch.cuda.memory_reserved() / (1024**2)
                _dbg(f"[BG] cuda mem after : alloc={a:.1f}MiB reserved={r:.1f}MiB")
        except Exception:
            pass


