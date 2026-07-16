"""Read-only, SHA-guarded access to the frozen PaAno baseline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable

import numpy as np


class VendorMismatchError(RuntimeError):
    """Raised before use when the PaAno checkout is not the frozen revision."""


@dataclass(frozen=True, slots=True)
class VendorFingerprint:
    root: Path
    git_sha: str
    dirty: bool
    required_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class VendorSymbols:
    PatchEncoder: type
    basic_metricor: type
    generate_curve: Callable[..., Any]
    find_length_rank: Callable[..., Any]
    fingerprint: VendorFingerprint


_REQUIRED_RELATIVE = (
    Path("model.py"),
    Path("utils/metrics.py"),
    Path("utils/basic_metrics.py"),
    Path("utils/data_preprocess.py"),
)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VendorMismatchError(f"cannot inspect vendor checkout {root}: {exc}") from exc
    return result.stdout.strip()


def verify_vendor_repo(vendor_root: Path, expected_sha: str) -> VendorFingerprint:
    root = Path(vendor_root).resolve(strict=True)
    actual_sha = _git(root, "rev-parse", "HEAD")
    if actual_sha != expected_sha:
        raise VendorMismatchError(
            f"PaAno SHA mismatch: expected {expected_sha}, observed {actual_sha}"
        )
    if len(actual_sha) != 40:
        raise VendorMismatchError("vendor HEAD is not a full 40-character Git SHA")
    required = tuple(root / relative for relative in _REQUIRED_RELATIVE)
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise VendorMismatchError(f"vendor required files are missing: {missing}")
    dirty = bool(_git(root, "status", "--porcelain"))
    return VendorFingerprint(root=root, git_sha=actual_sha, dirty=dirty, required_files=required)


def _assert_module_under(module: ModuleType, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(root):
        raise VendorMismatchError(f"import {module.__name__} did not resolve under {root}")


def load_vendor_symbols(vendor_root: Path, expected_sha: str) -> VendorSymbols:
    fingerprint = verify_vendor_repo(vendor_root, expected_sha)
    root_text = str(fingerprint.root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    model_module = importlib.import_module("model")
    metrics_module = importlib.import_module("utils.metrics")
    basic_module = importlib.import_module("utils.basic_metrics")
    preprocess_module = importlib.import_module("utils.data_preprocess")
    for module in (model_module, metrics_module, basic_module, preprocess_module):
        _assert_module_under(module, fingerprint.root)

    patch_encoder = getattr(model_module, "PatchEncoder", None)
    metricor = getattr(basic_module, "basic_metricor", None)
    generate_curve = getattr(basic_module, "generate_curve", None)
    find_length_rank = getattr(preprocess_module, "find_length_rank", None)
    if not inspect.isclass(patch_encoder) or not inspect.isclass(metricor):
        raise VendorMismatchError("PaAno encoder/metric class surface changed")
    if not callable(generate_curve) or not callable(find_length_rank):
        raise VendorMismatchError("PaAno metric/window call surface changed")
    encoder_params = inspect.signature(patch_encoder).parameters
    if "in_channels" not in encoder_params or "use_revin" not in encoder_params:
        raise VendorMismatchError("PatchEncoder constructor surface changed")
    return VendorSymbols(
        PatchEncoder=patch_encoder,
        basic_metricor=metricor,
        generate_curve=generate_curve,
        find_length_rank=find_length_rank,
        fingerprint=fingerprint,
    )


def build_encoder(
    symbols: VendorSymbols,
    channels: int,
    use_revin: bool,
    device: Any,
) -> Any:
    if channels <= 0:
        raise ValueError("channels must be positive")
    model = symbols.PatchEncoder(in_channels=channels, use_revin=use_revin)
    return model.to(device)


def compute_baseline_window(
    symbols: VendorSymbols, x_first_channel: np.ndarray
) -> int:
    values = np.asarray(x_first_channel, dtype=np.float32)
    if values.ndim not in (1, 2) or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("window input must be a finite non-empty first-channel series")
    window = int(symbols.find_length_rank(values.squeeze(), rank=1))
    if window <= 0:
        raise VendorMismatchError(f"vendor returned invalid sliding window {window}")
    return window

