"""CPU-only preparation and cache scoring for the v3 spectrogram route.

This command deliberately has no encoder/device surface.  It can freeze the
route registry, render already prepared W=240 windows, or score a separately
created :class:`DynamicTokenCache` using the two mandatory representation
arms.  Model forward belongs to a later isolated encoder job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from measure_vit4ts.reducers import stitch_column_vectors
from measure_vit4ts.ritp import stitch_native_240

from .core import (
    apply_temporal_operator,
    build_candidate_mask,
    build_linear_nctp,
    column_top_fraction_mean,
    fuse_scale_subset,
    harmonic_incidence_projection,
    literal_incidence,
    released_incidence,
    streamed_median_reference_match,
)
from .dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    DynamicTokenCache,
    cache_digest,
    load_dynamic_cache,
)
from .spectrogram_registry import (
    ARMS,
    FULL_ARM,
    REGISTRY_ID,
    REL_ARM,
    SpectrogramRoute,
    build_spectrogram_registry,
    load_spectrogram_route,
    make_spectrogram_cache_key,
    renderer_identity,
    validate_spectrogram_cache_key,
)
from .spectrogram_renderer import (
    SCHEMA_VERSION as RENDERER_SCHEMA_VERSION,
    array_sha256,
    render_spectrogram_windows,
    renderer_config_sha256,
    renderer_source_sha256,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/vittrace_ablation_full_v3.yaml")
RENDER_FILE = "spectrogram_images.npy"
RENDER_MANIFEST = "spectrogram_render_manifest.json"
REGISTRY_FILE = "spectrogram_arm_registry.json"
ROUTE_MANIFEST = "spectrogram_route_manifest.json"
STATUS_FILE = "spectrogram_scores_status.json"
SOURCE_FILES = (
    "spectrogram_renderer.py",
    "spectrogram_registry.py",
    "spectrogram_runner.py",
)
_SAFE_SERIES = re.compile(r"^[A-Za-z0-9_.-]+$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def spectrogram_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        path = root / name
        payload = path.read_bytes()
        digest.update(name.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


@dataclass(frozen=True)
class ConfigBundle:
    path: Path
    payload: dict[str, Any]
    file_sha256: str
    route: SpectrogramRoute


def load_config(path: Path) -> ConfigBundle:
    config_path = Path(path).resolve()
    raw = config_path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("v3 config must be a mapping")
    route = load_spectrogram_route(payload)
    return ConfigBundle(config_path, payload, _sha256_bytes(raw), route)


def _route_manifest_payload(bundle: ConfigBundle) -> dict[str, Any]:
    route = bundle.route
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "config_path": str(bundle.path),
        "config_file_sha256": bundle.file_sha256,
        "route_config_sha256": route.route_config_sha256,
        "renderer_source_sha256": renderer_source_sha256(),
        "renderer_config_sha256": renderer_config_sha256(route.spec),
        "spectrogram_source_sha256": spectrogram_source_sha256(),
        "variant": route.variant.to_payload(),
        "spectrogram": route.spec.to_payload(),
        "renderer_identity": renderer_identity(route),
        "dynamic_cache_schema": {
            "cache_file": CACHE_FILE,
            "cache_manifest": CACHE_MANIFEST,
            "key_type": "DynamicCacheKey",
        },
        "arms": list(ARMS),
        "scoring": {
            REL_ARM: {
                "matching": "global_median_reference",
                "incidence": "released",
                "fusion": "legacy_intersection",
                "temporal": "legacy_top25_column",
            },
            FULL_ARM: {
                "matching": "global_median_reference",
                "incidence": "literal",
                "fusion": "active_valid",
                "temporal": "nctp_linear_native240",
            },
        },
        "encoder_calls": 0,
        "model_forward": False,
    }


def freeze_route(bundle: ConfigBundle, output_directory: Path) -> tuple[Path, Path]:
    root = Path(output_directory)
    registry_path = root / REGISTRY_FILE
    route_path = root / ROUTE_MANIFEST
    _atomic_json(registry_path, build_spectrogram_registry().to_payload())
    _atomic_json(route_path, _route_manifest_payload(bundle))
    return registry_path, route_path


def render_windows_file(
    bundle: ConfigBundle,
    windows_path: Path,
    output_directory: Path,
    *,
    series_id: str,
    data_sha256: str,
    model_sha256: str | None = None,
) -> Path:
    if not _SAFE_SERIES.fullmatch(str(series_id)):
        raise ValueError("series_id is not a safe cache identifier")
    windows_file = Path(windows_path).resolve()
    windows = np.load(windows_file, allow_pickle=False)
    render = render_spectrogram_windows(windows, bundle.route.spec)
    model_hash = str(
        model_sha256 or bundle.payload["vendor"]["default_model_sha256"]
    ).upper()
    key = make_spectrogram_cache_key(
        bundle.route,
        render,
        series_id=series_id,
        data_sha256=data_sha256,
        model_sha256=model_hash,
    )
    root = Path(output_directory)
    images_path = root / RENDER_FILE
    images_file_sha = _atomic_npy(images_path, render.images)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "renderer_schema_version": RENDERER_SCHEMA_VERSION,
        "series_id": series_id,
        "config_file_sha256": bundle.file_sha256,
        "route_config_sha256": bundle.route.route_config_sha256,
        "renderer_source_sha256": render.renderer_source_sha256,
        "renderer_config_sha256": render.renderer_config_sha256,
        "image_array_sha256": render.image_array_sha256,
        "image_file_sha256": images_file_sha,
        "image_shape": list(render.images.shape),
        "image_dtype": str(render.images.dtype),
        "stft_shape": list(render.stft_shape),
        "renderer_sha256": render.renderer_sha256,
        "windows_path": str(windows_file),
        "windows_file_sha256": sha256_file(windows_file),
        "images_path": str(images_path.resolve()),
        "cache_key": {**asdict(key), "image_size": list(key.image_size)},
        "cache_digest": cache_digest(key),
        "encoder_calls": 0,
        "model_forward": False,
    }
    manifest_path = root / RENDER_MANIFEST
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _dynamic_key_from_manifest(cache_directory: Path) -> DynamicCacheKey:
    payload = json.loads(
        (Path(cache_directory) / CACHE_MANIFEST).read_text(encoding="utf-8")
    )
    raw = dict(payload["key"])
    raw["image_size"] = tuple(map(int, raw["image_size"]))
    return DynamicCacheKey(**raw)


def _validate_render_manifest(
    path: Path,
    bundle: ConfigBundle,
    key: DynamicCacheKey,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key_payload = dict(payload.get("cache_key", {}))
    if "image_size" in key_payload:
        key_payload["image_size"] = tuple(map(int, key_payload["image_size"]))
    manifest_key = DynamicCacheKey(**key_payload)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "config_file_sha256": bundle.file_sha256,
        "route_config_sha256": bundle.route.route_config_sha256,
        "renderer_source_sha256": renderer_source_sha256(),
        "renderer_config_sha256": renderer_config_sha256(bundle.route.spec),
        "renderer_sha256": key.renderer_sha256,
        "cache_digest": cache_digest(key),
        "encoder_calls": 0,
        "model_forward": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"render manifest {name} changed")
    if manifest_key != key:
        raise ValueError("render manifest and dynamic cache keys differ")
    return payload


def _match_scale(tokens: np.ndarray, grid: tuple[int, int]) -> torch.Tensor:
    values = torch.from_numpy(np.ascontiguousarray(tokens))
    candidates = build_candidate_mask(grid, "global", device="cpu")
    return streamed_median_reference_match(
        values, candidates, query_chunk_size=32
    ).cost


def _project_fields(
    cache: DynamicTokenCache,
    matches: Mapping[str, torch.Tensor],
    *,
    literal: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cells = int(cache.patch_grid[0] * cache.patch_grid[1])
    fields: dict[str, torch.Tensor] = {"P": matches["P"]}
    valid: dict[str, torch.Tensor] = {
        "P": torch.ones(cells, dtype=torch.bool)
    }
    for scale, mask in (("M", cache.mid_mask), ("L", cache.large_mask)):
        incidence = (
            literal_incidence(mask, cache.patch_grid)
            if literal
            else released_incidence(mask, cache.patch_grid)
        )
        fields[scale], valid[scale] = harmonic_incidence_projection(
            matches[scale], incidence
        )
    return fields, valid


def _legacy_fuse(
    fields: Mapping[str, torch.Tensor],
    valid: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    fused = sum(fields[name].to(torch.float64) for name in ("P", "M", "L")) / 3.0
    intersection = valid["P"] & valid["M"] & valid["L"]
    return torch.where(intersection.unsqueeze(0), fused, torch.zeros_like(fused))


def _legacy_score(
    fused: torch.Tensor,
    *,
    full_length: int,
    interpolation_chunk: int = 64,
) -> np.ndarray:
    windows = int(fused.shape[0])
    values = fused.reshape(windows, 1, 14, 14)
    vectors: list[np.ndarray] = []
    for start in range(0, windows, interpolation_chunk):
        maps = (
            F.interpolate(
                values[start : start + interpolation_chunk],
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .numpy()
        )
        vectors.append(column_top_fraction_mean(maps, 0.25))
    return np.ascontiguousarray(
        stitch_column_vectors(
            np.concatenate(vectors, axis=0), full_length, 240, 60
        ),
        dtype=np.float64,
    )


def score_spectrogram_cache(
    cache: DynamicTokenCache,
    *,
    full_length: int,
) -> dict[str, np.ndarray]:
    """Compute the registered REL/FULL pair from one spectrogram cache."""

    length = int(full_length)
    if length < 240:
        raise ValueError("full_length must be at least 240")
    windows = int(cache.patch_tokens.shape[0])
    expected_windows = 1 + (length - 240) // 60
    if windows != expected_windows:
        raise ValueError("dynamic cache window count differs from W240/S60 coverage")
    if cache.patch_grid != (14, 14):
        raise ValueError("B16 spectrogram cache must have a 14x14 patch grid")
    starts = np.arange(windows, dtype=np.int64) * 60
    matches = {
        "P": _match_scale(cache.patch_tokens, (14, 14)),
        "M": _match_scale(cache.mid_tokens, (13, 13)),
        "L": _match_scale(cache.large_tokens, (12, 12)),
    }

    rel_fields, rel_valid = _project_fields(cache, matches, literal=False)
    rel = _legacy_score(
        _legacy_fuse(rel_fields, rel_valid), full_length=length
    )

    full_fields, full_valid = _project_fields(cache, matches, literal=True)
    full_fused = fuse_scale_subset(
        full_fields, ("P", "M", "L"), valid_masks=full_valid
    )
    local = apply_temporal_operator(
        build_linear_nctp(240, cache.patch_grid),
        np.ascontiguousarray(full_fused.numpy(), dtype=np.float64),
    )
    full = np.ascontiguousarray(
        stitch_native_240(local, starts, length), dtype=np.float64
    )
    output = {REL_ARM: rel, FULL_ARM: full}
    for arm, score in output.items():
        if score.shape != (length,) or score.dtype != np.float64:
            raise RuntimeError(f"{arm} did not produce canonical float64 [T]")
        if not np.isfinite(score).all():
            raise RuntimeError(f"{arm} produced non-finite scores")
    return output


def score_cache_directory(
    bundle: ConfigBundle,
    cache_directory: Path,
    render_manifest: Path,
    output_directory: Path,
    *,
    full_length: int,
) -> Path:
    cache_dir = Path(cache_directory).resolve()
    key = _dynamic_key_from_manifest(cache_dir)
    validate_spectrogram_cache_key(bundle.route, key)
    configured_model_sha = str(bundle.payload["vendor"]["default_model_sha256"]).upper()
    if key.model_sha256.upper() != configured_model_sha:
        raise ValueError("spectrogram cache model SHA256 differs from frozen B16")
    render_payload = _validate_render_manifest(render_manifest, bundle, key)
    cache = load_dynamic_cache(cache_dir, key)
    scores = score_spectrogram_cache(cache, full_length=full_length)

    root = Path(output_directory)
    registry_path, route_path = freeze_route(bundle, root)
    common = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "config_file_sha256": bundle.file_sha256,
        "route_config_sha256": bundle.route.route_config_sha256,
        "spectrogram_source_sha256": spectrogram_source_sha256(),
        "renderer_source_sha256": renderer_source_sha256(),
        "renderer_config_sha256": renderer_config_sha256(bundle.route.spec),
        "renderer_sha256": key.renderer_sha256,
        "cache_dir": str(cache_dir),
        "cache_digest": cache_digest(key),
        "cache_file_sha256": sha256_file(cache_dir / CACHE_FILE),
        "cache_manifest_sha256": sha256_file(cache_dir / CACHE_MANIFEST),
        "render_manifest_path": str(Path(render_manifest).resolve()),
        "render_manifest_sha256": sha256_file(render_manifest),
        "image_array_sha256": render_payload["image_array_sha256"],
        "registry_sha256": sha256_file(registry_path),
        "route_manifest_sha256": sha256_file(route_path),
        "full_length": int(full_length),
        "window_count": int(cache.patch_tokens.shape[0]),
        "encoder_calls": 0,
        "model_forward": False,
    }
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = root / arm
        score_path = arm_root / "score.npy"
        score_file_sha = _atomic_npy(score_path, scores[arm])
        manifest_path = arm_root / "score_manifest.json"
        payload = {
            **common,
            "arm": arm,
            "score_path": str(score_path.resolve()),
            "score_file_sha256": score_file_sha,
            "score_array_sha256": array_sha256(scores[arm]),
            "score_shape": list(scores[arm].shape),
            "score_dtype": str(scores[arm].dtype),
        }
        _atomic_json(manifest_path, payload)
        rows.append(
            {
                "arm": arm,
                "status": "PASS",
                "score_path": str(score_path.resolve()),
                "score_file_sha256": score_file_sha,
                "score_manifest_sha256": sha256_file(manifest_path),
            }
        )
    status_path = root / STATUS_FILE
    _atomic_json(
        status_path,
        {
            **common,
            "status": "COMPLETE",
            "expected_arms": list(ARMS),
            "completed_arms": len(rows),
            "rows": rows,
        },
    )
    return status_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-registry")
    freeze.add_argument("--output-dir", type=Path, required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--windows", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--series-id", required=True)
    render.add_argument("--data-sha256", required=True)
    render.add_argument("--model-sha256")

    score = subparsers.add_parser("score-cache")
    score.add_argument("--cache-dir", type=Path, required=True)
    score.add_argument("--render-manifest", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--full-length", type=int, required=True)

    args = parser.parse_args(argv)
    bundle = load_config(args.config)
    if args.command == "freeze-registry":
        registry, route = freeze_route(bundle, args.output_dir)
        print(json.dumps({"registry": str(registry), "route": str(route)}))
    elif args.command == "render":
        path = render_windows_file(
            bundle,
            args.windows,
            args.output_dir,
            series_id=args.series_id,
            data_sha256=args.data_sha256,
            model_sha256=args.model_sha256,
        )
        print(path)
    else:
        print(
            score_cache_directory(
                bundle,
                args.cache_dir,
                args.render_manifest,
                args.output_dir,
                full_length=args.full_length,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ConfigBundle",
    "freeze_route",
    "load_config",
    "main",
    "render_windows_file",
    "score_cache_directory",
    "score_spectrogram_cache",
    "sha256_file",
    "spectrogram_source_sha256",
]
