"""Frozen registry and cache identity for the ViTTrace v3 spectrogram route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .dynamic_cache import DynamicCacheKey, cache_digest
from .registry import ArmRegistry, ArmSpec, ContrastSpec, validate_arm_registry
from .spectrogram_renderer import (
    SpectrogramRenderBatch,
    SpectrogramSpec,
    renderer_config_sha256,
    spec_from_config,
)


SCHEMA_VERSION = 1
EXPECTED_STAGE = "vittrace_ablation_full_v3"
REGISTRY_ID = "VITTRACE_V3_VIS_SPECTROGRAM_B16_W240_S60"
REL_ARM = "VIS_SPECTROGRAM_REL"
FULL_ARM = "VIS_SPECTROGRAM_FULL"
ARMS = (REL_ARM, FULL_ARM)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


@dataclass(frozen=True)
class SpectrogramVariant:
    representation: str = "spectrogram"
    model_key: str = "B16"
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    image_size: tuple[int, int] = (224, 224)
    patch_size: int = 16
    window: int = 240
    stride: int = 60
    batch_size: int = 64

    def __post_init__(self) -> None:
        expected = {
            "representation": "spectrogram",
            "model_key": "B16",
            "model_name": "ViT-B-16",
            "pretrained": "openai",
            "image_size": (224, 224),
            "patch_size": 16,
            "window": 240,
            "stride": 60,
            "batch_size": 64,
        }
        actual = asdict(self)
        actual["image_size"] = tuple(actual["image_size"])
        if actual != expected:
            raise ValueError("spectrogram route is mandatory B16/W240/S60/batch64")

    def to_payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_size"] = list(self.image_size)
        return result


@dataclass(frozen=True)
class SpectrogramRoute:
    spec: SpectrogramSpec
    variant: SpectrogramVariant
    route_config_sha256: str


def build_spectrogram_registry() -> ArmRegistry:
    registry = ArmRegistry(
        registry_id=REGISTRY_ID,
        primary_arm=FULL_ARM,
        control_arm=REL_ARM,
        arms=(
            ArmSpec(REL_ARM, "representation_control_released", 0, 0.0),
            ArmSpec(FULL_ARM, "representation_full", 1, 0.0),
        ),
        contrasts=(
            ContrastSpec(
                "VIS_SPECTROGRAM_FULL_VS_REL",
                "VISUAL_REPRESENTATION",
                FULL_ARM,
                REL_ARM,
            ),
        ),
    )
    return validate_arm_registry(registry.to_payload())


def route_config_sha256(
    spec: SpectrogramSpec,
    variant: SpectrogramVariant,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "arms": list(ARMS),
        "spectrogram": spec.to_payload(),
        "variant": variant.to_payload(),
    }
    return _sha256(_canonical_json(payload))


def load_spectrogram_route(config: Mapping[str, Any]) -> SpectrogramRoute:
    if not isinstance(config, Mapping) or config.get("stage") != EXPECTED_STAGE:
        raise ValueError("spectrogram route requires the isolated v3 config")
    defaults = config.get("defaults")
    grid = config.get("grid")
    runtime = config.get("runtime")
    if not all(isinstance(value, Mapping) for value in (defaults, grid, runtime)):
        raise ValueError("v3 defaults/grid/runtime blocks are required")
    if tuple(map(int, defaults.get("image_size", ()))) != (224, 224):
        raise ValueError("spectrogram route is frozen to 224x224")
    if "spectrogram" not in tuple(map(str, grid.get("representations", ()))):
        raise ValueError("spectrogram representation is not registered")
    if 240 not in tuple(map(int, grid.get("windows", ()))):
        raise ValueError("W240 is missing from the representation grid")
    if 60 not in tuple(map(int, grid.get("strides_w240", ()))):
        raise ValueError("S60 is missing from the W240 stride grid")
    if int(runtime.get("batch_size", -1)) != 64:
        raise ValueError("spectrogram route is frozen to encoder batch size 64")

    backbones = grid.get("backbones")
    if not isinstance(backbones, list):
        raise ValueError("backbone registry must be a list")
    rows = [row for row in backbones if isinstance(row, Mapping) and row.get("key") == "B16"]
    if len(rows) != 1 or dict(rows[0]) != {
        "key": "B16",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "patch_size": 16,
    }:
        raise ValueError("B16 backbone provenance differs from the mandatory route")
    spec = spec_from_config(config.get("spectrogram"))
    variant = SpectrogramVariant()
    return SpectrogramRoute(spec, variant, route_config_sha256(spec, variant))


def renderer_identity(route: SpectrogramRoute) -> str:
    digest = renderer_config_sha256(route.spec)[:16].lower()
    return f"spectrogram.{digest}.batch{route.variant.batch_size}"


def make_spectrogram_cache_key(
    route: SpectrogramRoute,
    render: SpectrogramRenderBatch,
    *,
    series_id: str,
    data_sha256: str,
    model_sha256: str,
) -> DynamicCacheKey:
    if render.renderer_config_sha256 != renderer_config_sha256(route.spec):
        raise ValueError("render batch does not match the frozen spectrogram config")
    key = DynamicCacheKey(
        series_id=str(series_id),
        data_sha256=str(data_sha256).upper(),
        renderer=renderer_identity(route),
        renderer_sha256=render.renderer_sha256,
        model_name=route.variant.model_name,
        pretrained=route.variant.pretrained,
        model_sha256=str(model_sha256).upper(),
        image_size=route.variant.image_size,
        patch_size=route.variant.patch_size,
        window=route.variant.window,
        stride=route.variant.stride,
    )
    # Exercise the exact DynamicTokenCache directory identity at construction.
    if len(cache_digest(key)) != 64:  # pragma: no cover - cryptographic guard
        raise RuntimeError("dynamic cache digest is not SHA256")
    return key


def validate_spectrogram_cache_key(
    route: SpectrogramRoute,
    key: DynamicCacheKey,
) -> None:
    variant = route.variant
    expected = {
        "renderer": renderer_identity(route),
        "model_name": variant.model_name,
        "pretrained": variant.pretrained,
        "image_size": variant.image_size,
        "patch_size": variant.patch_size,
        "window": variant.window,
        "stride": variant.stride,
    }
    actual = {
        "renderer": key.renderer,
        "model_name": key.model_name,
        "pretrained": key.pretrained,
        "image_size": key.image_size,
        "patch_size": key.patch_size,
        "window": key.window,
        "stride": key.stride,
    }
    if actual != expected:
        raise ValueError("dynamic cache is not the mandatory spectrogram B16/W240/S60 cache")


__all__ = [
    "ARMS",
    "FULL_ARM",
    "REGISTRY_ID",
    "REL_ARM",
    "SpectrogramRoute",
    "SpectrogramVariant",
    "build_spectrogram_registry",
    "load_spectrogram_route",
    "make_spectrogram_cache_key",
    "renderer_identity",
    "route_config_sha256",
    "validate_spectrogram_cache_key",
]
