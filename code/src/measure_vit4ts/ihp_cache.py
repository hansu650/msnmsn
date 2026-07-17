"""Label-free IHP scoring from a frozen ViT4TS token cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ihp import index_consistent_harmonic_projection


def universal_token_cost(tokens: torch.Tensor) -> torch.Tensor:
    """Return the frozen universal-memory cosine cost for every query token."""

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ValueError("tokens must have shape [windows, tokens, features]")
    if not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all()):
        raise ValueError("tokens must be finite floating-point values")
    query = F.normalize(tokens, dim=-1)
    memory = F.normalize(torch.median(tokens, dim=0).values, dim=-1)
    similarity = torch.matmul(query, memory.transpose(-1, -2)).clamp(-1.0, 1.0)
    return (0.5 * (1.0 - similarity)).min(dim=-1).values


def score_token_cache(
    cache_path: Path, device: torch.device | str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return IHP base-grid maps and 224-column window scores from a cache."""

    device = torch.device(device)
    with np.load(Path(cache_path), allow_pickle=False) as payload:
        required = {
            "large_tokens",
            "mid_tokens",
            "patch_tokens",
            "large_mask",
            "mid_mask",
        }
        if set(payload.files) != required:
            raise ValueError("token cache has an unexpected schema")
        large = torch.from_numpy(payload["large_tokens"]).to(device)
        mid = torch.from_numpy(payload["mid_tokens"]).to(device)
        patch = torch.from_numpy(payload["patch_tokens"]).to(device)
        large_mask = torch.from_numpy(payload["large_mask"]).to(device)
        mid_mask = torch.from_numpy(payload["mid_mask"]).to(device)

    maps = index_consistent_harmonic_projection(
        universal_token_cost(large),
        universal_token_cost(mid),
        universal_token_cost(patch),
        large_mask,
        mid_mask,
    )
    raster = F.interpolate(
        maps.unsqueeze(1),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    # Preserve the released ViT4TS reducer used by the frozen full experiment:
    # average the largest quarter of raster values in every image column.
    top_rows = max(1, int(np.ceil(raster.shape[-2] * 0.25)))
    window_columns = torch.topk(
        raster,
        k=top_rows,
        dim=-2,
        largest=True,
        sorted=False,
    ).values.mean(dim=-2)
    return (
        np.ascontiguousarray(maps.detach().cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(
            window_columns.detach().cpu().numpy(), dtype=np.float64
        ),
    )
