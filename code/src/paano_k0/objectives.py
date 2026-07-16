"""Objective functions for the frozen PaAno K0 trajectories.

This module deliberately contains no data loading or metric code.  It receives
embeddings and replayed indices and implements only the two negative-selection
semantics registered in the frozen protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .schemas import Trajectory


@dataclass(frozen=True)
class TripletBatch:
    """Triplet objective and detached mechanism observables for one batch."""

    unscaled_loss: Tensor
    scaled_loss: Tensor
    d_positive: Tensor
    d_negative: Tensor
    hinge_margin: Tensor
    hinge: Tensor
    active_mask: Tensor
    negative_candidate_indices: Tensor
    negative_data_indices: Tensor
    temporal_gap: Tensor
    index_collision: Tensor


@dataclass(frozen=True)
class PretextBatch:
    """Official adjacent-vs-unadjacent pretext result for one batch."""

    loss: Tensor
    positive_loss: Tensor
    negative_loss: Tensor
    logits: Tensor
    labels: Tensor
    valid_count: int
    accuracy: float


@dataclass(frozen=True)
class GradDiagnostics:
    """Encoder-only gradient geometry without populating ``parameter.grad``."""

    triplet_norm: float
    pretext_norm: float
    cosine: float | None


def _require_matrix(name: str, value: Tensor, *, minimum_rows: int = 1) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if value.shape[0] < minimum_rows or value.shape[1] < 1:
        raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _require_vector(name: str, value: Tensor, length: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() != length:
        raise ValueError(f"{name} must have shape [{length}]")


def pretext_weight(
    iteration: int,
    total_iterations: int,
    initial_weight: float = 1.0,
) -> float:
    """Return the released linear pretext schedule.

    Iterations are one-based.  For 100 total iterations this is positive only
    for iterations 1--19 and exactly zero from iteration 20 onward.
    """

    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if iteration < 1 or iteration > total_iterations:
        raise ValueError("iteration must be in [1, total_iterations]")
    if not math.isfinite(initial_weight) or initial_weight < 0:
        raise ValueError("initial_weight must be finite and non-negative")
    cutoff = total_iterations / 5.0
    if iteration < cutoff:
        return float(initial_weight * (1.0 - iteration / cutoff))
    return 0.0


def official_negative_indices(z_anchor: Tensor, z_positive: Tensor) -> Tensor:
    """Select the farthest off-diagonal projected positive view per anchor."""

    _require_matrix("z_anchor", z_anchor, minimum_rows=2)
    _require_matrix("z_positive", z_positive, minimum_rows=2)
    if z_anchor.shape != z_positive.shape:
        raise ValueError("z_anchor and z_positive must have identical shape")
    # ``compute_triplet_batch`` supplies normalized tensors.  Normalizing again
    # is intentionally avoided so this function is a direct vendor transcription.
    similarity = z_anchor @ z_positive.T
    distance = 1.0 - similarity
    distance = distance.clone()
    distance.diagonal().fill_(-float("inf"))
    return torch.argmax(distance, dim=1)


def paper_negative_indices(h_anchor: Tensor) -> Tensor:
    """Select the farthest off-diagonal anchor in encoder cosine space."""

    _require_matrix("h_anchor", h_anchor, minimum_rows=2)
    normalized = F.normalize(h_anchor, p=2, dim=1, eps=1e-12)
    distance = 1.0 - normalized @ normalized.T
    distance = distance.clone()
    distance.diagonal().fill_(-float("inf"))
    return torch.argmax(distance, dim=1)


def compute_triplet_batch(
    model: nn.Module,
    h_anchor: Tensor,
    h_positive: Tensor,
    anchor_indices: Tensor,
    positive_indices: Tensor,
    trajectory: Trajectory,
    margin: float,
    divisor: float,
    temperature: float,
) -> TripletBatch:
    """Compute the exact registered triplet objective for one trajectory."""

    _require_matrix("h_anchor", h_anchor, minimum_rows=2)
    _require_matrix("h_positive", h_positive, minimum_rows=2)
    if h_anchor.shape != h_positive.shape:
        raise ValueError("h_anchor and h_positive must have identical shape")
    batch_size = h_anchor.shape[0]
    _require_vector("anchor_indices", anchor_indices, batch_size)
    _require_vector("positive_indices", positive_indices, batch_size)
    if anchor_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("anchor_indices must have integer dtype")
    if positive_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("positive_indices must have integer dtype")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    try:
        trajectory = Trajectory(trajectory)
    except ValueError as exc:
        raise ValueError(f"unregistered trajectory: {trajectory!r}") from exc
    if trajectory is Trajectory.RAND_BN:
        raise ValueError("RAND_BN has no triplet objective")

    z_anchor = F.normalize(model.projection(h_anchor), p=2, dim=1, eps=1e-12)
    z_positive = F.normalize(model.projection(h_positive), p=2, dim=1, eps=1e-12)
    _require_matrix("z_anchor", z_anchor, minimum_rows=2)
    _require_matrix("z_positive", z_positive, minimum_rows=2)

    if trajectory is Trajectory.OFFICIAL:
        candidate_rows = official_negative_indices(z_anchor, z_positive)
        z_negative = z_positive[candidate_rows]
        negative_data_indices = positive_indices[candidate_rows]
    else:
        candidate_rows = paper_negative_indices(h_anchor)
        z_negative = F.normalize(
            model.projection(h_anchor[candidate_rows]), p=2, dim=1, eps=1e-12
        )
        negative_data_indices = anchor_indices[candidate_rows]

    positive_similarity = (z_anchor * z_positive).sum(dim=1) / temperature
    negative_similarity = (z_anchor * z_negative).sum(dim=1) / temperature
    d_positive = 1.0 - positive_similarity
    d_negative = 1.0 - negative_similarity
    hinge_margin = d_positive - d_negative + margin
    hinge = F.relu(hinge_margin)
    unscaled_loss = hinge.mean()
    scaled_loss = unscaled_loss / divisor

    anchor_long = anchor_indices.to(dtype=torch.long)
    negative_long = negative_data_indices.to(dtype=torch.long)
    temporal_gap = (anchor_long - negative_long).abs()
    index_collision = anchor_long.eq(negative_long)
    return TripletBatch(
        unscaled_loss=unscaled_loss,
        scaled_loss=scaled_loss,
        d_positive=d_positive,
        d_negative=d_negative,
        hinge_margin=hinge_margin,
        hinge=hinge,
        active_mask=hinge_margin > 0,
        negative_candidate_indices=candidate_rows,
        negative_data_indices=negative_data_indices,
        temporal_gap=temporal_gap,
        index_collision=index_collision,
    )


def compute_pretext_batch(
    model: nn.Module,
    h_anchor: Tensor,
    h_pretext: Tensor,
    valid_mask: Tensor,
    unadjacent_indices: Tensor,
    criterion: nn.Module,
) -> PretextBatch:
    """Compute the released temporal adjacency classification objective."""

    _require_matrix("h_anchor", h_anchor)
    _require_matrix("h_pretext", h_pretext)
    if h_anchor.shape != h_pretext.shape:
        raise ValueError("h_anchor and h_pretext must have identical shape")
    batch_size, _ = h_anchor.shape
    _require_vector("valid_mask", valid_mask, batch_size)
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must be bool")
    if (
        not isinstance(unadjacent_indices, Tensor)
        or unadjacent_indices.ndim != 2
        or unadjacent_indices.shape[0] != batch_size
        or unadjacent_indices.shape[1] < 1
    ):
        raise ValueError("unadjacent_indices must have shape [B,K], K>=1")
    if unadjacent_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("unadjacent_indices must have integer dtype")
    if unadjacent_indices.min().item() < 0 or unadjacent_indices.max().item() >= batch_size:
        raise ValueError("unadjacent_indices is out of minibatch range")

    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise ValueError("pretext batch has no valid adjacent pair")
    adjacent_features = torch.cat(
        [h_anchor[valid_mask], h_pretext[valid_mask]], dim=1
    )
    num_random = unadjacent_indices.shape[1]
    flat_unadjacent = unadjacent_indices.to(dtype=torch.long).reshape(-1)
    unadjacent_features = torch.cat(
        [
            h_anchor.repeat_interleave(num_random, dim=0),
            h_anchor[flat_unadjacent],
        ],
        dim=1,
    )
    features = torch.cat([adjacent_features, unadjacent_features], dim=0)
    labels = torch.cat(
        [
            torch.ones(valid_count, device=h_anchor.device, dtype=h_anchor.dtype),
            torch.zeros(
                unadjacent_features.shape[0],
                device=h_anchor.device,
                dtype=h_anchor.dtype,
            ),
        ]
    )
    logits = model.classification_head(features).squeeze(1)
    if logits.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("classification_head must emit one logit per pair")
    per_example = criterion(logits, labels)
    if per_example.ndim == 0:
        raise ValueError("criterion must use reduction='none'")
    positive_loss = per_example[:valid_count].mean()
    negative_loss = per_example[valid_count:].mean()
    loss = positive_loss + negative_loss
    accuracy = float(((logits >= 0) == (labels >= 0.5)).float().mean().item())
    return PretextBatch(
        loss=loss,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        logits=logits,
        labels=labels,
        valid_count=valid_count,
        accuracy=accuracy,
    )


def _gradient_vector(
    loss: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, float]:
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    pieces = [
        (torch.zeros_like(param) if grad is None else grad).reshape(-1)
        for param, grad in zip(parameters, grads, strict=True)
    ]
    vector = torch.cat(pieces) if pieces else loss.new_zeros(0)
    norm = float(torch.linalg.vector_norm(vector).detach().cpu().item())
    return vector, norm


def encoder_gradient_diagnostics(
    triplet_scaled: Tensor,
    weighted_pretext: Tensor,
    encoder_parameters: Sequence[nn.Parameter],
) -> GradDiagnostics:
    """Measure separate encoder gradients without mutating ``.grad`` fields."""

    parameters = tuple(p for p in encoder_parameters if p.requires_grad)
    if not parameters:
        return GradDiagnostics(0.0, 0.0, None)
    if triplet_scaled.ndim != 0 or not torch.isfinite(triplet_scaled):
        raise ValueError("triplet_scaled must be a finite scalar")
    before = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    triplet_vector, triplet_norm = _gradient_vector(triplet_scaled, parameters)

    if (
        weighted_pretext.ndim != 0
        or not torch.isfinite(weighted_pretext)
        or not weighted_pretext.requires_grad
        or float(weighted_pretext.detach().abs().cpu().item()) == 0.0
    ):
        pretext_vector = triplet_vector.new_zeros(triplet_vector.shape)
        pretext_norm = 0.0
    else:
        pretext_vector, pretext_norm = _gradient_vector(weighted_pretext, parameters)

    cosine: float | None
    if triplet_norm == 0.0 or pretext_norm == 0.0:
        cosine = None
    else:
        value = F.cosine_similarity(
            triplet_vector.unsqueeze(0), pretext_vector.unsqueeze(0), dim=1, eps=1e-12
        )[0]
        cosine = float(value.detach().cpu().item())

    for parameter, original in zip(parameters, before, strict=True):
        if original is None:
            if parameter.grad is not None:
                raise RuntimeError("gradient diagnostics populated parameter.grad")
        elif parameter.grad is None or not torch.equal(parameter.grad, original):
            raise RuntimeError("gradient diagnostics modified an existing parameter.grad")
    return GradDiagnostics(triplet_norm, pretext_norm, cosine)


__all__ = [
    "GradDiagnostics",
    "PretextBatch",
    "TripletBatch",
    "compute_pretext_batch",
    "compute_triplet_batch",
    "encoder_gradient_diagnostics",
    "official_negative_indices",
    "paper_negative_indices",
    "pretext_weight",
]
