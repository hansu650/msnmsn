from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from paano_k0.objectives import (
    compute_triplet_batch,
    encoder_gradient_diagnostics,
    paper_negative_indices,
    pretext_weight,
)
from paano_k0.schemas import Trajectory


class IdentityHeads(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection_head = nn.Linear(dimension, dimension, bias=False)
        self.classification_head = nn.Linear(2 * dimension, 1)
        with torch.no_grad():
            self.projection_head.weight.copy_(torch.eye(dimension))

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)


def test_official_triplet_numerical_parity() -> None:
    model = IdentityHeads(3)
    h_anchor = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [-0.7, 0.1, 1.0]],
        requires_grad=True,
    )
    h_positive = torch.tensor(
        [[0.8, 0.1, 0.0], [-0.2, 1.0, 0.1], [-0.8, 0.0, 1.0]],
        requires_grad=True,
    )
    anchors = torch.tensor([10, 20, 30])
    positives = torch.tensor([11, 19, 31])
    actual = compute_triplet_batch(
        model,
        h_anchor,
        h_positive,
        anchors,
        positives,
        Trajectory.OFFICIAL,
        margin=0.1,
        divisor=10.0,
        temperature=1.0,
    )

    z_anchor = F.normalize(model.projection(h_anchor), dim=1)
    z_positive = F.normalize(model.projection(h_positive), dim=1)
    similarity = z_anchor @ z_positive.T
    positive_similarity = similarity.diag()
    filtered = similarity.clone()
    filtered.diagonal().fill_(float("inf"))
    negative_distance, negative_rows = torch.max(1.0 - filtered, dim=1)
    positive_distance = 1.0 - positive_similarity
    direct = F.relu(positive_distance - negative_distance + 0.1).mean() / 10.0

    torch.testing.assert_close(actual.scaled_loss, direct, atol=1e-7, rtol=0)
    torch.testing.assert_close(actual.d_positive, positive_distance, atol=1e-7, rtol=0)
    torch.testing.assert_close(actual.d_negative, negative_distance, atol=1e-7, rtol=0)
    assert torch.equal(actual.negative_candidate_indices, negative_rows)
    assert torch.equal(actual.negative_data_indices, positives[negative_rows])


def test_paper_negative_uses_anchor_encoder_space() -> None:
    h_anchor = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    selected = paper_negative_indices(h_anchor)
    assert selected.tolist() == [2, 2, 0, 1]

    model = IdentityHeads(2)
    result = compute_triplet_batch(
        model,
        h_anchor,
        h_anchor + torch.tensor([[0.0, 0.1]]),
        torch.tensor([100, 200, 300, 400]),
        torch.tensor([101, 201, 301, 401]),
        Trajectory.PAPERNEG,
        margin=0.1,
        divisor=10.0,
        temperature=1.0,
    )
    assert torch.equal(result.negative_candidate_indices, selected)
    assert result.negative_data_indices.tolist() == [300, 300, 100, 200]


def test_pretext_schedule_exact() -> None:
    weights = [pretext_weight(iteration, 100) for iteration in range(1, 101)]
    assert all(weight > 0 for weight in weights[:19])
    assert weights[0] == pytest.approx(0.95)
    assert weights[18] == pytest.approx(0.05)
    assert weights[19:] == [0.0] * 81


def test_gradient_diagnostics_do_not_accumulate_grad() -> None:
    first = nn.Linear(4, 3, bias=False)
    second = copy.deepcopy(first)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7.0

    output = first(x)
    triplet = (output.square()).mean()
    pretext = 0.25 * (output - 1.0).square().mean()
    diagnostic = encoder_gradient_diagnostics(
        triplet, pretext, tuple(first.parameters())
    )
    assert diagnostic.triplet_norm > 0
    assert diagnostic.pretext_norm > 0
    assert all(parameter.grad is None for parameter in first.parameters())
    (triplet + pretext).backward()

    control_output = second(x)
    control_loss = control_output.square().mean() + 0.25 * (
        control_output - 1.0
    ).square().mean()
    control_loss.backward()
    for measured, control in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(measured.grad, control.grad, atol=0, rtol=0)


def test_zero_pretext_gradient_reports_null_cosine() -> None:
    layer = nn.Linear(2, 2, bias=False)
    output = layer(torch.ones(2, 2))
    diagnostic = encoder_gradient_diagnostics(
        output.square().mean(), output.new_zeros(()), tuple(layer.parameters())
    )
    assert diagnostic.pretext_norm == 0.0
    assert diagnostic.cosine is None
    assert layer.weight.grad is None
