from pathlib import Path

import numpy as np
import torch

from paano_k0.replay import (
    _store_unit_uniforms,
    ReplayIdentity,
    build_initial_state,
    build_replay_plan,
    load_replay_plan,
    materialize_positive_indices,
    materialize_unadjacent_indices,
    save_replay_plan,
)


def test_float32_storage_preserves_open_unit_endpoint() -> None:
    raw = np.asarray(
        [0.0, 0.25, 0.9999999, 0.9999999801825278], dtype=np.float64
    )
    ordinary_cast = raw.astype(np.float32)
    assert ordinary_cast[-1] == np.float32(1.0)

    stored = _store_unit_uniforms(raw)

    assert stored.dtype == np.float32
    assert np.array_equal(stored[:-1], ordinary_cast[:-1])
    assert stored[-1] == np.nextafter(np.float32(1.0), np.float32(0.0))
    assert np.all((stored >= 0) & (stored < 1))


def test_regression_replay_seed_that_rounded_to_one() -> None:
    plan = build_replay_plan(
        17_580,
        512,
        100,
        1_741_137_625,
        series_id="170_MITDB_id_1_Medical_tr_17675_1st_17775",
    )
    endpoint = np.nextafter(np.float32(1.0), np.float32(0.0))
    assert plan.iterations[94].unadjacent_uniform[385, 0] == endpoint
    assert all(
        np.all((step.positive_uniform >= 0) & (step.positive_uniform < 1))
        and np.all((step.unadjacent_uniform >= 0) & (step.unadjacent_uniform < 1))
        for step in plan.iterations
    )


def _assert_plans_equal(left, right) -> None:
    assert left.payload_sha256 == right.payload_sha256
    assert len(left.iterations) == len(right.iterations)
    for a, b in zip(left.iterations, right.iterations, strict=True):
        assert np.array_equal(a.anchor_indices, b.anchor_indices)
        assert np.array_equal(a.positive_uniform, b.positive_uniform)
        assert np.array_equal(a.unadjacent_uniform, b.unadjacent_uniform)


def test_replay_is_bitwise_repeatable(tmp_path: Path) -> None:
    first = build_replay_plan(913, 512, 100, 2027, series_id="NAB")
    second = build_replay_plan(913, 512, 100, 2027, series_id="NAB")
    _assert_plans_equal(first, second)
    replay_path = tmp_path / "plan.npz"
    assert save_replay_plan(first, replay_path) == first.payload_sha256
    loaded = load_replay_plan(
        replay_path,
        ReplayIdentity("NAB", 2027, 913, 512, 100),
    )
    _assert_plans_equal(first, loaded)


def test_all_arms_share_anchor_batches() -> None:
    shared = build_replay_plan(1000, 128, 7, 99, series_id="fixture")
    arm_views = {
        arm: tuple(step.anchor_indices for step in shared.iterations)
        for arm in ("OFFICIAL", "PAPERNEG", "PAPERNEG_NONOVERLAP", "RAND_BN")
    }
    reference = arm_views["OFFICIAL"]
    for batches in arm_views.values():
        assert all(np.array_equal(a, b) for a, b in zip(reference, batches, strict=True))


def test_local_arms_share_positive_indices() -> None:
    plan = build_replay_plan(300, 64, 1, 7)
    step = plan.iterations[0]
    official = materialize_positive_indices(
        step.anchor_indices, 300, (-2, -1, 1, 2), step.positive_uniform
    )
    paperneg = materialize_positive_indices(
        step.anchor_indices, 300, (-2, -1, 1, 2), step.positive_uniform
    )
    assert np.array_equal(official, paperneg)


def test_nonoverlap_offsets_have_zero_overlap() -> None:
    anchors = np.array([0, 95, 96, 203, 299], dtype=np.int64)
    draws = np.array([0.0, 0.9, 0.1, 0.9, 0.2], dtype=np.float32)
    positives = materialize_positive_indices(anchors, 300, (-96, 96), draws)
    offsets = positives - anchors
    assert np.all(np.abs(offsets) == 96)
    assert np.all(np.maximum(0, 96 - np.abs(offsets)) == 0)


def test_unadjacent_indices_never_self_pair() -> None:
    uniform = np.linspace(0, 0.999, 40, dtype=np.float32).reshape(8, 5)
    indices = materialize_unadjacent_indices(8, uniform)
    assert indices.shape == (8, 5)
    assert not np.any(indices == np.arange(8, dtype=np.int64)[:, None])


def test_initial_state_is_identical() -> None:
    build = lambda: torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    first, first_hash = build_initial_state(build, 123)
    second, second_hash = build_initial_state(build, 123)
    assert first_hash == second_hash
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
