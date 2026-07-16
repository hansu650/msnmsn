from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from paano_k0.instrumentation import IterationRecorder
from paano_k0.objectives import TripletBatch
from paano_k0.replay import state_dict_sha256
from paano_k0.schemas import CheckpointKind, Trajectory
import paano_k0.trainer as trainer_module


class TinyStore:
    def __init__(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4)
        self.patches = values / values.max()

    def __len__(self) -> int:
        return self.patches.shape[0]

    def take(self, indices: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.patches[torch.as_tensor(indices, dtype=torch.long)]


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.encoder = nn.Linear(4, 2)
        self.projection_head = nn.Linear(2, 2)
        self.classification_head = nn.Linear(4, 1)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.bn(x).squeeze(1))

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)


def _protocol(iterations: int = 2) -> SimpleNamespace:
    hp = SimpleNamespace(
        iterations=iterations,
        margin=0.1,
        triplet_divisor=10.0,
        learning_rate=0.01,
        weight_decay=0.0001,
        pretext_step=2,
        patch_size=4,
    )
    registered = [
        SimpleNamespace(id=Trajectory.OFFICIAL, positive_offsets=(-1, 1)),
        SimpleNamespace(id=Trajectory.PAPERNEG, positive_offsets=(-1, 1)),
        SimpleNamespace(
            id=Trajectory.PAPERNEG_NONOVERLAP, positive_offsets=(-4, 4)
        ),
    ]
    return SimpleNamespace(official_hyperparameters=hp, trajectories=registered)


def _replay(iterations: int = 2) -> SimpleNamespace:
    records = tuple(
        SimpleNamespace(
            anchor_indices=np.asarray([1, 3], dtype=np.int64),
            positive_uniform=np.asarray([0.1, 0.9], dtype=np.float32),
            unadjacent_uniform=np.full((2, 5), 0.25, dtype=np.float32),
        )
        for _ in range(iterations)
    )
    return SimpleNamespace(
        iterations=records,
        n_train_patches=6,
        series_id="tiny",
        seed=2027,
        payload_sha256="a" * 64,
    )


def _initial_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_best_is_post_update_minibatch_loss_state(tmp_path, monkeypatch) -> None:
    model = TinyEncoder()
    initial = _initial_state(model)
    calls = {"count": 0}

    def fake_triplet(model, h_anchor, h_positive, *args, **kwargs):
        calls["count"] += 1
        constant = float(calls["count"])
        scaled = model.encoder.weight.sum() * 0.01 + constant
        batch = h_anchor.shape[0]
        zeros = torch.zeros(batch, device=h_anchor.device)
        return TripletBatch(
            unscaled_loss=scaled * 10.0,
            scaled_loss=scaled,
            d_positive=zeros,
            d_negative=zeros,
            hinge_margin=zeros,
            hinge=zeros,
            active_mask=torch.zeros(batch, dtype=torch.bool, device=h_anchor.device),
            negative_candidate_indices=torch.arange(batch, device=h_anchor.device),
            negative_data_indices=torch.arange(batch, device=h_anchor.device),
            temporal_gap=torch.zeros(batch, dtype=torch.long, device=h_anchor.device),
            index_collision=torch.zeros(batch, dtype=torch.bool, device=h_anchor.device),
        )

    monkeypatch.setattr(trainer_module, "compute_triplet_batch", fake_triplet)
    recorder = IterationRecorder(tmp_path / "train.jsonl", expected_iterations=2)
    result = trainer_module.train_trajectory(
        model,
        initial,
        TinyStore(),
        _replay(),
        Trajectory.OFFICIAL,
        _protocol(),
        torch.device("cpu"),
        recorder,
    )
    assert result.summary.best_iteration == 1
    assert result.summary.last_iteration == 2
    assert result.summary.initial_state_sha256 == state_dict_sha256(initial)
    best = result.checkpoints[CheckpointKind.BEST]
    last = result.checkpoints[CheckpointKind.LAST]
    assert not torch.equal(best["encoder.weight"], initial["encoder.weight"])
    assert not torch.equal(best["encoder.weight"], last["encoder.weight"])

    manual = TinyEncoder()
    manual.load_state_dict(initial)
    optimizer = torch.optim.AdamW(
        manual.parameters(), lr=0.01, weight_decay=0.0001
    )
    optimizer.param_groups[0]["lr"] = trainer_module.cosine_learning_rate(1, 2, 0.01)
    optimizer.zero_grad(set_to_none=True)
    (manual.encoder.weight.sum() * 0.01 + 1.0).backward()
    optimizer.step()
    torch.testing.assert_close(
        best["encoder.weight"], manual.encoder.weight.detach(), atol=0, rtol=0
    )


def test_rand_bn_changes_only_bn_buffers(tmp_path) -> None:
    model = TinyEncoder()
    initial = _initial_state(model)
    recorder = IterationRecorder(tmp_path / "rand.jsonl", expected_iterations=2)
    result = trainer_module.replay_rand_bn(
        model,
        initial,
        TinyStore(),
        _replay(),
        _protocol(),
        torch.device("cpu"),
        recorder,
    )
    state = result.checkpoints[CheckpointKind.BN_CALIBRATED]
    parameter_names = {name for name, _ in model.named_parameters()}
    for name in parameter_names:
        assert torch.equal(state[name], initial[name]), name
    assert state["bn.num_batches_tracked"].item() == (
        initial["bn.num_batches_tracked"].item() + 2
    )
    assert not torch.equal(state["bn.running_mean"], initial["bn.running_mean"])
    assert result.summary.best_iteration is None
    assert result.summary.last_iteration == 2
