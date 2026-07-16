from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from paano_k0.config import (
    ProtocolError,
    expand_primary_jobs,
    load_protocol,
)
from paano_k0.run_series import parse_args
from paano_k0.schemas import (
    SeriesSpec,
    Trajectory,
    make_run_id,
    scored_checkpoints,
)


ROOT = Path(__file__).resolve().parents[2]


def _series() -> tuple[SeriesSpec, ...]:
    families = (
        ("NAB", "U"),
        ("IOPS", "U"),
        ("Exathlon", "M"),
        ("SMD", "M"),
        ("SMAP", "M"),
        ("SWaT", "M"),
    )
    return tuple(
        SeriesSpec(
            series_id=f"series-{index}-{family}",
            family=family,
            track=track,
            csv_path=Path(f"fixture-{index}.csv"),
            csv_sha256=f"{index + 1:x}" * 64,
            rows=1024,
            channels=1,
            train_end=512,
            feature_columns=("Data",),
            label_column="Label",
        )
        for index, (family, track) in enumerate(families)
    )


def test_primary_job_matrix(tmp_path: Path) -> None:
    protocol = load_protocol(ROOT / "configs" / "k0_protocol.yaml")
    jobs = expand_primary_jobs(
        protocol,
        _series(),
        ROOT / "vendor-placeholder",
        tmp_path,
    )

    assert len(jobs) == 6 * 4
    assert {job.seed for job in jobs} == {2027}
    assert Counter(job.trajectory for job in jobs) == {
        Trajectory.OFFICIAL: 6,
        Trajectory.PAPERNEG: 6,
        Trajectory.PAPERNEG_NONOVERLAP: 6,
        Trajectory.RAND_BN: 6,
    }
    assert all(
        {job.trajectory for job in jobs if job.series.series_id == spec.series_id}
        == set(Trajectory)
        for spec in _series()
    )

    scored = [
        make_run_id(job.series.series_id, job.seed, job.trajectory, checkpoint)
        for job in jobs
        for checkpoint in scored_checkpoints(job.trajectory)
    ]
    assert len(scored) == 6 * (2 + 2 + 2 + 1) == 42
    assert len(set(scored)) == 42


def test_forbidden_arm_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Trajectory("RESCUE_ALPHA_BLEND")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--config",
                str(ROOT / "configs" / "k0_protocol.yaml"),
                "--manifest",
                str(ROOT / "docs" / "K0_DATA_MANIFEST.csv"),
                "--series-id",
                "fixture",
                "--trajectory",
                "RESCUE_ALPHA_BLEND",
                "--seed",
                "2027",
                "--vendor-root",
                str(ROOT / "vendor-placeholder"),
                "--output-root",
                str(tmp_path),
            ]
        )

    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(
        (ROOT / "configs" / "k0_protocol.yaml").read_text(encoding="utf-8")
        + "\nrescue_arm: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="unknown=.*rescue_arm"):
        load_protocol(mutated)
