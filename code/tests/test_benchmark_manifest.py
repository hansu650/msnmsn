from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

import paano_k0.benchmark_manifest as module
from paano_k0.config import ProtocolError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(tmp_path: Path, names: tuple[tuple[str, str], ...]) -> Path:
    rows = []
    for index, (track, name) in enumerate(names):
        path = tmp_path / name
        path.write_text("x,Label\n0.0,0\n1.0,1\n", encoding="utf-8")
        rows.append(
            {
                "family": name.split("_")[1],
                "track": track,
                "file": name,
                "sha256": _sha(path),
                "rows": "2",
                "channels": "1",
                "train_end": "1",
                "bytes": str(path.stat().st_size),
                "local_path": path.as_posix(),
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_full_loader_allows_repeated_family_and_single_series_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        ("U", "001_NAB_id_1_Facility_tr_1_1st_2.csv"),
        ("U", "002_NAB_id_2_Facility_tr_1_1st_2.csv"),
        ("M", "003_SMD_id_1_Sensor_tr_1_1st_2.csv"),
    )
    manifest = _write_manifest(tmp_path, names)
    monkeypatch.setattr(module, "EXPECTED_TRACK_COUNTS", {"U": 2, "M": 1})
    # The production minimum is 96; this fixture tests manifest mechanics only.
    monkeypatch.setattr(module, "_parse_identity", lambda name: (name.split("_")[1], 1))
    specs = module.load_benchmark_manifest(manifest)
    assert [spec.family for spec in specs] == ["NAB", "NAB", "SMD"]
    selected = module.load_benchmark_series(manifest, Path(names[-1][1]).stem)
    assert selected.track == "M"


def test_full_loader_rejects_duplicate_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "001_NAB_id_1_Facility_tr_1_1st_2.csv"
    manifest = _write_manifest(tmp_path, (("U", name),))
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows.append(dict(rows[0]))
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(module, "EXPECTED_TRACK_COUNTS", {"U": 2, "M": 0})
    with pytest.raises(ProtocolError, match="unique series"):
        module.load_benchmark_manifest(manifest)
