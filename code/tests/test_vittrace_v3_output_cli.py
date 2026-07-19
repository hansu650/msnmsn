from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from measure_vit4ts_v3.output_cli import main


def test_structural_cli_writes_mandatory_audit(tmp_path: Path, capsys) -> None:
    output = tmp_path / "structural_audit.csv"
    assert (
        main(
            [
                "structural",
                "--patch-grid",
                "14x14",
                "--window",
                "240",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    frame = pd.read_csv(output)
    assert payload["rows"] == len(frame)
    assert {"released", "literal", "nctp_linear"}.issubset(
        set(frame["operator"])
    )


def test_package_cli_builds_verified_zip(tmp_path: Path, capsys) -> None:
    root = tmp_path / "result"
    root.mkdir()
    (root / "README.md").write_text("compact\n", encoding="utf-8")
    results = root / "results"
    results.mkdir()
    (results / "metrics.csv").write_text("arm,value\nFULL,0.5\n", encoding="utf-8")
    archive = tmp_path / "compact.zip"
    assert (
        main(
            [
                "package",
                "--root",
                str(root),
                "--zip",
                str(archive),
                "--allow-incomplete",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert archive.is_file()
    assert payload["payload_file_count"] == 2
    assert len(payload["sha256sums_sha256"]) == 64
