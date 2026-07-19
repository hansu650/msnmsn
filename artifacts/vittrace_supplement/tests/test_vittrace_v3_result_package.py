from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from measure_vit4ts_v3.result_package import (
    CHECKSUM_NAME,
    FIXED_ZIP_DATETIME,
    MANIFEST_NAME,
    REQUIRED_PATHS,
    PackageValidationError,
    build_result_package,
    collect_package_payload,
    verify_package_root,
    verify_result_zip,
)


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _compact_tree(root: Path) -> None:
    _write(root / "README.md", "result package\n")
    _write(root / "STATUS.md", "COMPLETE\n")
    _write(root / "code" / "runner.py", "print('ok')\n")
    _write(root / "results" / "metrics.csv", "arm,value\nFULL,0.5\n")
    _write(root / "tables" / "table.csv", "arm,value\nFULL,0.5\n")
    _write(root / "plot_data" / "window.csv", "window,value\n240,0.5\n")
    _write(root / "rough_figures" / "window.svg", "<svg/>\n")
    _write(root / "caches" / "cache_index.csv", "cache,sha256\na,abc\n")


def test_build_and_verify_compact_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    # These files must never enter the compact package.
    _write(root / "caches" / "large_token_cache.npz", "not-a-real-cache")
    _write(root / "logs" / "runner.log", "verbose")
    _write(root / "datasets" / "data.csv", "private")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    archive = tmp_path / "result.zip"
    built, manifest = build_result_package(
        root,
        zip_path=archive,
        allow_incomplete=True,
        max_file_bytes=1024 * 1024,
    )
    assert built == archive.resolve()
    assert verify_package_root(root)
    assert verify_result_zip(archive)["sha256sums_sha256"] == manifest["sha256sums_sha256"]

    with (root / CHECKSUM_NAME).open("r", encoding="utf-8", newline="") as handle:
        paths = [row["relative_path"] for row in csv.DictReader(handle)]
    assert paths == sorted(paths)
    assert "caches/cache_index.csv" in paths
    assert "caches/large_token_cache.npz" not in paths
    assert not any(path.startswith("logs/") or path.startswith("datasets/") for path in paths)

    with zipfile.ZipFile(archive) as package:
        infos = package.infolist()
        assert [info.filename for info in infos] == sorted(info.filename for info in infos)
        assert all(info.date_time == FIXED_ZIP_DATETIME for info in infos)
        assert CHECKSUM_NAME in package.namelist()
        assert MANIFEST_NAME in package.namelist()
        embedded = json.loads(package.read(MANIFEST_NAME).decode("utf-8"))
        assert embedded["exclusion_contract"]["model_weights"] is True


def test_modified_payload_fails_verification(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    build_result_package(root, zip_path=tmp_path / "result.zip", allow_incomplete=True)
    _write(root / "results" / "metrics.csv", "arm,value\nFULL,0.9\n")
    with pytest.raises(PackageValidationError, match="mismatch"):
        verify_package_root(root)


def test_supplement_entries_are_optional_and_allow_listed(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    supplement_files = {
        "protocol.json": "{}\n",
        "input_identities.csv": "path,sha256\ninput,abc\n",
        "_SUPPLEMENT_DELIVERY_COMPLETE.json": "{}\n",
        "CONFIRMATION_BLOCKED.md": "No untouched cohort.\n",
        "qualitative/selection.csv": "series_id,role\na,success\n",
        "runtime/runtime_postcache.csv": "arm,median_ms\nFULL,1.0\n",
    }
    for relative, contents in supplement_files.items():
        _write(root / relative, contents)

    rows = collect_package_payload(root, allow_incomplete=True)
    packaged = {row["relative_path"] for row in rows}
    assert set(supplement_files).issubset(packaged)
    # Supplement support must not strengthen the legacy completeness contract.
    assert "protocol.json" not in REQUIRED_PATHS
    assert "qualitative" not in REQUIRED_PATHS
    assert "runtime" not in REQUIRED_PATHS


def test_supplement_directories_preserve_forbidden_payload_protection(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    _write(root / "qualitative" / "datasets" / "private.csv", "private\n")
    _write(root / "runtime" / "logs" / "trace.log", "verbose\n")
    _write(root / "runtime" / "weights.pt", "weight\n")

    with pytest.raises(PackageValidationError, match="forbidden"):
        collect_package_payload(root, allow_incomplete=True)

    (root / "runtime" / "weights.pt").unlink()
    packaged = {
        row["relative_path"]
        for row in collect_package_payload(root, allow_incomplete=True)
    }
    assert "qualitative/datasets/private.csv" not in packaged
    assert "runtime/logs/trace.log" not in packaged


def test_forbidden_model_payload_and_large_file_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    _write(root / "results" / "weights.pt", "weight")
    with pytest.raises(PackageValidationError, match="forbidden"):
        collect_package_payload(root, allow_incomplete=True)

    (root / "results" / "weights.pt").unlink()
    _write(root / "results" / "too_large.csv", "0123456789")
    with pytest.raises(PackageValidationError, match="exceeds"):
        collect_package_payload(root, allow_incomplete=True, max_file_bytes=5)


def test_strict_mode_requires_complete_delivery_tree(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    with pytest.raises(PackageValidationError, match="incomplete"):
        collect_package_payload(root, allow_incomplete=False)


def test_reparse_member_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _compact_tree(root)
    target = tmp_path / "external.txt"
    _write(target, "outside")
    link = root / "results" / "external_link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(PackageValidationError, match="reparse"):
        collect_package_payload(root, allow_incomplete=True)
