"""Initialize the isolated ViTTrace v3 experiment record.

This module is deliberately read-only with respect to the frozen data, model,
token caches, and previous score transactions.  It creates only the new v3
output tree and records enough host/provenance information to make later arm
manifests auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil
import yaml


REQUIRED_DIRECTORIES = (
    "config",
    "code",
    "tests",
    "manifests",
    "provenance",
    "failures",
    "caches",
    "results",
    "results/qualitative_plot_data",
    "tables",
    "plot_data",
    "rough_figures",
    "logs",
    "runs",
)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _run(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _drive(path: str) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gib": usage.total / 1024**3,
        "used_gib": usage.used / 1024**3,
        "free_gib": usage.free / 1024**3,
    }


def _path_record(path: str | Path, *, hash_file: bool = False) -> dict[str, Any]:
    target = Path(path)
    record: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
    }
    if target.is_file():
        record["bytes"] = target.stat().st_size
        if hash_file:
            record["sha256"] = sha256_file(target)
    return record


def collect_inventory(config_path: Path, repository: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("inventory requires stage=vittrace_ablation_full_v3")
    repository = Path(repository).resolve()
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    try:
        import torch

        torch_record: Mapping[str, Any] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        torch_record = {"error": f"{type(exc).__name__}: {exc}"}
    virtual = psutil.virtual_memory()
    config_sha = sha256_file(config_path)
    paths = {
        "manifest": _path_record(config["manifest"]["path"], hash_file=True),
        "data_root": _path_record(config["data"]["root"]),
        "anomalies_csv": _path_record(config["data"]["anomalies_csv"], hash_file=True),
        "vendor_root": _path_record(config["vendor"]["root"]),
        "frozen_result_package": _path_record(
            config["frozen_inputs"]["result_package"]
        ),
        "frozen_result_zip": _path_record(
            config["frozen_inputs"]["result_package_zip"], hash_file=True
        ),
        "coordinate_cache_root": _path_record(
            config["frozen_inputs"]["coordinate_cache_root"]
        ),
        "coordinate_run_root": _path_record(
            config["frozen_inputs"]["coordinate_run_root"]
        ),
        "vittrace_run_root": _path_record(
            config["frozen_inputs"]["vittrace_run_root"]
        ),
        "vittrace_metric_file": _path_record(
            config["frozen_inputs"]["vittrace_metric_file"], hash_file=True
        ),
        "paper_pdf": _path_record(config["paper_reference"]["pdf"], hash_file=True),
    }
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "repository": str(repository),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha,
        "git": {
            "head": _run([*git_prefix, "rev-parse", "HEAD"]),
            "status": _run([*git_prefix, "status", "--short"]),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        },
        "packages": _package_versions(
            (
                "torch",
                "torchvision",
                "open_clip_torch",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "TSB-AD",
                "psutil",
                "matplotlib",
                "seaborn",
                "PyYAML",
            )
        ),
        "torch": dict(torch_record),
        "gpu": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "memory": {
            "total_gib": virtual.total / 1024**3,
            "available_gib": virtual.available / 1024**3,
            "percent": virtual.percent,
        },
        "drives": {"C": _drive("C:\\"), "D": _drive("D:\\")},
        "paths": paths,
        "frozen_identities": {
            "manifest_expected_sha256": config["manifest"]["sha256"],
            "vendor_commit": config["vendor"]["commit"],
            "default_model_sha256": config["vendor"]["default_model_sha256"],
            "frozen_result_zip_expected_sha256": config["frozen_inputs"][
                "result_package_zip_sha256"
            ],
        },
    }
    return inventory


def _status_markdown(inventory: Mapping[str, Any]) -> str:
    git = inventory["git"]
    paths = inventory["paths"]
    dirty = git["status"]["stdout"] or "(clean)"
    path_lines = "\n".join(
        f"- `{name}`: `{record['path']}` - {'available' if record['exists'] else 'MISSING'}"
        for name, record in paths.items()
    )
    return f"""# ViTTrace Ablation Full v3 Status

- Stage: `INITIALIZED`
- Created (UTC): `{inventory['created_utc']}`
- Working directory: `{inventory['cwd']}`
- Repository: `{inventory['repository']}`
- Git commit: `{git['head']['stdout']}`
- Config SHA256: `{inventory['config_sha256']}`
- Python: `{inventory['python']['executable']}`
- GPU query: `{inventory['gpu']['stdout'] or inventory['gpu']['stderr']}`
- RAM available: `{inventory['memory']['available_gib']:.2f} GiB`
- C free: `{inventory['drives']['C']['free_gib']:.2f} GiB`
- D free: `{inventory['drives']['D']['free_gib']:.2f} GiB`

## Frozen and external inputs

{path_lines}

## Git status at initialization

```text
{dirty}
```

## Execution state

The frozen `ViTTrace_results_20260718_113945` package and the previous
13-arm transactions are read-only.  Bulk v3 arms remain blocked until the
B/16, W=240, stride=60 parity gate passes for `REL_U`, `IHP_LEGACY`, and
`FULL_COLUMN_240`.  Independent arms will retain explicit failure records
rather than being silently removed.
"""


def initialize(config_path: Path, repository: Path) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["paths"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_DIRECTORIES:
        (output / relative).mkdir(parents=True, exist_ok=True)
    inventory = collect_inventory(config_path, repository)
    inventory_path = output / "provenance" / "system_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "config" / Path(config_path).name)
    (output / "STATUS.md").write_text(_status_markdown(inventory), encoding="utf-8")
    log = output / "EXPERIMENT_LOG.md"
    if not log.exists():
        log.write_text(
            "# ViTTrace Ablation Full v3 Experiment Log\n\n"
            f"- `{inventory['created_utc']}` initialized isolated output tree; "
            "frozen inputs remain read-only.\n",
            encoding="utf-8",
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    print(initialize(args.config, args.repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
