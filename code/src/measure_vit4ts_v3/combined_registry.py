"""Append-only utility for registering completed ViTTrace v3 score stages.

Stage producers keep local arm names (for example ``REL/IHP/FULL``).  A stage
fragment maps those names to globally unique combined arm IDs and freezes the
presentation/provenance metadata required by downstream tables.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .combined_protocol import load_combined_protocol, validate_combined_protocol


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_stage_payload(
    protocol_payload: Mapping[str, Any], fragment: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a validated protocol with one namespaced stage appended."""

    payload = json.loads(json.dumps(dict(protocol_payload)))
    if set(fragment) != {"stage", "contrasts"}:
        raise ValueError("combined stage fragment requires stage and contrasts only")
    stage = dict(fragment["stage"])
    contrasts = [dict(row) for row in fragment["contrasts"]]
    existing_stage_ids = {str(row["stage_id"]) for row in payload["stages"]}
    if str(stage.get("stage_id")) in existing_stage_ids:
        raise ValueError("combined stage is already registered")
    existing_arms = {
        str(arm["arm"])
        for existing in payload["stages"]
        for arm in existing["arms"]
    }
    next_order = len(existing_arms)
    normalized_arms = []
    for offset, raw in enumerate(stage.get("arms", [])):
        arm = dict(raw)
        if str(arm.get("arm")) in existing_arms:
            raise ValueError("combined output arm is already registered")
        arm["order"] = next_order + offset
        metadata = dict(arm.get("metadata", {}))
        if "display_name" not in metadata or "is_final" not in metadata:
            raise ValueError("incremental arms must explicitly freeze display_name/is_final")
        arm["metadata"] = metadata
        normalized_arms.append(arm)
    if not normalized_arms:
        raise ValueError("incremental stage has no arms")
    stage["arms"] = normalized_arms
    payload["stages"].append(stage)
    payload["contrasts"].extend(contrasts)
    # Validation also checks config/manifest hashes and every contrast endpoint.
    validate_combined_protocol(payload)
    return payload


def append_stage(registry: Path, fragment: Path, output: Path | None = None) -> Path:
    registry_path = Path(registry).resolve(strict=True)
    load_combined_protocol(registry_path)
    protocol_payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    fragment_payload = json.loads(Path(fragment).read_text(encoding="utf-8-sig"))
    if not isinstance(fragment_payload, Mapping):
        raise ValueError("combined stage fragment must be a JSON object")
    updated = append_stage_payload(protocol_payload, fragment_payload)
    destination = Path(output or registry_path)
    _atomic_json(destination, updated)
    load_combined_protocol(destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--stage-fragment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(append_stage(args.registry, args.stage_fragment, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["append_stage", "append_stage_payload", "main"]
