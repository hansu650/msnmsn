"""Fail-closed source provenance for method score transactions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


SCORE_SOURCE_FILES = (
    "data.py",
    "geometry.py",
    "renderers.py",
    "cache.py",
    "reducers.py",
    "runner.py",
    "provenance.py",
)


def score_source_sha256(package_root: Path | None = None) -> str:
    """Hash the exact project sources that can change committed scores.

    Config, vendor, renderer, and model-state identities are bound separately
    in each score manifest.  This digest prevents a reducer/runner correction
    from silently adopting score files produced by an older implementation.
    """

    root = Path(package_root) if package_root is not None else Path(__file__).parent
    root = root.resolve(strict=True)
    digest = sha256()
    for name in SCORE_SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing score-producing source: {path}")
        payload = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()
