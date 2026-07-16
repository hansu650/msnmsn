import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paano_k0.feature_data import read_feature_series
from paano_k0.schemas import SeriesSpec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_spec(path: Path) -> SeriesSpec:
    return SeriesSpec(
        series_id="fixture",
        family="Fixture",
        track="U",
        csv_path=path,
        csv_sha256=_sha(path),
        rows=3,
        channels=1,
        train_end=3,
        feature_columns=("Data",),
        label_column="Label",
    )


def test_feature_reader_never_loads_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("Data,Label\n1,0\n2,1\n3,0\n", encoding="utf-8")
    spec = _fixture_spec(csv_path)
    observed: list[tuple[str, ...]] = []
    original = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        usecols = tuple(kwargs.get("usecols") or ())
        observed.append(usecols)
        assert "Label" not in usecols
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    values = read_feature_series(spec)
    assert values.dtype == np.float32
    assert values.shape == (3, 1)
    assert observed == [("Data",)]


def test_runner_has_no_label_surface() -> None:
    module_spec = importlib.util.find_spec("paano_k0.run_series")
    if module_spec is None or module_spec.origin is None:
        pytest.skip("run_series is implemented in a later overlay step")
    source_path = Path(module_spec.origin)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_modules = {"label_data", "evaluate_scores", "aggregate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported = (node.module or "").split(".")[-1]
            assert imported not in forbidden_modules
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert all("label" not in arg.arg.lower() for arg in node.args.args)


def test_evaluator_reads_label_after_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_spec = importlib.util.find_spec("paano_k0.evaluate_scores")
    if module_spec is None:
        pytest.skip("evaluate_scores is implemented in a later overlay step")
    from paano_k0 import evaluate_scores

    called = 0

    def forbidden_read(*args, **kwargs):
        nonlocal called
        called += 1
        raise AssertionError("label reader ran before score verification")

    monkeypatch.setattr(evaluate_scores, "read_labels", forbidden_read)
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("Data,Label\n1,0\n2,1\n3,0\n", encoding="utf-8")
    spec = _fixture_spec(csv_path)
    signature = inspect.signature(evaluate_scores.evaluate_score_artifact)
    kwargs = {"run_dir": tmp_path / "missing", "spec": spec, "vendor": None}
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate_scores.evaluate_score_artifact(**kwargs)
    assert called == 0

