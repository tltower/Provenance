from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise AssertionError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_classifier_train_dispatches_span_training_and_transfer(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    script_path = (
        Path("/Users/tatetower/Codex/memex-research/scripts/run_classifier_train.py")
    )
    module = _load_module("run_classifier_train_test", script_path)
    calls: list[tuple[str, object]] = []

    def fake_validate(task: str, dataset: str) -> None:
        calls.append(("validate", (task, dataset)))

    def fake_train_span_classifier(**kwargs):
        calls.append(("train_span", kwargs["model_name"]))
        return {"f1": 0.5}

    def fake_run_span_transfer_hf_model(**kwargs) -> None:
        calls.append(("transfer_span", kwargs["model_dir"]))

    monkeypatch.setattr(module, "validate_task_dataset", fake_validate)
    monkeypatch.setattr(module, "train_span_classifier", fake_train_span_classifier)
    monkeypatch.setattr(module, "run_span_transfer_hf_model", fake_run_span_transfer_hf_model)
    monkeypatch.setattr(
        module,
        "train_source_classifier",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("source trainer should not run")),
    )
    monkeypatch.setattr(
        module,
        "run_source_transfer_hf_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("source transfer should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_classifier_train.py",
            "--task",
            "span_role",
            "--dataset",
            "pe",
            "--model-name",
            "microsoft/deberta-v3-base",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert ("validate", ("span_role", "pe")) in calls
    assert ("train_span", "microsoft/deberta-v3-base") in calls
    assert ("transfer_span", tmp_path / "output" / "model") in calls
    assert '"f1": 0.5' in out


def test_run_probe_experiment_dispatches_source_probe_and_transfer(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    script_path = (
        Path("/Users/tatetower/Codex/memex-research/scripts/run_probe_experiment.py")
    )
    module = _load_module("run_probe_experiment_test", script_path)
    calls: list[tuple[str, object]] = []

    def fake_validate(task: str, dataset: str) -> None:
        calls.append(("validate", (task, dataset)))

    def fake_run_probe_experiment(**kwargs):
        calls.append(("probe", kwargs["model_name"]))
        return {"best_layer": 12, "best_f1": 0.42}

    def fake_run_source_transfer_probe(**kwargs) -> None:
        calls.append(("transfer_probe", kwargs["probe_dir"]))

    monkeypatch.setattr(module, "validate_task_dataset", fake_validate)
    monkeypatch.setattr(module, "run_probe_experiment", fake_run_probe_experiment)
    monkeypatch.setattr(module, "run_source_transfer_probe", fake_run_source_transfer_probe)
    monkeypatch.setattr(
        module,
        "run_span_transfer_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("span probe transfer should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe_experiment.py",
            "--task",
            "source_materiality",
            "--dataset",
            "scicite",
            "--model-name",
            "meta-llama/Llama-3.1-8B",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert ("validate", ("source_materiality", "scicite")) in calls
    assert ("probe", "meta-llama/Llama-3.1-8B") in calls
    assert ("transfer_probe", tmp_path / "output") in calls
    assert '"best_layer": 12' in out
