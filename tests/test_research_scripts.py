from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    script_path = _repo_root() / "scripts" / "run_classifier_train.py"
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
            "--run-transfer",
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
    script_path = _repo_root() / "scripts" / "run_probe_experiment.py"
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
            "Qwen/Qwen2.5-7B-Instruct",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-transfer",
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert ("validate", ("source_materiality", "scicite")) in calls
    assert ("probe", "Qwen/Qwen2.5-7B-Instruct") in calls
    assert ("transfer_probe", tmp_path / "output") in calls
    assert '"best_layer": 12' in out


def test_parse_layers_helper() -> None:
    script_path = _repo_root() / "scripts" / "run_probe_experiment.py"
    module = _load_module("run_probe_experiment_parse_layers_test", script_path)

    assert module._parse_layers(None) is None
    assert module._parse_layers("") is None
    assert module._parse_layers("1, 3,5") == (1, 3, 5)


def test_run_probe_experiment_skips_transfer_by_default(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    script_path = _repo_root() / "scripts" / "run_probe_experiment.py"
    module = _load_module("run_probe_experiment_skip_transfer_test", script_path)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "validate_task_dataset", lambda task, dataset: None)
    monkeypatch.setattr(
        module,
        "run_probe_experiment",
        lambda **kwargs: calls.append(("probe", kwargs["model_name"])) or {"best_layer": 2, "best_f1": 0.1},
    )
    monkeypatch.setattr(
        module,
        "run_source_transfer_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("transfer should be opt-in")),
    )
    monkeypatch.setattr(
        module,
        "run_span_transfer_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("transfer should be opt-in")),
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
            "Qwen/Qwen2.5-7B-Instruct",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert ("probe", "Qwen/Qwen2.5-7B-Instruct") in calls
    assert '"best_layer": 2' in out


def test_run_scicite_source_experiment_transfer_only_requires_run_transfer(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = _repo_root() / "scripts" / "run_scicite_source_experiment.py"
    module = _load_module("run_scicite_source_experiment_test", script_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scicite_source_experiment.py",
            "--model-name",
            "microsoft/deberta-v3-base",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "runs"),
            "--transfer-only",
        ],
    )

    try:
        module.main()
    except RuntimeError as exc:
        assert "--transfer-only requires --run-transfer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected transfer-only without run-transfer to fail")


def test_run_source_candidate_transfer_dispatches_model_transfer(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = _repo_root() / "scripts" / "run_source_candidate_transfer.py"
    module = _load_module("run_source_candidate_transfer_test", script_path)
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        module,
        "run_source_candidate_transfer_hf_model",
        lambda **kwargs: calls.append(("model", kwargs["model_dir"])),
    )
    monkeypatch.setattr(
        module,
        "run_source_candidate_transfer_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("probe transfer should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_source_candidate_transfer.py",
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    module.main()
    assert calls == [("model", tmp_path / "model")]


def test_run_sae_experiment_dispatches_with_parsed_layers(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    script_path = _repo_root() / "scripts" / "run_sae_experiment.py"
    module = _load_module("run_sae_experiment_test", script_path)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "validate_task_dataset", lambda task, dataset: calls.append(("validate", (task, dataset))))
    monkeypatch.setattr(
        module,
        "run_sae_experiment",
        lambda **kwargs: calls.append(("sae", kwargs["layers"])) or {"best_layer": 7, "best_f1": 0.8},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sae_experiment.py",
            "--task",
            "source_materiality",
            "--dataset",
            "scicite",
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--layers",
            "3,7,11",
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert ("validate", ("source_materiality", "scicite")) in calls
    assert ("sae", (3, 7, 11)) in calls
    assert '"best_layer": 7' in out


def test_run_source_candidate_transfer_requires_probe_model_name(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = _repo_root() / "scripts" / "run_source_candidate_transfer.py"
    module = _load_module("run_source_candidate_transfer_error_test", script_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_source_candidate_transfer.py",
            "--probe-dir",
            str(tmp_path / "probe"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    try:
        module.main()
    except RuntimeError as exc:
        assert "--probe-model-name is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing probe model name to fail")


def test_package_run_artifacts_script_dispatches(monkeypatch, tmp_path: Path, capsys) -> None:
    script_path = _repo_root() / "scripts" / "package_run_artifacts.py"
    module = _load_module("package_run_artifacts_test", script_path)
    calls: list[dict[str, object]] = []

    def fake_package_run_artifacts(**kwargs):
        calls.append(kwargs)
        return {"run_count": 1, "runs": [{"run_name": "scicite_deberta"}]}

    monkeypatch.setattr(module, "package_run_artifacts", fake_package_run_artifacts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_run_artifacts.py",
            "--run-root",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "out" / "runs.tar.gz"),
            "--run-name",
            "scicite_deberta",
            "--exclude-transfer",
            "--include-trainer",
        ],
    )

    module.main()
    out = capsys.readouterr().out

    assert calls[0]["run_root"] == tmp_path / "runs"
    assert calls[0]["output_path"] == tmp_path / "out" / "runs.tar.gz"
    assert calls[0]["run_names"] == ["scicite_deberta"]
    assert calls[0]["include_transfer"] is False
    assert calls[0]["include_trainer"] is True
    assert '"run_count": 1' in out
