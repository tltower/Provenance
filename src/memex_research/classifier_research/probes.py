from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import create_probe_components
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import load_jsonl_splits
from memex_research.classifier_research.tasks import (
    SOURCE_MATERIALITY_LABELS,
    SPAN_ROLE_LABELS,
    TASK_SOURCE_MATERIALITY,
    TASK_SPAN_ROLE,
)


def _require_probe_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import joblib  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            precision_recall_fscore_support,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Probe dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return joblib, np, torch, (LogisticRegression, accuracy_score, precision_recall_fscore_support)


def _select_layers(requested_layers: tuple[int, ...] | None, hidden_state_count: int) -> tuple[int, ...]:
    if requested_layers is None:
        return tuple(range(hidden_state_count))
    invalid = [layer for layer in requested_layers if layer < 0 or layer >= hidden_state_count]
    if invalid:
        raise ValueError(
            f"Requested probe layers {invalid} outside valid range 0..{hidden_state_count - 1}"
        )
    return tuple(dict.fromkeys(requested_layers))


def _device_for_model(model: Any) -> Any:
    _joblib, _np, torch, _libs = _require_probe_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return device


def _create_logistic_regression(logistic_regression_cls: Any) -> Any:
    kwargs: dict[str, Any] = {"max_iter": 1000}
    if "multi_class" in inspect.signature(logistic_regression_cls).parameters:
        kwargs["multi_class"] = "auto"
    return logistic_regression_cls(**kwargs)


def _sequence_hidden_state_cache(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
    cache_dir: Path,
    split_name: str,
    layers: tuple[int, ...] | None,
) -> tuple[int, ...]:
    _joblib, np, torch, _libs = _require_probe_stack()
    layer_vectors: dict[int, list[Any]] = {}
    device = _device_for_model(model)
    selected_layers: tuple[int, ...] | None = None
    for row in records:
        tokenized = tokenizer(
            str(row["context"]),
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            output = model(**tokenized)
        hidden_states = output.hidden_states
        if selected_layers is None:
            selected_layers = _select_layers(layers, len(hidden_states))
            layer_vectors = {layer_index: [] for layer_index in selected_layers}
        attention_mask = tokenized["attention_mask"][0].detach().cpu().numpy().astype(bool)
        for layer_index in selected_layers:
            hidden_state = hidden_states[layer_index]
            vectors = hidden_state[0].detach().cpu().numpy()
            pooled = vectors[attention_mask].mean(axis=0).astype(np.float16)
            layer_vectors[layer_index].append(pooled)
    if selected_layers is None:
        raise RuntimeError(f"No records available for split {split_name!r}")
    for layer_index, vectors in layer_vectors.items():
        np.save(cache_dir / f"{split_name}.layer_{layer_index:02d}.npy", np.vstack(vectors))
    return selected_layers


def _token_hidden_state_cache(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
    cache_dir: Path,
    split_name: str,
    layers: tuple[int, ...] | None,
) -> tuple[int, ...]:
    _joblib, np, torch, _libs = _require_probe_stack()
    layer_vectors: dict[int, list[Any]] = {}
    labels: list[str] = []
    device = _device_for_model(model)
    selected_layers: tuple[int, ...] | None = None
    for row in records:
        tokenized: Any = tokenizer(
            list(row["tokens"]),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            output = model(**tokenized)
        hidden_states = output.hidden_states
        if selected_layers is None:
            selected_layers = _select_layers(layers, len(hidden_states))
            layer_vectors = {layer_index: [] for layer_index in selected_layers}
        word_ids = tokenized.word_ids(0)
        first_positions: dict[int, int] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in first_positions:
                continue
            first_positions[word_id] = position
        for word_id, position in sorted(first_positions.items()):
            labels.append(str(row["labels"][word_id]))
            for layer_index in selected_layers:
                vector = hidden_states[layer_index][0, position].detach().cpu().numpy().astype(np.float16)
                layer_vectors[layer_index].append(vector)
    if selected_layers is None:
        raise RuntimeError(f"No records available for split {split_name!r}")
    for layer_index, vectors in layer_vectors.items():
        np.save(cache_dir / f"{split_name}.layer_{layer_index:02d}.npy", np.vstack(vectors))
    np.save(cache_dir / f"{split_name}.labels.npy", np.array(labels, dtype=object), allow_pickle=True)
    return selected_layers


def _load_cached_layer_vectors(cache_dir: Path, *, split_name: str, layer_index: int) -> Any:
    _joblib, np, _torch, _libs = _require_probe_stack()
    return np.load(cache_dir / f"{split_name}.layer_{layer_index:02d}.npy").astype(np.float32)


def _load_cached_label_strings(cache_dir: Path, *, split_name: str) -> list[str]:
    _joblib, np, _torch, _libs = _require_probe_stack()
    return np.load(cache_dir / f"{split_name}.labels.npy", allow_pickle=True).tolist()


def _compute_classification_metrics(gold: list[int], pred: list[int], *, labels: list[int]) -> dict[str, float]:
    _joblib, _np, _torch, libs = _require_probe_stack()
    _LogisticRegression, accuracy_score, precision_recall_fscore_support = libs
    precision, recall, f1, _support = precision_recall_fscore_support(
        gold,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def run_probe_experiment(
    *,
    task: str,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    max_length: int = 256,
    layers: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    joblib, _np, _torch, libs = _require_probe_stack()
    LogisticRegression, _accuracy_score, _prf = libs

    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    dev_rows = split_rows.get("dev")
    test_rows = split_rows["test"]

    tokenizer, model = create_probe_components(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_list: list[str]
    selected_layers: tuple[int, ...]
    gold_dev: list[int] | None
    with tempfile.TemporaryDirectory(prefix="probe-cache-", dir=output_dir) as cache_root:
        cache_dir = Path(cache_root)

        if task == TASK_SOURCE_MATERIALITY:
            label_list = list(SOURCE_MATERIALITY_LABELS)
            selected_layers = _sequence_hidden_state_cache(
                train_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="train",
                layers=layers,
            )
            if dev_rows:
                _sequence_hidden_state_cache(
                    dev_rows,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    split_name="dev",
                    layers=selected_layers,
                )
            _sequence_hidden_state_cache(
                test_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="test",
                layers=selected_layers,
            )
            gold_train = [label_list.index(str(row["label"])) for row in train_rows]
            gold_dev = [label_list.index(str(row["label"])) for row in dev_rows] if dev_rows else None
            gold_test = [label_list.index(str(row["label"])) for row in test_rows]
        elif task == TASK_SPAN_ROLE:
            label_list = list(SPAN_ROLE_LABELS)
            selected_layers = _token_hidden_state_cache(
                train_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="train",
                layers=layers,
            )
            if dev_rows:
                _token_hidden_state_cache(
                    dev_rows,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    split_name="dev",
                    layers=selected_layers,
                )
            _token_hidden_state_cache(
                test_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="test",
                layers=selected_layers,
            )
            gold_train = [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="train")]
            gold_dev = (
                [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="dev")]
                if dev_rows
                else None
            )
            gold_test = [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="test")]
        else:
            raise ValueError(f"Unsupported probe task: {task!r}")

        layer_summaries: list[dict[str, Any]] = []
        best_summary: dict[str, Any] | None = None
        best_score = float("-inf")
        selection_key = "dev_f1_macro" if gold_dev is not None else "test_f1_macro"

        for layer_index in selected_layers:
            clf = _create_logistic_regression(LogisticRegression)
            clf.fit(_load_cached_layer_vectors(cache_dir, split_name="train", layer_index=layer_index), gold_train)

            test_pred = clf.predict(_load_cached_layer_vectors(cache_dir, split_name="test", layer_index=layer_index)).tolist()
            test_metrics = _compute_classification_metrics(
                gold_test,
                test_pred,
                labels=list(range(len(label_list))),
            )
            summary: dict[str, Any] = {
                "layer": layer_index,
                "task": task,
                "model_name": model_name,
                "label_list": label_list,
                "test_metrics": test_metrics,
                "test_f1_macro": test_metrics["f1_macro"],
            }
            if gold_dev is not None:
                dev_pred = clf.predict(
                    _load_cached_layer_vectors(cache_dir, split_name="dev", layer_index=layer_index)
                ).tolist()
                dev_metrics = _compute_classification_metrics(
                    gold_dev,
                    dev_pred,
                    labels=list(range(len(label_list))),
                )
                summary["dev_metrics"] = dev_metrics
                summary["dev_f1_macro"] = dev_metrics["f1_macro"]

            layer_path = output_dir / f"layer_{layer_index:02d}.metrics.json"
            layer_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            joblib.dump(clf, output_dir / f"layer_{layer_index:02d}.probe.joblib")
            layer_summaries.append(summary)

            score = float(summary[selection_key])
            if score > best_score:
                best_score = score
                best_summary = summary

    if best_summary is None:
        raise RuntimeError("Probe experiment produced no layer summaries.")

    summary = {
        "task": task,
        "model_name": model_name,
        "selection_key": selection_key,
        "best_layer": best_summary["layer"],
        "selected_layers": list(selected_layers),
        "best_metrics": best_summary,
        "layer_rankings": sorted(
            layer_summaries,
            key=lambda row: float(row[selection_key]),
            reverse=True,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": task,
                "model_name": model_name,
                "max_length": max_length,
                "layers": list(selected_layers),
                "label_list": label_list,
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_run_reports(
        output_dir=output_dir,
        repo_root=Path(__file__).resolve().parents[3],
        title="Probe Experiment Run",
        payload=summary,
        extra={
            "input_dir": str(input_dir),
            "selection_key": selection_key,
            "max_length": max_length,
        },
    )
    return summary
