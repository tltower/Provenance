from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import create_probe_components
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


def _sequence_hidden_state_vectors(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
) -> dict[int, Any]:
    _joblib, np, torch, _libs = _require_probe_stack()
    layer_vectors: dict[int, list[Any]] = {}
    model.eval()
    device = next(model.parameters()).device
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
        attention_mask = tokenized["attention_mask"][0].detach().cpu().numpy().astype(bool)
        for layer_index, hidden_state in enumerate(hidden_states):
            vectors = hidden_state[0].detach().cpu().numpy()
            pooled = vectors[attention_mask].mean(axis=0)
            layer_vectors.setdefault(layer_index, []).append(pooled)
    return {layer: np.vstack(vectors) for layer, vectors in layer_vectors.items()}


def _token_hidden_state_vectors(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
) -> tuple[dict[int, Any], list[str]]:
    _joblib, np, torch, _libs = _require_probe_stack()
    layer_vectors: dict[int, list[Any]] = {}
    labels: list[str] = []
    model.eval()
    device = next(model.parameters()).device
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
        word_ids = tokenized.word_ids(0)
        first_positions: dict[int, int] = {}
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in first_positions:
                continue
            first_positions[word_id] = position
        for word_id, position in sorted(first_positions.items()):
            labels.append(str(row["labels"][word_id]))
            for layer_index, hidden_state in enumerate(hidden_states):
                vector = hidden_state[0, position].detach().cpu().numpy()
                layer_vectors.setdefault(layer_index, []).append(vector)
    return {layer: np.vstack(vectors) for layer, vectors in layer_vectors.items()}, labels


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
) -> dict[str, Any]:
    joblib, np, _torch, libs = _require_probe_stack()
    LogisticRegression, _accuracy_score, _prf = libs

    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    dev_rows = split_rows.get("dev")
    test_rows = split_rows["test"]

    tokenizer, model = create_probe_components(model_name)

    if task == TASK_SOURCE_MATERIALITY:
        label_list = list(SOURCE_MATERIALITY_LABELS)
        train_vectors = _sequence_hidden_state_vectors(train_rows, tokenizer=tokenizer, model=model, max_length=max_length)
        dev_vectors = (
            _sequence_hidden_state_vectors(dev_rows, tokenizer=tokenizer, model=model, max_length=max_length)
            if dev_rows
            else None
        )
        test_vectors = _sequence_hidden_state_vectors(test_rows, tokenizer=tokenizer, model=model, max_length=max_length)
        gold_train = [label_list.index(str(row["label"])) for row in train_rows]
        gold_dev = [label_list.index(str(row["label"])) for row in dev_rows] if dev_rows else None
        gold_test = [label_list.index(str(row["label"])) for row in test_rows]
    elif task == TASK_SPAN_ROLE:
        label_list = list(SPAN_ROLE_LABELS)
        train_vectors, train_labels = _token_hidden_state_vectors(
            train_rows,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
        )
        if dev_rows:
            dev_vectors, dev_labels = _token_hidden_state_vectors(
                dev_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
            )
        else:
            dev_vectors = None
            dev_labels = None
        test_vectors, test_labels = _token_hidden_state_vectors(
            test_rows,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
        )
        gold_train = [label_list.index(label) for label in train_labels]
        gold_dev = [label_list.index(label) for label in dev_labels] if dev_labels else None
        gold_test = [label_list.index(label) for label in test_labels]
    else:
        raise ValueError(f"Unsupported probe task: {task!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_summaries: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_score = float("-inf")
    selection_key = "dev_f1_macro" if gold_dev is not None else "test_f1_macro"

    for layer_index in sorted(train_vectors):
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        clf.fit(train_vectors[layer_index], gold_train)

        test_pred = clf.predict(test_vectors[layer_index]).tolist()
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
        if gold_dev is not None and dev_vectors is not None:
            dev_pred = clf.predict(dev_vectors[layer_index]).tolist()
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
                "label_list": label_list,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary
