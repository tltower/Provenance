from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from memex_research.classifier_research.hf_models import create_token_classification_components
from memex_research.classifier_research.multidataset import infer_span_label_list
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import ensure_dev_split, load_jsonl_splits
from memex_research.classifier_research.status import (
    HFTrainerHeartbeatCallback,
    emit_run_status,
    write_run_analytics,
)
from memex_research.classifier_research.tasks import SPAN_ROLE_LABELS


def _require_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            precision_recall_fscore_support,
        )
        from transformers import (  # type: ignore[import-not-found]
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Training dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return np, torch, (accuracy_score, precision_recall_fscore_support), (
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )


def _build_metrics_fn(label_names: tuple[str, ...]) -> Any:
    np, _torch, metrics_libs, _trainer_libs = _require_training_stack()
    accuracy_score, precision_recall_fscore_support = metrics_libs
    label_to_id = {label: index for index, label in enumerate(SPAN_ROLE_LABELS)}
    metric_label_ids = [label_to_id[label] for label in label_names if label in label_to_id]

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        gold_flat: list[int] = []
        pred_flat: list[int] = []
        for pred_row, gold_row in zip(predictions, labels, strict=True):
            for pred_label, gold_label in zip(pred_row, gold_row, strict=True):
                if int(gold_label) == -100:
                    continue
                gold_flat.append(int(gold_label))
                pred_flat.append(int(pred_label))
        precision, recall, f1, _support = precision_recall_fscore_support(
            gold_flat,
            pred_flat,
            labels=metric_label_ids,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(gold_flat, pred_flat)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    return compute_metrics


def _encode_span_records(records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(SPAN_ROLE_LABELS)}
    encoded: list[dict[str, Any]] = []
    for row in records:
        tokenized = tokenizer(
            list(row["tokens"]),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        word_ids = tokenized.word_ids()
        aligned_labels: list[int] = []
        previous_word_id: int | None = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(label_to_id[str(row["labels"][word_id])])
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id
        encoded.append(
            {
                "id": str(row["id"]),
                "dataset": row.get("dataset"),
                "source_dataset": row.get("source_dataset"),
                "tokens": list(row["tokens"]),
                "gold_labels": list(row["labels"]),
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "word_ids": word_ids,
                "labels": aligned_labels,
            }
        )
    return encoded


def _build_dataset(records: list[dict[str, Any]]) -> Any:
    _np, torch, _metrics, _trainer_libs = _require_training_stack()

    class EncodedTokenDataset:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows[index]
            return {
                "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(row["labels"], dtype=torch.long),
            }

    return EncodedTokenDataset(records)


def _project_token_predictions(encoded_row: dict[str, Any], pred_ids: list[int]) -> list[str]:
    label_lookup = {index: label for index, label in enumerate(SPAN_ROLE_LABELS)}
    projected = ["OTHER"] * len(encoded_row["tokens"])
    seen_word_ids: set[int] = set()
    for index, word_id in enumerate(encoded_row["word_ids"]):
        if word_id is None or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        projected[word_id] = label_lookup[int(pred_ids[index])]
    return projected


def _compute_metrics_from_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    label_names: Sequence[str],
    prediction_label_space: Sequence[str] | None = None,
) -> dict[str, float]:
    _np, _torch, metrics_libs, _trainer_libs = _require_training_stack()
    accuracy_score, precision_recall_fscore_support = metrics_libs
    full_label_space = tuple(prediction_label_space or label_names)
    label_to_id = {label: index for index, label in enumerate(full_label_space)}
    metric_label_ids = [label_to_id[label] for label in label_names if label in label_to_id]
    gold_flat: list[int] = []
    pred_flat: list[int] = []
    for row in prediction_rows:
        for gold_label, pred_label in zip(row["gold_labels"], row["predicted_labels"], strict=True):
            if str(gold_label) not in label_to_id:
                continue
            gold_flat.append(label_to_id[str(gold_label)])
            pred_flat.append(label_to_id.get(str(pred_label), -1))
    if not gold_flat:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }
    precision, recall, f1, _support = precision_recall_fscore_support(
        gold_flat,
        pred_flat,
        labels=metric_label_ids,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold_flat, pred_flat)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _count_span_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {label: 0 for label in SPAN_ROLE_LABELS}
    token_count = 0
    for row in records:
        labels = [str(label) for label in row["labels"]]
        token_count += len(labels)
        for label in labels:
            if label in counts:
                counts[label] += 1
    return {
        "record_count": len(records),
        "token_count": token_count,
        "label_counts": counts,
    }


def _create_training_args(TrainingArguments: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy"
    if strategy_key not in parameters and "eval_strategy" in parameters:
        kwargs["eval_strategy"] = kwargs.pop(strategy_key)
    return TrainingArguments(**kwargs)


def train_span_classifier(
    *,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    dataset_name: str = "pe",
    max_length: int = 256,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: float = 3.0,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    seed: int = 17,
) -> dict[str, Any]:
    requested_dataset_name = dataset_name
    _np, _torch, _metrics, trainer_libs = _require_training_stack()
    DataCollatorForTokenClassification, Trainer, TrainingArguments = trainer_libs

    split_rows = ensure_dev_split(load_jsonl_splits(input_dir), dev_fraction=0.1, seed=seed)
    train_rows = split_rows["train"]
    eval_rows = split_rows["dev"]
    test_rows = split_rows["test"]
    metric_label_list = infer_span_label_list([*train_rows, *eval_rows, *test_rows]) or SPAN_ROLE_LABELS
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "train": len(train_rows),
        "eval": len(eval_rows),
        "test": len(test_rows),
    }
    split_analytics = {
        "train": _count_span_labels(train_rows),
        "eval": _count_span_labels(eval_rows),
        "test": _count_span_labels(test_rows),
    }
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="loading_data",
        message="Loaded span-role benchmark splits",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        split_counts=split_counts,
        split_analytics=split_analytics,
        input_dir=str(input_dir),
    )
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="loading_model",
        message="Loading tokenizer and token classifier model",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        split_counts=split_counts,
    )

    tokenizer, model = create_token_classification_components(model_name=model_name, labels=SPAN_ROLE_LABELS)
    encoded_train = _encode_span_records(train_rows, tokenizer, max_length=max_length)
    encoded_eval = _encode_span_records(eval_rows, tokenizer, max_length=max_length)
    encoded_test = _encode_span_records(test_rows, tokenizer, max_length=max_length)
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="encoding_complete",
        message="Finished encoding span-role records",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        encoded_counts={
            "train": len(encoded_train),
            "eval": len(encoded_eval),
            "test": len(encoded_test),
        },
        max_length=max_length,
    )

    train_dataset = _build_dataset(encoded_train)
    eval_dataset = _build_dataset(encoded_eval)
    test_dataset = _build_dataset(encoded_test)

    trainer_dir = output_dir / "trainer"
    training_args = _create_training_args(
        TrainingArguments,
        output_dir=str(trainer_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
    heartbeat_callback = HFTrainerHeartbeatCallback(
        output_dir=output_dir,
        task="span_role",
        dataset_name=requested_dataset_name,
        model_name=model_name,
        split_counts=split_counts,
    )
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="trainer_ready",
        message="Configured Hugging Face trainer",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        split_counts=split_counts,
        training_args={
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
        },
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=_build_metrics_fn(metric_label_list),
        callbacks=[heartbeat_callback],
    )
    trainer.train()

    heartbeat_callback.begin_prediction_stage("predict", total_examples=len(encoded_test))
    prediction_output = trainer.predict(test_dataset)
    test_metrics = _build_metrics_fn(metric_label_list)(
        (prediction_output.predictions, prediction_output.label_ids)
    )

    model_dir = output_dir / "model"
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="saving_artifacts",
        message="Saving trained model artifacts",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        model_dir=str(model_dir),
    )
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    pred_label_ids = prediction_output.predictions.argmax(axis=-1).tolist()
    prediction_rows: list[dict[str, Any]] = []
    for row, pred_ids in zip(encoded_test, pred_label_ids, strict=True):
        prediction_rows.append(
            {
                "id": row["id"],
                "dataset": row.get("source_dataset", row.get("dataset")),
                "tokens": row["tokens"],
                "gold_labels": row["gold_labels"],
                "predicted_labels": _project_token_predictions(row, pred_ids),
            }
        )

    grouped_prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        row_dataset_name = str(row.get("dataset") or requested_dataset_name)
        grouped_prediction_rows.setdefault(row_dataset_name, []).append(row)

    metrics = {
        "task": "span_role",
        "dataset": requested_dataset_name,
        "model_name": model_name,
        "metric_label_list": list(metric_label_list),
        "model_label_list": list(SPAN_ROLE_LABELS),
        "test_metrics": test_metrics,
        "per_dataset_test_metrics": {
            name: _compute_metrics_from_prediction_rows(
                rows,
                label_names=infer_span_label_list(rows) or metric_label_list,
                prediction_label_space=SPAN_ROLE_LABELS,
            )
            for name, rows in grouped_prediction_rows.items()
        },
        "counts": {
            "train": len(encoded_train),
            "eval": len(encoded_eval),
            "test": len(encoded_test),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": "span_role",
                "dataset": requested_dataset_name,
                "model_name": model_name,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "seed": seed,
                "label_list": list(SPAN_ROLE_LABELS),
                "metric_label_list": list(metric_label_list),
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_run_analytics(
        output_dir,
        filename="train_analytics.json",
        payload={
            "task": "span_role",
            "dataset": requested_dataset_name,
            "model_name": model_name,
            "split_counts": split_counts,
            "split_analytics": split_analytics,
            "metric_label_list": list(metric_label_list),
            "trainer_state": {
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "log_history": trainer.state.log_history,
            },
            "test_metrics": test_metrics,
            "per_dataset_test_metrics": metrics["per_dataset_test_metrics"],
            "prediction_row_count": len(prediction_rows),
        },
    )
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="complete",
        message="Completed span-role training run",
        task="span_role",
        dataset=requested_dataset_name,
        model_name=model_name,
        metrics=test_metrics,
        model_dir=str(model_dir),
        per_dataset_test_metrics=metrics["per_dataset_test_metrics"],
    )
    write_run_reports(
        output_dir=output_dir,
        repo_root=Path(__file__).resolve().parents[3],
        title="Span Role Training Run",
        payload=metrics,
        extra={
            "input_dir": str(input_dir),
            "model_dir": str(model_dir),
            "problem_type": "token_classification",
        },
    )
    return metrics
