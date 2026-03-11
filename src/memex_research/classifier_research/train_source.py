from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import (
    create_sequence_classification_components,
)
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import load_jsonl_splits
from memex_research.classifier_research.status import (
    HFTrainerHeartbeatCallback,
    emit_run_status,
    write_run_analytics,
)
from memex_research.classifier_research.tasks import SOURCE_MATERIALITY_LABELS


def _require_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            precision_recall_fscore_support,
        )
        from transformers import (  # type: ignore[import-not-found]
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Training dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return np, torch, (accuracy_score, precision_recall_fscore_support), (Trainer, TrainingArguments)


def _build_metrics_fn(label_names: tuple[str, ...]) -> Any:
    np, _torch, metrics_libs, _trainer_libs = _require_training_stack()
    accuracy_score, precision_recall_fscore_support = metrics_libs

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    return compute_metrics


def _encode_source_records(records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(SOURCE_MATERIALITY_LABELS)}
    encoded: list[dict[str, Any]] = []
    for row in records:
        tokenized = tokenizer(
            str(row["context"]),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded.append(
            {
                "id": str(row["id"]),
                "context": str(row["context"]),
                "gold_label": str(row["label"]),
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": label_to_id[str(row["label"])],
            }
        )
    return encoded


def _build_dataset(records: list[dict[str, Any]]) -> Any:
    _np, torch, _metrics, _trainer_libs = _require_training_stack()

    class EncodedSequenceDataset:
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

    return EncodedSequenceDataset(records)


def _count_source_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {label: 0 for label in SOURCE_MATERIALITY_LABELS}
    for row in records:
        label = str(row["label"])
        if label in counts:
            counts[label] += 1
    return {
        "record_count": len(records),
        "label_counts": counts,
    }


def _create_training_args(TrainingArguments: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy"
    if strategy_key not in parameters and "eval_strategy" in parameters:
        kwargs["eval_strategy"] = kwargs.pop(strategy_key)
    return TrainingArguments(**kwargs)


def train_source_classifier(
    *,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: float = 3.0,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    seed: int = 17,
) -> dict[str, Any]:
    _np, _torch, _metrics, trainer_libs = _require_training_stack()
    Trainer, TrainingArguments = trainer_libs

    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    eval_rows = split_rows.get("dev") or split_rows["test"]
    test_rows = split_rows["test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "train": len(train_rows),
        "eval": len(eval_rows),
        "test": len(test_rows),
    }
    split_analytics = {
        "train": _count_source_labels(train_rows),
        "eval": _count_source_labels(eval_rows),
        "test": _count_source_labels(test_rows),
    }
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="loading_data",
        message="Loaded source-materiality benchmark splits",
        task="source_materiality",
        dataset="scicite",
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
        message="Loading tokenizer and classifier model",
        task="source_materiality",
        dataset="scicite",
        model_name=model_name,
        split_counts=split_counts,
    )

    tokenizer, model = create_sequence_classification_components(
        model_name=model_name,
        labels=SOURCE_MATERIALITY_LABELS,
    )
    encoded_train = _encode_source_records(train_rows, tokenizer, max_length=max_length)
    encoded_eval = _encode_source_records(eval_rows, tokenizer, max_length=max_length)
    encoded_test = _encode_source_records(test_rows, tokenizer, max_length=max_length)
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="encoding_complete",
        message="Finished encoding source-materiality records",
        task="source_materiality",
        dataset="scicite",
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
        task="source_materiality",
        dataset_name="scicite",
        model_name=model_name,
        split_counts=split_counts,
    )
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="trainer_ready",
        message="Configured Hugging Face trainer",
        task="source_materiality",
        dataset="scicite",
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
        compute_metrics=_build_metrics_fn(SOURCE_MATERIALITY_LABELS),
        callbacks=[heartbeat_callback],
    )
    trainer.train()

    heartbeat_callback.begin_prediction_stage("predict", total_examples=len(encoded_test))
    prediction_output = trainer.predict(test_dataset)
    test_metrics = _build_metrics_fn(SOURCE_MATERIALITY_LABELS)(
        (prediction_output.predictions, prediction_output.label_ids)
    )

    model_dir = output_dir / "model"
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="saving_artifacts",
        message="Saving trained model artifacts",
        task="source_materiality",
        dataset="scicite",
        model_name=model_name,
        model_dir=str(model_dir),
    )
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    label_lookup = {index: label for index, label in enumerate(SOURCE_MATERIALITY_LABELS)}
    predicted_ids = prediction_output.predictions.argmax(axis=-1).tolist()
    prediction_rows: list[dict[str, Any]] = []
    for row, pred_id in zip(encoded_test, predicted_ids, strict=True):
        prediction_rows.append(
            {
                "id": row["id"],
                "context": row["context"],
                "gold_label": row["gold_label"],
                "predicted_label": label_lookup[int(pred_id)],
            }
        )

    metrics = {
        "task": "source_materiality",
        "dataset": "scicite",
        "model_name": model_name,
        "test_metrics": test_metrics,
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
                "task": "source_materiality",
                "dataset": "scicite",
                "model_name": model_name,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "seed": seed,
                "label_list": list(SOURCE_MATERIALITY_LABELS),
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
            "task": "source_materiality",
            "dataset": "scicite",
            "model_name": model_name,
            "split_counts": split_counts,
            "split_analytics": split_analytics,
            "trainer_state": {
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "log_history": trainer.state.log_history,
            },
            "test_metrics": test_metrics,
            "prediction_row_count": len(prediction_rows),
        },
    )
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="train",
        phase="complete",
        message="Completed source-materiality training run",
        task="source_materiality",
        dataset="scicite",
        model_name=model_name,
        metrics=test_metrics,
        model_dir=str(model_dir),
    )
    write_run_reports(
        output_dir=output_dir,
        repo_root=Path(__file__).resolve().parents[3],
        title="Source Materiality Training Run",
        payload=metrics,
        extra={
            "input_dir": str(input_dir),
            "model_dir": str(model_dir),
            "problem_type": "sequence_classification",
        },
    )
    return metrics
