from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import (
    create_sequence_classification_components,
)
from memex_research.classifier_research.splits import load_jsonl_splits
from memex_research.classifier_research.tasks import SOURCE_MATERIALITY_LABELS


def _require_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            precision_recall_fscore_support,
        )
        from transformers import Trainer, TrainingArguments  # type: ignore[import-not-found]
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
    seed: int = 17,
) -> dict[str, Any]:
    _np, _torch, _metrics, trainer_libs = _require_training_stack()
    Trainer, TrainingArguments = trainer_libs

    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    eval_rows = split_rows.get("dev") or split_rows["test"]
    test_rows = split_rows["test"]

    tokenizer, model = create_sequence_classification_components(
        model_name=model_name,
        labels=SOURCE_MATERIALITY_LABELS,
    )
    encoded_train = _encode_source_records(train_rows, tokenizer, max_length=max_length)
    encoded_eval = _encode_source_records(eval_rows, tokenizer, max_length=max_length)
    encoded_test = _encode_source_records(test_rows, tokenizer, max_length=max_length)

    train_dataset = _build_dataset(encoded_train)
    eval_dataset = _build_dataset(encoded_eval)
    test_dataset = _build_dataset(encoded_test)

    trainer_dir = output_dir / "trainer"
    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_build_metrics_fn(SOURCE_MATERIALITY_LABELS),
    )
    trainer.train()

    prediction_output = trainer.predict(test_dataset)
    test_metrics = _build_metrics_fn(SOURCE_MATERIALITY_LABELS)(
        (prediction_output.predictions, prediction_output.label_ids)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
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
    return metrics
