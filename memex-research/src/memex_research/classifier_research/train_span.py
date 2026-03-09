from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import (
    create_token_classification_components,
)
from memex_research.classifier_research.splits import load_jsonl_splits
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
            labels=list(range(len(label_names))),
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


def train_span_classifier(
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
    DataCollatorForTokenClassification, Trainer, TrainingArguments = trainer_libs

    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    eval_rows = split_rows.get("dev") or split_rows["test"]
    test_rows = split_rows["test"]

    tokenizer, model = create_token_classification_components(model_name=model_name, labels=SPAN_ROLE_LABELS)
    encoded_train = _encode_span_records(train_rows, tokenizer, max_length=max_length)
    encoded_eval = _encode_span_records(eval_rows, tokenizer, max_length=max_length)
    encoded_test = _encode_span_records(test_rows, tokenizer, max_length=max_length)

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
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=_build_metrics_fn(SPAN_ROLE_LABELS),
    )
    trainer.train()

    prediction_output = trainer.predict(test_dataset)
    test_metrics = _build_metrics_fn(SPAN_ROLE_LABELS)(
        (prediction_output.predictions, prediction_output.label_ids)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    pred_label_ids = prediction_output.predictions.argmax(axis=-1).tolist()
    prediction_rows: list[dict[str, Any]] = []
    for row, pred_ids in zip(encoded_test, pred_label_ids, strict=True):
        prediction_rows.append(
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "gold_labels": row["gold_labels"],
                "predicted_labels": _project_token_predictions(row, pred_ids),
            }
        )

    metrics = {
        "task": "span_role",
        "dataset": "pe",
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
                "task": "span_role",
                "dataset": "pe",
                "model_name": model_name,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "seed": seed,
                "label_list": list(SPAN_ROLE_LABELS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return metrics
