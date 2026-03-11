from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memex_research.classifier_research.hf_models import get_tokenizer
from memex_research.classifier_research.multidataset import infer_span_label_list
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import ensure_dev_split, load_jsonl_splits
from memex_research.classifier_research.status import emit_run_status, write_run_analytics
from memex_research.classifier_research.tasks import SPAN_ROLE_LABELS


def _require_training_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            precision_recall_fscore_support,
        )
        from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModel,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Training dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return (
        np,
        torch,
        (accuracy_score, precision_recall_fscore_support),
        (DataLoader, Dataset),
        (AutoModel, get_linear_schedule_with_warmup),
    )


@dataclass(frozen=True)
class SpanDatasetSpec:
    name: str
    input_dir: Path


def _emit_multi_span_progress(
    output_dir: Path,
    *,
    phase: str,
    message: str,
    model_name: str,
    dataset_specs: Sequence[SpanDatasetSpec],
    **extra: Any,
) -> None:
    emit_run_status(
        output_dir,
        filename="train_status.json",
        prefix="multi-train",
        phase=phase,
        message=message,
        task="span_role_multidataset",
        model_name=model_name,
        datasets=[spec.name for spec in dataset_specs],
        dataset_inputs={spec.name: str(spec.input_dir) for spec in dataset_specs},
        **extra,
    )


def _validate_dataset_specs(dataset_specs: Sequence[SpanDatasetSpec]) -> None:
    if len(dataset_specs) < 2:
        raise ValueError("Expected at least two span datasets for multi-dataset training.")

    seen_names: set[str] = set()
    for spec in dataset_specs:
        if spec.name in seen_names:
            raise ValueError(f"Duplicate dataset name in multi-dataset training config: {spec.name}")
        seen_names.add(spec.name)
        if not spec.input_dir.exists():
            raise FileNotFoundError(f"Prepared benchmark directory does not exist: {spec.input_dir}")


def _target_training_steps(loader_length: int, num_train_epochs: float) -> int:
    if loader_length <= 0:
        raise ValueError("Expected a non-empty training loader for multi-dataset span training.")
    if num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive.")
    return max(1, math.ceil(loader_length * num_train_epochs))


class _EncodedSpanDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "row_index": index,
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "dataset_name": row["dataset_name"],
        }


def _collate_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _np, torch, _metrics, _data, _transformers = _require_training_stack()
    return {
        "row_indices": torch.tensor([int(item["row_index"]) for item in batch], dtype=torch.long),
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "dataset_names": [str(item["dataset_name"]) for item in batch],
    }


def _set_seed(seed: int) -> None:
    _np, torch, _metrics, _data, _transformers = _require_training_stack()
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_device() -> Any:
    _np, torch, _metrics, _data, _transformers = _require_training_stack()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_span_records(
    *,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    label_lists: Mapping[str, Sequence[str]],
    max_length: int,
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for row in records:
        dataset_name = str(row["dataset_name"])
        label_list = tuple(label_lists[dataset_name])
        label_to_id = {label: index for index, label in enumerate(label_list)}
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
                "dataset_name": dataset_name,
                "tokens": list(row["tokens"]),
                "gold_labels": list(row["labels"]),
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "word_ids": word_ids,
                "labels": aligned_labels,
            }
        )
    return encoded


def _project_token_predictions(
    *,
    encoded_row: Mapping[str, Any],
    pred_ids: Sequence[int],
    label_list: Sequence[str],
) -> list[str]:
    projected = ["OTHER"] * len(encoded_row["tokens"])
    seen_word_ids: set[int] = set()
    for index, word_id in enumerate(encoded_row["word_ids"]):
        if word_id is None or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        projected[word_id] = str(label_list[int(pred_ids[index])])
    return projected


def _flatten_label_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_list: Sequence[str],
) -> tuple[list[int], list[int]]:
    label_to_id = {label: index for index, label in enumerate(label_list)}
    gold_flat: list[int] = []
    pred_flat: list[int] = []
    for row in rows:
        for gold_label, pred_label in zip(row["gold_labels"], row["predicted_labels"], strict=True):
            if str(gold_label) not in label_to_id or str(pred_label) not in label_to_id:
                continue
            gold_flat.append(label_to_id[str(gold_label)])
            pred_flat.append(label_to_id[str(pred_label)])
    return gold_flat, pred_flat


def _compute_metrics_from_prediction_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    label_list: Sequence[str],
) -> dict[str, float]:
    _np, _torch, metrics_libs, _data, _transformers = _require_training_stack()
    accuracy_score, precision_recall_fscore_support = metrics_libs
    gold_flat, pred_flat = _flatten_label_rows(prediction_rows, label_list=label_list)
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
        labels=list(range(len(label_list))),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold_flat, pred_flat)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _count_span_labels(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _group_prediction_rows_by_dataset(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[str(row["dataset_name"])].append(dict(row))
    return grouped


def _compute_per_dataset_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    label_lists: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    grouped = _group_prediction_rows_by_dataset(prediction_rows)
    return {
        dataset_name: _compute_metrics_from_prediction_rows(rows, label_list=label_lists[dataset_name])
        for dataset_name, rows in grouped.items()
    }


class SharedBackboneMultiHeadSpanModel:
    def __init__(self, *, model_name: str, dataset_label_lists: Mapping[str, Sequence[str]]) -> None:
        _np, torch, _metrics, _data, transformers_lib = _require_training_stack()
        AutoModel, _scheduler = transformers_lib
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        dropout_prob = float(
            getattr(self.encoder.config, "hidden_dropout_prob", getattr(self.encoder.config, "dropout", 0.1))
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.heads = torch.nn.ModuleDict(
            {
                dataset_name: torch.nn.Linear(hidden_size, len(label_list))
                for dataset_name, label_list in dataset_label_lists.items()
            }
        )

    def parameters(self) -> Any:
        return list(self.encoder.parameters()) + list(self.dropout.parameters()) + list(self.heads.parameters())

    def train(self) -> None:
        self.encoder.train()
        self.dropout.train()
        self.heads.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.dropout.eval()
        self.heads.eval()

    def to(self, device: Any) -> None:
        self.encoder.to(device)
        self.dropout.to(device)
        self.heads.to(device)

    def encode(self, *, input_ids: Any, attention_mask: Any) -> Any:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)


def _load_dataset_rows(
    *,
    dataset_specs: Sequence[SpanDatasetSpec],
    seed: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, tuple[str, ...]]]:
    split_rows_by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    label_lists: dict[str, tuple[str, ...]] = {}
    for spec in dataset_specs:
        normalized_splits = ensure_dev_split(load_jsonl_splits(spec.input_dir), dev_fraction=0.1, seed=seed)
        for split_name, rows in normalized_splits.items():
            normalized_splits[split_name] = [
                {**dict(row), "dataset_name": spec.name}
                for row in rows
            ]
        split_rows_by_dataset[spec.name] = normalized_splits
        label_lists[spec.name] = infer_span_label_list(
            [
                row
                for rows in normalized_splits.values()
                for row in rows
            ]
        )
        if not label_lists[spec.name]:
            raise ValueError(f"Dataset {spec.name!r} did not yield any usable span labels.")
    return split_rows_by_dataset, label_lists


def _combine_split_rows(
    split_rows_by_dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for dataset_name in split_rows_by_dataset:
        combined.extend(dict(row) for row in split_rows_by_dataset[dataset_name].get(split_name, []))
    return combined


def _evaluate_model(
    *,
    model: SharedBackboneMultiHeadSpanModel,
    encoded_rows: list[dict[str, Any]],
    dataset_label_lists: Mapping[str, Sequence[str]],
    batch_size: int,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, float], float]:
    _np, torch, _metrics, data_libs, _transformers = _require_training_stack()
    DataLoader, _Dataset = data_libs
    loader = DataLoader(_EncodedSpanDataset(encoded_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)
    prediction_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden_states = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            row_indices = batch["row_indices"].tolist()
            dataset_names = list(batch["dataset_names"])
            for batch_index, row_index in enumerate(row_indices):
                dataset_name = dataset_names[batch_index]
                logits = model.heads[dataset_name](hidden_states[batch_index : batch_index + 1])[0]
                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                encoded_row = encoded_rows[row_index]
                prediction_rows.append(
                    {
                        "id": encoded_row["id"],
                        "dataset_name": dataset_name,
                        "tokens": encoded_row["tokens"],
                        "gold_labels": encoded_row["gold_labels"],
                        "predicted_labels": _project_token_predictions(
                            encoded_row=encoded_row,
                            pred_ids=pred_ids,
                            label_list=dataset_label_lists[dataset_name],
                        ),
                    }
                )

    per_dataset_metrics = _compute_per_dataset_metrics(
        prediction_rows,
        label_lists=dataset_label_lists,
    )
    overall_metrics = _compute_metrics_from_prediction_rows(
        prediction_rows,
        label_list=SPAN_ROLE_LABELS,
    )
    mean_dataset_f1 = _mean([metrics["f1_macro"] for metrics in per_dataset_metrics.values()])
    return prediction_rows, per_dataset_metrics, overall_metrics, mean_dataset_f1


def train_multi_span_classifier(
    *,
    dataset_specs: Sequence[SpanDatasetSpec],
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
    _validate_dataset_specs(dataset_specs)
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_multi_span_progress(
        output_dir,
        phase="loading_data",
        message="Loading prepared span-role benchmarks",
        model_name=model_name,
        dataset_specs=dataset_specs,
        max_length=max_length,
    )
    _set_seed(seed)
    _np, torch, _metrics, data_libs, transformers_lib = _require_training_stack()
    DataLoader, _Dataset = data_libs
    _AutoModel, get_linear_schedule_with_warmup = transformers_lib

    split_rows_by_dataset, dataset_label_lists = _load_dataset_rows(dataset_specs=dataset_specs, seed=seed)
    split_counts_by_dataset = {
        dataset_name: {split_name: len(rows) for split_name, rows in split_rows.items()}
        for dataset_name, split_rows in split_rows_by_dataset.items()
    }
    split_analytics_by_dataset = {
        dataset_name: {
            split_name: _count_span_labels(rows)
            for split_name, rows in split_rows.items()
        }
        for dataset_name, split_rows in split_rows_by_dataset.items()
    }
    _emit_multi_span_progress(
        output_dir,
        phase="loading_model",
        message="Loading tokenizer for shared-backbone run",
        model_name=model_name,
        dataset_specs=dataset_specs,
        split_counts_by_dataset=split_counts_by_dataset,
    )
    tokenizer = get_tokenizer(model_name)

    _emit_multi_span_progress(
        output_dir,
        phase="encoding",
        message="Encoding multi-dataset span-role records",
        model_name=model_name,
        dataset_specs=dataset_specs,
        split_counts_by_dataset=split_counts_by_dataset,
        split_analytics_by_dataset=split_analytics_by_dataset,
        dataset_label_lists={name: list(labels) for name, labels in dataset_label_lists.items()},
    )
    encoded_train = _encode_span_records(
        records=_combine_split_rows(split_rows_by_dataset, split_name="train"),
        tokenizer=tokenizer,
        label_lists=dataset_label_lists,
        max_length=max_length,
    )
    encoded_eval = _encode_span_records(
        records=_combine_split_rows(split_rows_by_dataset, split_name="dev"),
        tokenizer=tokenizer,
        label_lists=dataset_label_lists,
        max_length=max_length,
    )
    encoded_test = _encode_span_records(
        records=_combine_split_rows(split_rows_by_dataset, split_name="test"),
        tokenizer=tokenizer,
        label_lists=dataset_label_lists,
        max_length=max_length,
    )
    _emit_multi_span_progress(
        output_dir,
        phase="encoding_complete",
        message="Finished encoding multi-dataset span-role records",
        model_name=model_name,
        dataset_specs=dataset_specs,
        counts={
            "train": len(encoded_train),
            "eval": len(encoded_eval),
            "test": len(encoded_test),
        },
        max_length=max_length,
    )

    train_loader = DataLoader(
        _EncodedSpanDataset(encoded_train),
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )

    model = SharedBackboneMultiHeadSpanModel(
        model_name=model_name,
        dataset_label_lists=dataset_label_lists,
    )
    device = _create_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = _target_training_steps(len(train_loader), num_train_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_eval_mean_dataset_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epoch_summaries: list[dict[str, Any]] = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    completed_steps = 0
    epoch_count = max(1, math.ceil(num_train_epochs))
    step_heartbeat_interval = max(1, len(train_loader) // 4)
    _emit_multi_span_progress(
        output_dir,
        phase="trainer_ready",
        message="Configured shared-backbone optimizer and scheduler",
        model_name=model_name,
        dataset_specs=dataset_specs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    _emit_multi_span_progress(
        output_dir,
        phase="train_begin",
        message="Starting multi-dataset span-role training",
        model_name=model_name,
        dataset_specs=dataset_specs,
        split_counts_by_dataset=split_counts_by_dataset,
        counts={
            "train": len(encoded_train),
            "eval": len(encoded_eval),
            "test": len(encoded_test),
        },
        total_steps=total_steps,
        planned_epochs=epoch_count,
        warmup_steps=warmup_steps,
        device=str(device),
    )

    for epoch_index in range(epoch_count):
        if completed_steps >= total_steps:
            break
        model.train()
        epoch_loss_sum = 0.0
        epoch_token_count = 0
        _emit_multi_span_progress(
            output_dir,
            phase="epoch_begin",
            message=f"Starting epoch {epoch_index + 1}",
            model_name=model_name,
            dataset_specs=dataset_specs,
            epoch=epoch_index + 1,
            completed_steps=completed_steps,
            total_steps=total_steps,
        )

        for batch_index, batch in enumerate(train_loader, start=1):
            if completed_steps >= total_steps:
                break
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            dataset_names = list(batch["dataset_names"])

            hidden_states = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            total_loss_sum = torch.tensor(0.0, device=device)
            total_token_count = torch.tensor(0, device=device)

            for dataset_name in sorted(set(dataset_names)):
                row_indices = [index for index, name in enumerate(dataset_names) if name == dataset_name]
                selection = torch.tensor(row_indices, dtype=torch.long, device=device)
                subset_hidden = hidden_states.index_select(0, selection)
                subset_labels = labels.index_select(0, selection)
                logits = model.heads[dataset_name](subset_hidden)
                total_loss_sum = total_loss_sum + loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    subset_labels.reshape(-1),
                )
                total_token_count = total_token_count + (subset_labels != -100).sum()

            if int(total_token_count.item()) == 0:
                continue
            loss = total_loss_sum / total_token_count
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed_steps += 1

            epoch_loss_sum += float(total_loss_sum.detach().cpu().item())
            epoch_token_count += int(total_token_count.detach().cpu().item())
            if (
                completed_steps == 1
                or completed_steps == total_steps
                or batch_index % step_heartbeat_interval == 0
            ):
                _emit_multi_span_progress(
                    output_dir,
                    phase="train_step",
                    message="Completed optimizer step",
                    model_name=model_name,
                    dataset_specs=dataset_specs,
                    epoch=epoch_index + 1,
                    batch_index=batch_index,
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    train_loss_per_token=float(loss.detach().cpu().item()),
                    active_datasets=sorted(set(dataset_names)),
                )

        _eval_rows, per_dataset_eval_metrics, overall_eval_metrics, eval_mean_dataset_f1 = _evaluate_model(
            model=model,
            encoded_rows=encoded_eval,
            dataset_label_lists=dataset_label_lists,
            batch_size=per_device_eval_batch_size,
            device=device,
        )
        epoch_summaries.append(
            {
                "epoch": epoch_index + 1,
                "train_loss_per_token": (epoch_loss_sum / epoch_token_count) if epoch_token_count else 0.0,
                "eval_mean_dataset_f1": eval_mean_dataset_f1,
                "overall_eval_metrics": overall_eval_metrics,
                "per_dataset_eval_metrics": per_dataset_eval_metrics,
            }
        )
        _emit_multi_span_progress(
            output_dir,
            phase="eval",
            message=f"Completed evaluation for epoch {epoch_index + 1}",
            model_name=model_name,
            dataset_specs=dataset_specs,
            epoch=epoch_index + 1,
            completed_steps=completed_steps,
            total_steps=total_steps,
            eval_mean_dataset_f1=eval_mean_dataset_f1,
            overall_eval_metrics=overall_eval_metrics,
            per_dataset_eval_metrics=per_dataset_eval_metrics,
        )

        if eval_mean_dataset_f1 > best_eval_mean_dataset_f1:
            best_eval_mean_dataset_f1 = eval_mean_dataset_f1
            best_state = {
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "heads": copy.deepcopy(model.heads.state_dict()),
                "epoch": epoch_index + 1,
            }
            _emit_multi_span_progress(
                output_dir,
                phase="best_checkpoint",
                message=f"New best checkpoint at epoch {epoch_index + 1}",
                model_name=model_name,
                dataset_specs=dataset_specs,
                epoch=epoch_index + 1,
                eval_mean_dataset_f1=eval_mean_dataset_f1,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a best state.")

    model.encoder.load_state_dict(best_state["encoder"])
    model.heads.load_state_dict(best_state["heads"])

    _emit_multi_span_progress(
        output_dir,
        phase="predicting",
        message="Running held-out test prediction",
        model_name=model_name,
        dataset_specs=dataset_specs,
        best_epoch=int(best_state["epoch"]),
        best_eval_mean_dataset_f1=best_eval_mean_dataset_f1,
    )
    test_prediction_rows, per_dataset_test_metrics, overall_test_metrics, test_mean_dataset_f1 = _evaluate_model(
        model=model,
        encoded_rows=encoded_test,
        dataset_label_lists=dataset_label_lists,
        batch_size=per_device_eval_batch_size,
        device=device,
    )

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    _emit_multi_span_progress(
        output_dir,
        phase="saving_artifacts",
        message="Saving shared encoder and dataset heads",
        model_name=model_name,
        dataset_specs=dataset_specs,
        model_dir=str(model_dir),
    )
    model.encoder.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    torch.save(model.heads.state_dict(), model_dir / "dataset_heads.pt")
    (model_dir / "dataset_label_lists.json").write_text(
        json.dumps({name: list(labels) for name, labels in dataset_label_lists.items()}, indent=2, sort_keys=True)
    )

    metrics = {
        "task": "span_role_multidataset",
        "datasets": [spec.name for spec in dataset_specs],
        "model_name": model_name,
        "test_metrics": overall_test_metrics,
        "test_mean_dataset_f1": test_mean_dataset_f1,
        "per_dataset_test_metrics": per_dataset_test_metrics,
        "counts": {
            "train": len(encoded_train),
            "eval": len(encoded_eval),
            "test": len(encoded_test),
        },
        "dataset_label_lists": {name: list(labels) for name, labels in dataset_label_lists.items()},
        "epoch_summaries": epoch_summaries,
        "selection_metric": "mean_dataset_f1_macro",
        "best_epoch": int(best_state["epoch"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": "span_role_multidataset",
                "datasets": [spec.name for spec in dataset_specs],
                "dataset_inputs": {spec.name: str(spec.input_dir) for spec in dataset_specs},
                "model_name": model_name,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "seed": seed,
                "dataset_label_lists": {name: list(labels) for name, labels in dataset_label_lists.items()},
                "selection_metric": "mean_dataset_f1_macro",
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_run_analytics(
        output_dir,
        filename="train_analytics.json",
        payload={
            "task": "span_role_multidataset",
            "datasets": [spec.name for spec in dataset_specs],
            "model_name": model_name,
            "split_counts_by_dataset": split_counts_by_dataset,
            "split_analytics_by_dataset": split_analytics_by_dataset,
            "dataset_label_lists": {name: list(labels) for name, labels in dataset_label_lists.items()},
            "epoch_summaries": epoch_summaries,
            "best_epoch": int(best_state["epoch"]),
            "best_eval_mean_dataset_f1": best_eval_mean_dataset_f1,
            "test_metrics": overall_test_metrics,
            "test_mean_dataset_f1": test_mean_dataset_f1,
            "per_dataset_test_metrics": per_dataset_test_metrics,
            "prediction_row_count": len(test_prediction_rows),
        },
    )
    _emit_multi_span_progress(
        output_dir,
        phase="complete",
        message="Completed multi-dataset span-role training run",
        model_name=model_name,
        dataset_specs=dataset_specs,
        best_epoch=int(best_state["epoch"]),
        test_mean_dataset_f1=test_mean_dataset_f1,
        test_metrics=overall_test_metrics,
        per_dataset_test_metrics=per_dataset_test_metrics,
        model_dir=str(model_dir),
    )
    write_run_reports(
        output_dir=output_dir,
        repo_root=Path(__file__).resolve().parents[3],
        title="Multi-Dataset Span Role Training Run",
        payload=metrics,
        extra={
            "dataset_inputs": {spec.name: str(spec.input_dir) for spec in dataset_specs},
            "model_dir": str(model_dir),
            "problem_type": "multidataset_token_classification",
            "selection_metric": "mean_dataset_f1_macro",
        },
    )
    return metrics
