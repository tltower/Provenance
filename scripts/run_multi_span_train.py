from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.tasks import validate_task_dataset
from memex_research.classifier_research.train_multi_span import (
    SpanDatasetSpec,
    train_multi_span_classifier,
)


def _parse_dataset_input(value: str) -> SpanDatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected dataset inputs in the form `name=/path/to/dir`.")
    dataset_name, raw_path = value.split("=", 1)
    clean_name = dataset_name.strip().lower()
    clean_path = Path(raw_path.strip())
    if not clean_name:
        raise argparse.ArgumentTypeError("Dataset input name must not be empty.")
    validate_task_dataset("span_role", clean_name)
    return SpanDatasetSpec(name=clean_name, input_dir=clean_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-input",
        action="append",
        type=_parse_dataset_input,
        required=True,
        help="Dataset input in the form `name=/path/to/prepared_dir`.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    metrics = train_multi_span_classifier(
        dataset_specs=args.dataset_input,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
