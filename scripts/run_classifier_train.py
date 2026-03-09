from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.tasks import (
    TASK_SOURCE_MATERIALITY,
    TASK_SPAN_ROLE,
    validate_task_dataset,
)
from memex_research.classifier_research.train_source import (
    train_source_classifier,
)
from memex_research.classifier_research.train_span import train_span_classifier
from memex_research.classifier_research.transfer_eval import (
    DEFAULT_TRANSFER_MANIFEST,
    run_source_transfer_hf_model,
    run_span_transfer_hf_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[TASK_SPAN_ROLE, TASK_SOURCE_MATERIALITY], required=True)
    parser.add_argument("--dataset", choices=["pe", "scicite"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-manifest", type=Path, default=DEFAULT_TRANSFER_MANIFEST)
    args = parser.parse_args()

    validate_task_dataset(args.task, args.dataset)

    if args.task == TASK_SPAN_ROLE:
        metrics = train_span_classifier(
            input_dir=args.input_dir,
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
        if args.run_transfer and not args.skip_transfer:
            run_span_transfer_hf_model(
                model_dir=args.output_dir / "model",
                output_dir=args.output_dir / "transfer",
                manifest_path=args.transfer_manifest,
                max_length=args.max_length,
            )
    else:
        metrics = train_source_classifier(
            input_dir=args.input_dir,
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
        if args.run_transfer and not args.skip_transfer:
            run_source_transfer_hf_model(
                model_dir=args.output_dir / "model",
                output_dir=args.output_dir / "transfer",
                manifest_path=args.transfer_manifest,
                max_length=args.max_length,
            )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
