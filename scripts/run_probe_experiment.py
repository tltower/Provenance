from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.probes import run_probe_experiment
from memex_research.classifier_research.tasks import (
    TASK_SOURCE_MATERIALITY,
    TASK_SPAN_ROLE,
    validate_task_dataset,
)
from memex_research.classifier_research.transfer_eval import (
    DEFAULT_TRANSFER_MANIFEST,
    run_source_transfer_probe,
    run_span_transfer_probe,
)


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(chunk.strip()) for chunk in value.split(",") if chunk.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[TASK_SPAN_ROLE, TASK_SOURCE_MATERIALITY], required=True)
    parser.add_argument("--dataset", choices=["pe", "cdcp", "scicite"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices to probe.")
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-manifest", type=Path, default=DEFAULT_TRANSFER_MANIFEST)
    args = parser.parse_args()

    validate_task_dataset(args.task, args.dataset)
    summary = run_probe_experiment(
        task=args.task,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        layers=_parse_layers(args.layers),
    )

    if args.run_transfer and not args.skip_transfer:
        if args.task == TASK_SPAN_ROLE:
            run_span_transfer_probe(
                probe_dir=args.output_dir,
                output_dir=args.output_dir / "transfer",
                model_name=args.model_name,
                manifest_path=args.transfer_manifest,
                max_length=args.max_length,
            )
        else:
            run_source_transfer_probe(
                probe_dir=args.output_dir,
                output_dir=args.output_dir / "transfer",
                model_name=args.model_name,
                manifest_path=args.transfer_manifest,
                max_length=args.max_length,
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
