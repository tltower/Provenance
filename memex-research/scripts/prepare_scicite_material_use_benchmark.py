from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.datasets import (
    balanced_sample_by_label,
    load_jsonl,
    normalize_scicite_source_row,
)
from memex_research.classifier_research.splits import write_split_jsonl
from memex_research.classifier_research.tasks import SourceMaterialityExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Legacy single-split SciCite JSONL input.")
    parser.add_argument("--output", type=Path, default=None, help="Legacy single-file output path.")
    parser.add_argument("--train-input", type=Path, default=None)
    parser.add_argument("--dev-input", type=Path, default=None)
    parser.add_argument("--test-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--per-label",
        type=int,
        default=0,
        help="Balanced sample size per label. Use 0 to keep the full split.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.output_dir is not None:
        if args.train_input is None or args.test_input is None:
            raise RuntimeError("--output-dir requires at least --train-input and --test-input.")
        split_map: dict[str, list[SourceMaterialityExample]] = {}
        for split, input_path in (("train", args.train_input), ("dev", args.dev_input), ("test", args.test_input)):
            if input_path is None:
                continue
            split_rows: list[SourceMaterialityExample] = [
                normalize_scicite_source_row(row) for row in load_jsonl(input_path)
            ]
            if args.per_label > 0:
                split_rows = balanced_sample_by_label(
                    split_rows, label_key="label", per_label=args.per_label, seed=args.seed
                )
            split_map[split] = split_rows
        write_split_jsonl(args.output_dir, split_map)
        return

    if args.input is None or args.output is None:
        raise RuntimeError("Expected either legacy --input/--output or multi-split --output-dir arguments.")

    rows: list[SourceMaterialityExample] = [normalize_scicite_source_row(row) for row in load_jsonl(args.input)]
    if args.per_label > 0:
        rows = balanced_sample_by_label(rows, label_key="label", per_label=args.per_label, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
