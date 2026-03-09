from __future__ import annotations

import argparse
from pathlib import Path

from memex_research.classifier_research.datasets import normalize_pe_span_record
from memex_research.classifier_research.splits import (
    discover_named_splits,
    ensure_dev_split,
    write_split_jsonl,
)
from memex_research.classifier_research.tasks import SpanRoleExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing .txt/.ann files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write split JSONL files.")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--dev-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    split_files = discover_named_splits(
        root=args.input_root,
        suffix=".txt",
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        test_manifest=args.test_manifest,
    )

    split_rows: dict[str, list[SpanRoleExample]] = {}
    for split, text_paths in split_files.items():
        rows: list[SpanRoleExample] = []
        for text_path in text_paths:
            annotation_path = text_path.with_suffix(".ann")
            if not annotation_path.exists():
                raise RuntimeError(f"Missing annotation file for {text_path}")
            text = text_path.read_text(encoding="utf-8")
            annotation_text = annotation_path.read_text(encoding="utf-8")
            rows.append(normalize_pe_span_record(text_path.stem, text, annotation_text))
        split_rows[split] = rows

    split_rows = ensure_dev_split(split_rows, dev_fraction=args.dev_fraction, seed=args.seed)
    write_split_jsonl(args.output_dir, split_rows)


if __name__ == "__main__":
    main()
