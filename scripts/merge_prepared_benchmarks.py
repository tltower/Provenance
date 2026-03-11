from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.multidataset import merge_prepared_benchmark_dirs


def _parse_dataset_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected dataset inputs in the form `name=/path/to/dir`.")
    dataset_name, raw_path = value.split("=", 1)
    clean_name = dataset_name.strip().lower()
    clean_path = Path(raw_path.strip())
    if not clean_name:
        raise argparse.ArgumentTypeError("Dataset input name must not be empty.")
    return clean_name, clean_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-input",
        action="append",
        type=_parse_dataset_input,
        required=True,
        help="Dataset input in the form `name=/path/to/prepared_dir`.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataset_inputs = dict(args.dataset_input)
    summary = merge_prepared_benchmark_dirs(
        dataset_inputs=dataset_inputs,
        output_dir=args.output_dir,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
