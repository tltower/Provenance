from __future__ import annotations

import argparse
from pathlib import Path

from memex_research.classifier_research.transfer_eval import (
    DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH,
    run_source_candidate_transfer_hf_model,
    run_source_candidate_transfer_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--probe-model-name")
    args = parser.parse_args()

    has_model = args.model_dir is not None
    has_probe = args.probe_dir is not None
    if has_model == has_probe:
        raise RuntimeError("Provide exactly one of --model-dir or --probe-dir.")
    if args.probe_dir is not None and not args.probe_model_name:
        raise RuntimeError("--probe-model-name is required when using --probe-dir.")

    if args.model_dir is not None:
        run_source_candidate_transfer_hf_model(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            candidates_path=args.candidates_path,
            max_length=args.max_length,
        )
        return

    run_source_candidate_transfer_probe(
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        model_name=str(args.probe_model_name),
        candidates_path=args.candidates_path,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
