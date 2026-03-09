from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.artifacts import package_run_artifacts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=_repo_root() / "analysis" / "memex_runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "analysis" / "packaged_runs" / "memex_runs.tar.gz",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="Repeat to package only selected run directories.",
    )
    parser.add_argument(
        "--exclude-models",
        action="store_true",
        help="Skip the model/ directory for each run.",
    )
    parser.add_argument(
        "--exclude-transfer",
        action="store_true",
        help="Skip transfer/ artifacts.",
    )
    parser.add_argument(
        "--exclude-predictions",
        action="store_true",
        help="Skip predictions.jsonl artifacts.",
    )
    parser.add_argument(
        "--exclude-reports",
        action="store_true",
        help="Skip diagnostics.json, report.md, and metrics/config files.",
    )
    parser.add_argument(
        "--include-trainer",
        action="store_true",
        help="Include trainer/ checkpoints. This can make archives very large.",
    )
    args = parser.parse_args()

    manifest = package_run_artifacts(
        run_root=args.run_root,
        output_path=args.output,
        run_names=args.run_name,
        include_models=not args.exclude_models,
        include_transfer=not args.exclude_transfer,
        include_predictions=not args.exclude_predictions,
        include_reports=not args.exclude_reports,
        include_trainer=args.include_trainer,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
