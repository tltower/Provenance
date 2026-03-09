from __future__ import annotations

import argparse
import json
from pathlib import Path

from memex_research.classifier_research.artifacts import write_run_inventory


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
        "--output-dir",
        type=Path,
        default=_repo_root() / "analysis" / "run_inventory",
    )
    parser.add_argument("--title", default="Run Inventory")
    args = parser.parse_args()

    payload = write_run_inventory(args.output_dir, run_root=args.run_root, title=args.title)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
