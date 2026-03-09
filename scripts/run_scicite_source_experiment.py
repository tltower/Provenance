from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from memex_research.classifier_research.datasets import (
    SourceMaterialityExample,
    balanced_sample_by_label,
    load_jsonl,
    normalize_scicite_source_row,
)
from memex_research.classifier_research.splits import write_split_jsonl
from memex_research.classifier_research.train_source import train_source_classifier
from memex_research.classifier_research.transfer_eval import (
    DEFAULT_TRANSFER_MANIFEST,
    run_source_transfer_hf_model,
)

SCICITE_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_scicite(download_root: Path) -> Path:
    download_root.mkdir(parents=True, exist_ok=True)
    tar_path = download_root / "scicite.tar.gz"
    extract_root = download_root / "scicite"
    if not extract_root.exists():
        if not tar_path.exists():
            urllib.request.urlretrieve(SCICITE_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(download_root, filter="data")
    return extract_root


def _prepare_scicite_splits(
    *,
    raw_root: Path,
    output_dir: Path,
    per_label: int,
    seed: int,
) -> None:
    split_map: dict[str, list[SourceMaterialityExample]] = {}
    for split in ("train", "dev", "test"):
        input_path = raw_root / f"{split}.jsonl"
        split_rows = [normalize_scicite_source_row(row) for row in load_jsonl(input_path)]
        if per_label > 0:
            split_rows = balanced_sample_by_label(
                split_rows,
                label_key="label",
                per_label=per_label,
                seed=seed,
            )
        split_map[split] = split_rows
    write_split_jsonl(output_dir, split_map)


def _run_quality_checks(repo_root: Path) -> None:
    commands = [
        ["python", "-m", "pytest", "-q"],
        ["python", "-m", "mypy", "src"],
        ["ruff", "check", "src", "tests", "scripts"],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        choices=[
            "microsoft/deberta-v3-base",
            "allenai/scibert_scivocab_uncased",
        ],
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <repo>/analysis/memex_runs",
    )
    parser.add_argument("--per-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-quality-checks", action="store_true")
    parser.add_argument("--transfer-manifest", type=Path, default=DEFAULT_TRANSFER_MANIFEST)
    args = parser.parse_args()

    if args.transfer_only and not (args.run_transfer and not args.skip_transfer):
        raise RuntimeError("--transfer-only requires --run-transfer.")

    repo_root = _repo_root()
    output_root = args.output_root or (repo_root / "analysis" / "memex_runs")
    run_tag = "scicite_deberta" if "deberta" in args.model_name else "scicite_scibert"
    output_dir = output_root / run_tag

    data_root = args.data_root
    download_root = data_root / "scicite_download"
    raw_root = data_root / "scicite_raw"
    benchmark_root = data_root / "scicite_material_benchmark"

    if not args.skip_quality_checks and not args.transfer_only:
        _run_quality_checks(repo_root)

    if not args.skip_download and not args.transfer_only:
        extracted = _download_scicite(download_root)
        raw_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            source_path = extracted / f"{split}.jsonl"
            target_path = raw_root / f"{split}.jsonl"
            if not target_path.exists():
                target_path.write_bytes(source_path.read_bytes())

    if args.transfer_only:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"Missing trained run at {output_dir}. Expected {metrics_path}.")
        metrics = json.loads(metrics_path.read_text())
    else:
        _prepare_scicite_splits(
            raw_root=raw_root,
            output_dir=benchmark_root,
            per_label=args.per_label,
            seed=args.seed,
        )

        metrics = train_source_classifier(
            input_dir=benchmark_root,
            output_dir=output_dir,
            model_name=args.model_name,
            seed=args.seed,
        )

    if args.run_transfer and not args.skip_transfer:
        run_source_transfer_hf_model(
            model_dir=output_dir / "model",
            output_dir=output_dir / "transfer",
            manifest_path=args.transfer_manifest,
        )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
