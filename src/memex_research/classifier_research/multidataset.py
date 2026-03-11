from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from memex_research.classifier_research.datasets import write_jsonl
from memex_research.classifier_research.splits import load_jsonl_splits
from memex_research.classifier_research.status import emit_run_status
from memex_research.classifier_research.tasks import SPAN_ROLE_LABELS


def infer_span_label_list(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    used_labels = {
        str(label)
        for row in records
        for label in row.get("labels", [])
        if str(label) in SPAN_ROLE_LABELS
    }
    return tuple(label for label in SPAN_ROLE_LABELS if label in used_labels)


def merge_prepared_benchmark_dirs(
    *,
    dataset_inputs: Mapping[str, Path],
    output_dir: Path,
    shuffle: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    if not dataset_inputs:
        raise ValueError("Expected at least one dataset input.")

    emit_run_status(
        output_dir,
        filename="merge_status.json",
        prefix="merge",
        phase="starting",
        message="Starting prepared benchmark merge",
        datasets=sorted(dataset_inputs),
        shuffle=shuffle,
        seed=seed,
    )

    merged_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    rng = random.Random(seed)

    for dataset_name, input_dir in dataset_inputs.items():
        if not input_dir.exists():
            raise FileNotFoundError(f"Prepared benchmark directory does not exist: {input_dir}")
        split_rows = load_jsonl_splits(input_dir)
        source_counts[dataset_name] = {}
        emit_run_status(
            output_dir,
            filename="merge_status.json",
            prefix="merge",
            phase="loading_dataset",
            message=f"Loading prepared splits for {dataset_name}",
            dataset_name=dataset_name,
            input_dir=str(input_dir),
            split_counts={split_name: len(rows) for split_name, rows in split_rows.items()},
        )
        for split_name, rows in split_rows.items():
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                normalized = dict(row)
                normalized["source_dataset"] = dataset_name
                normalized_rows.append(normalized)
            merged_rows.setdefault(split_name, []).extend(normalized_rows)
            source_counts[dataset_name][split_name] = len(normalized_rows)

    if shuffle:
        for rows in merged_rows.values():
            rng.shuffle(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_split in ("train", "dev", "test"):
        stale_path = output_dir / f"{stale_split}.jsonl"
        if stale_path.exists():
            stale_path.unlink()
    split_counts: dict[str, int] = {}
    for split_name, rows in merged_rows.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
        split_counts[split_name] = len(rows)

    summary = {
        "dataset_inputs": {name: str(path) for name, path in dataset_inputs.items()},
        "shuffle": shuffle,
        "seed": seed,
        "split_counts": split_counts,
        "source_counts": source_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    emit_run_status(
        output_dir,
        filename="merge_status.json",
        prefix="merge",
        phase="complete",
        message="Completed prepared benchmark merge",
        split_counts=split_counts,
        source_counts=source_counts,
    )
    return summary
