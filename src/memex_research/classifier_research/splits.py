from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from memex_research.classifier_research.datasets import load_jsonl, write_jsonl

STANDARD_SPLIT_DIRS = {
    "train": ("train",),
    "dev": ("dev", "valid", "validation"),
    "test": ("test",),
}


def _normalize_split_name(name: str) -> str:
    clean = (name or "").strip().lower()
    if clean in {"dev", "valid", "validation"}:
        return "dev"
    if clean in {"train", "test"}:
        return clean
    raise ValueError(f"Unsupported split name: {name!r}")


def _read_manifest_entries(path: Path) -> list[str]:
    entries: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean = line.strip()
            if clean:
                entries.append(clean)
    return entries


def _resolve_manifest_path(root: Path, entry: str, suffix: str) -> Path:
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate
    if candidate.suffix:
        return root / candidate
    return root / f"{entry}{suffix}"


def discover_named_splits(
    *,
    root: Path,
    suffix: str,
    train_manifest: Path | None = None,
    dev_manifest: Path | None = None,
    test_manifest: Path | None = None,
) -> dict[str, list[Path]]:
    manifests = {
        "train": train_manifest,
        "dev": dev_manifest,
        "test": test_manifest,
    }
    if any(path is not None for path in manifests.values()):
        split_map: dict[str, list[Path]] = {}
        for split, manifest_path in manifests.items():
            if manifest_path is None:
                continue
            entries = _read_manifest_entries(manifest_path)
            split_map[split] = [_resolve_manifest_path(root, entry, suffix) for entry in entries]
        if "train" not in split_map or "test" not in split_map:
            raise RuntimeError("Explicit split manifests must include at least train and test.")
        return split_map

    detected: dict[str, list[Path]] = {}
    for split, aliases in STANDARD_SPLIT_DIRS.items():
        for alias in aliases:
            directory = root / alias
            if directory.is_dir():
                detected[split] = sorted(directory.glob(f"*{suffix}"))
                break
    if "train" not in detected or "test" not in detected:
        raise RuntimeError(
            "Could not infer benchmark-native splits. Provide train/test manifests or standard split directories."
        )
    return detected


def write_split_jsonl(output_dir: Path, split_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, rows in split_rows.items():
        normalized_split = _normalize_split_name(split)
        write_jsonl(output_dir / f"{normalized_split}.jsonl", rows)
        counts[normalized_split] = len(rows)
    (output_dir / "summary.json").write_text(json.dumps({"counts": counts}, indent=2, sort_keys=True))
    return counts


def load_jsonl_splits(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "dev", "test"):
        path = input_dir / f"{split_name}.jsonl"
        if path.exists():
            splits[split_name] = load_jsonl(path)
    if "train" not in splits or "test" not in splits:
        raise RuntimeError(f"Expected at least train.jsonl and test.jsonl in {input_dir}")
    return splits
