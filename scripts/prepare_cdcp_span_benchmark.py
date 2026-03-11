from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from memex_research.classifier_research.datasets import (
    derive_cdcp_component_labels,
    extract_cdcp_proposition_offsets,
    normalize_cdcp_span_record,
)
from memex_research.classifier_research.splits import ensure_dev_split, write_split_jsonl
from memex_research.classifier_research.tasks import SpanRoleExample


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Dataset dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a remote ML environment."
        ) from exc
    return load_dataset


def _normalize_split_name(name: str) -> str:
    clean = (name or "").strip().lower()
    if clean in {"validation", "valid", "dev"}:
        return "dev"
    if clean in {"train", "test"}:
        return clean
    raise ValueError(f"Unsupported CDCP split name: {name!r}")


def _extract_text(row: dict[str, Any]) -> str:
    for key in ("text", "raw_text", "document", "doc_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"Unable to find document text field in CDCP row with keys: {sorted(row)}")


def _extract_nested_class_label_names(
    split: Any,
    *,
    field_name: str,
) -> list[str] | None:
    try:
        nested_feature = split.features[field_name].feature
        label_feature = nested_feature["label"]
        names = getattr(label_feature, "names", None)
    except (AttributeError, KeyError, TypeError):
        return None
    if names is None:
        return None
    return [str(name) for name in names]


def _extract_doc_id(row: dict[str, Any], *, split_name: str, row_index: int) -> str:
    for key in ("id", "doc_id", "docId", "document_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"{split_name}-{row_index:06d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="DFKI-SLT/cdcp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    load_dataset = _require_datasets()
    dataset = load_dataset(args.dataset_name, trust_remote_code=True)
    first_split = dataset[next(iter(dataset))]
    proposition_label_names = _extract_nested_class_label_names(first_split, field_name="propositions")
    relation_label_names = _extract_nested_class_label_names(first_split, field_name="relations")

    split_rows: dict[str, list[SpanRoleExample]] = {}
    for split_name, split in dataset.items():
        normalized_split = _normalize_split_name(str(split_name))
        rows: list[SpanRoleExample] = []
        for row_index, row in enumerate(split):
            text = _extract_text(row)
            starts, ends = extract_cdcp_proposition_offsets(row)
            labels = derive_cdcp_component_labels(
                row,
                proposition_label_names=proposition_label_names,
                relation_label_names=relation_label_names,
            )
            rows.append(
                normalize_cdcp_span_record(
                    _extract_doc_id(row, split_name=normalized_split, row_index=row_index),
                    text,
                    starts,
                    ends,
                    labels,
                )
            )
        split_rows[normalized_split] = rows

    split_rows = ensure_dev_split(split_rows, dev_fraction=args.dev_fraction, seed=args.seed)
    write_split_jsonl(args.output_dir, split_rows)


if __name__ == "__main__":
    main()
