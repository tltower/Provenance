from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from memex_research.classifier_research.tasks import (
    SOURCE_MATERIALITY_LABELS,
    SPAN_ROLE_LABELS,
    TASK_SOURCE_MATERIALITY,
    TASK_SPAN_ROLE,
    SourceMaterialityExample,
    SpanRoleExample,
)

TOKEN_RE = re.compile(r"\S+")
RowT = TypeVar("RowT", bound=Mapping[str, Any])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean = line.strip()
            if not clean:
                continue
            rows.append(json.loads(clean))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def balanced_sample_by_label(
    rows: list[RowT],
    *,
    label_key: str,
    per_label: int,
    seed: int,
) -> list[RowT]:
    grouped: dict[str, list[RowT]] = {}
    for row in rows:
        label = str(row.get(label_key) or "")
        grouped.setdefault(label, []).append(row)

    rng = random.Random(seed)
    sampled: list[RowT] = []
    for label in sorted(grouped):
        choices = list(grouped[label])
        rng.shuffle(choices)
        sampled.extend(choices[:per_label])
    return sampled


def normalize_scicite_source_label(label: str) -> str:
    clean = (label or "").strip().lower()
    if clean in {"method", "result"}:
        return "SOURCE"
    if clean == "background":
        return "NOT_SOURCE"
    raise ValueError(f"Unsupported SciCite label: {label!r}")


def normalize_scicite_source_row(row: dict[str, Any]) -> SourceMaterialityExample:
    label = normalize_scicite_source_label(str(row.get("label") or ""))
    if label not in SOURCE_MATERIALITY_LABELS:
        raise ValueError(f"Unexpected normalized SciCite label: {label!r}")
    return {
        "id": str(row.get("unique_id") or row.get("id") or ""),
        "context": str(row.get("string") or "").strip(),
        "label": label,
        "section_name": str(row.get("sectionName") or "").strip(),
        "source_type": str(row.get("source") or "").strip(),
        "dataset": "SciCite",
        "task": TASK_SOURCE_MATERIALITY,
    }


def normalize_pe_component_label(label: str) -> str | None:
    clean = (label or "").strip().lower().replace("-", "").replace("_", "")
    if clean in {"majorclaim", "claim"}:
        return "CLAIM"
    if clean == "premise":
        return "PREMISE"
    return None


def normalize_cdcp_component_label(label: str) -> str | None:
    clean = (label or "").strip().lower().replace("-", "").replace("_", "")
    if clean in {"value", "policy", "claim"}:
        return "CLAIM"
    if clean in {"fact", "testimony", "reference", "evidence"}:
        return "EVIDENCE"
    if clean in {"premise", "reason"}:
        return "PREMISE"
    return None


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def parse_brat_text_annotations(annotation_text: str) -> list[tuple[str, list[tuple[int, int]]]]:
    annotations: list[tuple[str, list[tuple[int, int]]]] = []
    for line in annotation_text.splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        meta = parts[1].strip()
        if not meta:
            continue
        meta_parts = meta.split(maxsplit=1)
        if len(meta_parts) != 2:
            continue
        raw_label, span_spec = meta_parts
        label = normalize_pe_component_label(raw_label)
        if label is None:
            continue
        spans: list[tuple[int, int]] = []
        for chunk in span_spec.split(";"):
            chunk_parts = chunk.strip().split()
            if len(chunk_parts) < 2:
                continue
            start = int(chunk_parts[0])
            end = int(chunk_parts[1])
            spans.append((start, end))
        if spans:
            annotations.append((label, spans))
    return annotations


def _spans_overlap(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    return token_start < span_end and span_start < token_end


def _validate_tokenization(text: str, token_offsets: Sequence[tuple[str, int, int]], *, doc_id: str) -> None:
    covered = [False] * len(text)
    for _token, start, end in token_offsets:
        for position in range(start, end):
            covered[position] = True
    uncovered_positions = [
        index for index, character in enumerate(text) if not character.isspace() and not covered[index]
    ]
    if uncovered_positions:
        first = uncovered_positions[0]
        snippet = text[max(0, first - 20) : first + 20]
        raise ValueError(
            f"Tokenization left uncovered non-whitespace text in document {doc_id!r} near offset {first}: {snippet!r}"
        )


def _validate_annotation_alignment(
    token_offsets: Sequence[tuple[str, int, int]],
    annotations: Sequence[tuple[str, Sequence[tuple[int, int]]]],
    *,
    doc_id: str,
) -> None:
    missing: list[tuple[str, int, int]] = []
    for label, spans in annotations:
        for start, end in spans:
            if not any(
                _spans_overlap(token_start, token_end, start, end)
                for _token, token_start, token_end in token_offsets
            ):
                missing.append((label, start, end))
    if missing:
        preview = ", ".join(f"{label}@{start}:{end}" for label, start, end in missing[:5])
        raise ValueError(
            f"Annotation spans did not align to any token in document {doc_id!r}: {preview}"
        )


def label_tokens_from_annotations(
    text: str,
    annotations: list[tuple[str, list[tuple[int, int]]]],
) -> tuple[list[str], list[str]]:
    token_offsets = tokenize_with_offsets(text)
    tokens = [token for token, _start, _end in token_offsets]
    labels = ["OTHER"] * len(tokens)
    for index, (_token, token_start, token_end) in enumerate(token_offsets):
        for label, spans in annotations:
            if any(_spans_overlap(token_start, token_end, span_start, span_end) for span_start, span_end in spans):
                labels[index] = label
                break
    return tokens, labels


def normalize_pe_span_record(doc_id: str, text: str, annotation_text: str) -> SpanRoleExample:
    annotations = parse_brat_text_annotations(annotation_text)
    token_offsets = tokenize_with_offsets(text)
    _validate_tokenization(text, token_offsets, doc_id=doc_id)
    _validate_annotation_alignment(token_offsets, annotations, doc_id=doc_id)
    tokens, labels = label_tokens_from_annotations(text, annotations)
    for label in labels:
        if label not in SPAN_ROLE_LABELS:
            raise ValueError(f"Unexpected span label {label!r} for document {doc_id}")
    return {
        "id": doc_id,
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "dataset": "PersuasiveEssays",
        "task": TASK_SPAN_ROLE,
    }


def normalize_cdcp_span_record(
    doc_id: str,
    text: str,
    proposition_starts: Sequence[int],
    proposition_ends: Sequence[int],
    proposition_labels: Sequence[str],
) -> SpanRoleExample:
    if not (
        len(proposition_starts) == len(proposition_ends) == len(proposition_labels)
    ):
        raise ValueError(
            f"CDCP proposition field lengths do not match for document {doc_id!r}: "
            f"{len(proposition_starts)} starts, {len(proposition_ends)} ends, "
            f"{len(proposition_labels)} labels"
        )

    annotations: list[tuple[str, list[tuple[int, int]]]] = []
    for start, end, raw_label in zip(
        proposition_starts, proposition_ends, proposition_labels, strict=True
    ):
        label = normalize_cdcp_component_label(raw_label)
        if label is None:
            continue
        annotations.append((label, [(int(start), int(end))]))

    token_offsets = tokenize_with_offsets(text)
    _validate_tokenization(text, token_offsets, doc_id=doc_id)
    _validate_annotation_alignment(token_offsets, annotations, doc_id=doc_id)
    tokens, labels = label_tokens_from_annotations(text, annotations)
    for label in labels:
        if label not in SPAN_ROLE_LABELS:
            raise ValueError(f"Unexpected CDCP span label {label!r} for document {doc_id}")
    return {
        "id": doc_id,
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "dataset": "CDCP",
        "task": TASK_SPAN_ROLE,
    }
