from __future__ import annotations

import json
from pathlib import Path

from memex_research.classifier_research.datasets import (
    normalize_pe_span_record,
    normalize_scicite_source_label,
    normalize_scicite_source_row,
)
from memex_research.classifier_research.splits import (
    discover_named_splits,
    load_jsonl_splits,
    write_split_jsonl,
)
from memex_research.classifier_research.tasks import (
    get_task_spec,
    validate_task_dataset,
)
from memex_research.classifier_research.transfer_eval import (
    build_transfer_source_candidates,
    load_transfer_manifest,
)


def test_get_task_spec_returns_expected_labels() -> None:
    spec = get_task_spec("span_role")
    assert spec.labels == ("CLAIM", "PREMISE", "EVIDENCE", "OTHER")
    assert spec.problem_type == "token_classification"


def test_validate_task_dataset_rejects_invalid_pair() -> None:
    try:
        validate_task_dataset("span_role", "scicite")
    except ValueError as exc:
        assert "not valid" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid task/dataset pair to raise.")


def test_normalize_scicite_source_row_maps_to_binary_labels() -> None:
    row = {
        "unique_id": "row-1",
        "string": "Prior work used the same method.",
        "sectionName": "Methods",
        "source": "explicit",
        "label": "method",
    }
    assert normalize_scicite_source_label("background") == "NOT_SOURCE"
    normalized = normalize_scicite_source_row(row)
    assert normalized["label"] == "SOURCE"
    assert normalized["task"] == "source_materiality"


def test_normalize_pe_span_record_labels_claim_and_premise_tokens() -> None:
    text = "Cats are great. Because they purr loudly."
    annotation_text = (
        "T1\tClaim 0 15\tCats are great.\n"
        "T2\tPremise 16 40\tBecause they purr loudly."
    )
    record = normalize_pe_span_record("doc-1", text, annotation_text)
    assert record["task"] == "span_role"
    assert "CLAIM" in record["labels"]
    assert "PREMISE" in record["labels"]
    assert len(record["tokens"]) == len(record["labels"])


def test_discover_named_splits_uses_standard_subdirectories(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "a.txt").write_text("train", encoding="utf-8")
    (test_dir / "b.txt").write_text("test", encoding="utf-8")

    split_map = discover_named_splits(root=tmp_path, suffix=".txt")
    assert list(split_map) == ["train", "test"]
    assert split_map["train"][0].name == "a.txt"


def test_write_and_load_jsonl_splits_roundtrip(tmp_path: Path) -> None:
    split_rows = {
        "train": [{"id": "train-1", "task": "source_materiality"}],
        "test": [{"id": "test-1", "task": "source_materiality"}],
    }
    write_split_jsonl(tmp_path, split_rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"]["train"] == 1
    loaded = load_jsonl_splits(tmp_path)
    assert loaded["test"][0]["id"] == "test-1"


def test_transfer_manifest_points_to_existing_seed_posts() -> None:
    rows = load_transfer_manifest()
    assert len(rows) == 10
    for row in rows:
        assert Path(str(row["raw_path"])).exists()
        assert Path(str(row["text_path"])).exists()


def test_build_transfer_source_candidates_collects_bibliography_and_links() -> None:
    text = (
        'Quoted discussion of "The Book of Proof". '
        "1. John Smith, The Book of Proof, https://example.com/book-of-proof"
    )
    raw_post = {"external_links": ["https://example.com/book-of-proof", "https://foo.test/bar-baz"]}
    candidates = build_transfer_source_candidates(raw_post=raw_post, text=text)
    names = {str(candidate["name"]) for candidate in candidates}
    assert "The Book of Proof" in names
    assert "bar baz" in names
