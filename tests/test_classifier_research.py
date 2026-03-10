from __future__ import annotations

import json
from pathlib import Path

from memex_research.classifier_research.artifacts import package_run_artifacts, write_run_inventory
from memex_research.classifier_research.datasets import (
    normalize_pe_span_record,
    normalize_scicite_source_label,
    normalize_scicite_source_row,
)
from memex_research.classifier_research.hf_models import SUPPORTED_PROBE_MODELS
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import (
    discover_named_splits,
    ensure_dev_split,
    load_jsonl_splits,
    write_split_jsonl,
)
from memex_research.classifier_research.tasks import (
    get_task_spec,
    validate_task_dataset,
)
from memex_research.classifier_research.transfer_eval import (
    _candidate_context_window,
    _write_source_transfer_summary,
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


def test_normalize_pe_span_record_rejects_unaligned_annotation_span() -> None:
    text = "Cats are great."
    annotation_text = "T1\tClaim 4 5\t "
    try:
        normalize_pe_span_record("doc-bad", text, annotation_text)
    except ValueError as exc:
        assert "did not align" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected misaligned annotation to raise.")


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


def test_ensure_dev_split_creates_deterministic_holdout() -> None:
    split_rows = {
        "train": [{"id": f"train-{index}"} for index in range(10)],
        "test": [{"id": "test-1"}],
    }
    split_a = ensure_dev_split(split_rows, dev_fraction=0.2, seed=17)
    split_b = ensure_dev_split(split_rows, dev_fraction=0.2, seed=17)
    assert len(split_a["dev"]) == 2
    assert len(split_a["train"]) == 8
    assert [row["id"] for row in split_a["dev"]] == [row["id"] for row in split_b["dev"]]


def test_transfer_manifest_points_to_existing_seed_posts() -> None:
    rows = load_transfer_manifest()
    assert len(rows) == 10
    for row in rows:
        assert Path(str(row["raw_path"])).exists()
        assert Path(str(row["text_path"])).exists()
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "analysis" / "memex_transfer_seed_10" / "manifest.json").read_text()
    )
    assert str(manifest["posts"][0]["raw_path"]).startswith("posts/")


def test_build_transfer_source_candidates_collects_bibliography_and_links() -> None:
    text = (
        'Quoted discussion of "The Book of Proof". '
        "1. John Smith, The Book of Proof, https://example.com/book-of-proof"
    )
    raw_post = {"external_links": ["https://example.com/book-of-proof", "https://foo.test/theorem-guides"]}
    candidates = build_transfer_source_candidates(raw_post=raw_post, text=text)
    names = {str(candidate["name"]) for candidate in candidates}
    assert "The Book of Proof" in names


def test_build_transfer_source_candidates_filters_unanchored_external_links() -> None:
    text = "This post discusses The Book of Proof."
    raw_post = {
        "external_links": [
            "https://example.com/book-of-proof",
            "https://foo.test/completely-unrelated-reading-list",
        ]
    }
    candidates = build_transfer_source_candidates(raw_post=raw_post, text=text)
    names = {str(candidate["name"]) for candidate in candidates}
    assert "book of proof" in {name.lower() for name in names}
    assert "completely unrelated reading list" not in {name.lower() for name in names}


def test_build_transfer_source_candidates_ignores_non_title_quoted_spans() -> None:
    text = (
        '"What is the difference between a technical understanding and a verbal understanding?" '
        'is not a title, but "The Book of Proof" is.'
    )
    candidates = build_transfer_source_candidates(raw_post={"external_links": []}, text=text)
    names = {str(candidate["name"]) for candidate in candidates}
    assert "The Book of Proof" in names
    assert (
        "What is the difference between a technical understanding and a verbal understanding?"
        not in names
    )


def test_candidate_context_window_returns_none_when_candidate_is_not_in_text() -> None:
    text = "This is a post about epistemology and politics."
    assert _candidate_context_window(text, "Completely Unrelated Book Title") is None


def test_public_probe_model_is_supported() -> None:
    assert "Qwen/Qwen2.5-7B-Instruct" in SUPPORTED_PROBE_MODELS


def test_write_run_reports_emits_diagnostics_and_markdown(tmp_path: Path) -> None:
    write_run_reports(
        output_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
        title="Test Run",
        payload={"test_metrics": {"accuracy": 0.5, "f1_macro": 0.4, "precision_macro": 0.3, "recall_macro": 0.2}},
        extra={"model": "dummy/model"},
    )
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "report.md").exists()


def test_write_source_transfer_summary_ignores_diagnostics_file(tmp_path: Path) -> None:
    (tmp_path / "abc123.json").write_text(
        json.dumps(
            {
                "post_id": "abc123",
                "generated_candidate_count": 2,
                "skipped_no_context_count": 1,
                "generated_origin_counts": {"bibliography": 1, "external_link": 1},
                "kept_origin_counts": {"bibliography": 1},
                "candidates": [{"predicted_label": "SOURCE"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "diagnostics.json").write_text(json.dumps({"generated_at_utc": "now"}), encoding="utf-8")
    _write_source_transfer_summary(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["post_count"] == 1
    assert summary["total_candidates"] == 1
    assert summary["generated_candidates"] == 2
    assert summary["skipped_no_context"] == 1
    assert summary["generated_origin_counts"]["external_link"] == 1


def test_write_run_inventory_summarizes_sizes_and_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "scicite_deberta"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"test_metrics": {"accuracy": 0.8, "f1_macro": 0.7}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"task": "source_materiality", "dataset": "scicite", "model_name": "dummy/model"}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "diagnostics.json").write_text(json.dumps({"python_version": "3.11"}), encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    (run_dir / "model").mkdir()
    (run_dir / "model" / "weights.bin").write_bytes(b"1234")

    output_dir = tmp_path / "inventory"
    payload = write_run_inventory(output_dir, run_root=run_root)

    assert payload["run_count"] == 1
    assert payload["runs"][0]["sizes"]["model_bytes"] == 4
    assert payload["runs"][0]["metrics"]["test_metrics"]["f1_macro"] == 0.7
    assert (output_dir / "inventory.json").exists()
    assert (output_dir / "inventory.md").exists()


def test_package_run_artifacts_excludes_trainer_by_default(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "scicite_deberta"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "trainer").mkdir()
    (run_dir / "transfer").mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (run_dir / "diagnostics.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    (run_dir / "model" / "weights.bin").write_bytes(b"model")
    (run_dir / "trainer" / "checkpoint.bin").write_bytes(b"trainer")
    (run_dir / "transfer" / "summary.json").write_text(json.dumps({"post_count": 1}), encoding="utf-8")

    output_path = tmp_path / "packaged" / "runs.tar.gz"
    manifest = package_run_artifacts(run_root=run_root, output_path=output_path)

    assert manifest["run_count"] == 1
    assert "trainer/" not in manifest["runs"][0]["included"]
    assert output_path.exists()
    assert (tmp_path / "packaged" / "runs.tar.gz.manifest.json").exists()
