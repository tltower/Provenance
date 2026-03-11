from __future__ import annotations

import json
import tarfile
from pathlib import Path

from memex_research.classifier_research import train_span as train_span_module
from memex_research.classifier_research.artifacts import package_run_artifacts, write_run_inventory
from memex_research.classifier_research.datasets import (
    derive_cdcp_component_labels,
    extract_cdcp_proposition_offsets,
    normalize_cdcp_component_label,
    normalize_cdcp_span_record,
    normalize_pe_span_record,
    normalize_scicite_source_label,
    normalize_scicite_source_row,
)
from memex_research.classifier_research.hf_models import (
    SUPPORTED_PROBE_MODELS,
    SUPPORTED_SAE_MODELS,
    get_sae_release_spec,
)
from memex_research.classifier_research.multidataset import (
    infer_span_label_list,
    merge_prepared_benchmark_dirs,
)
from memex_research.classifier_research.probes import (
    _create_logistic_regression,
    _emit_probe_progress,
)
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import (
    discover_named_splits,
    ensure_dev_split,
    load_jsonl_splits,
    write_split_jsonl,
)
from memex_research.classifier_research.status import (
    EventCompatibleCallback,
    HFTrainerHeartbeatCallback,
    emit_run_status,
    write_run_analytics,
)
from memex_research.classifier_research.tasks import (
    get_task_spec,
    validate_task_dataset,
)
from memex_research.classifier_research.train_multi_span import (
    SpanDatasetSpec,
    _target_training_steps,
    _validate_dataset_specs,
)
from memex_research.classifier_research.train_multi_span import (
    _compute_metrics_from_prediction_rows as compute_multi_span_metrics,
)
from memex_research.classifier_research.train_span import (
    _compute_metrics_from_prediction_rows as compute_span_metrics,
)
from memex_research.classifier_research.transfer_eval import (
    _candidate_context_window,
    _write_source_candidate_transfer_summary,
    _write_source_transfer_summary,
    build_transfer_source_candidates,
    load_source_candidate_transfer_rows,
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


def test_validate_task_dataset_accepts_cdcp_for_span_role() -> None:
    validate_task_dataset("span_role", "cdcp")


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


def test_normalize_cdcp_component_label_maps_evidence_like_types() -> None:
    assert normalize_cdcp_component_label("value") == "CLAIM"
    assert normalize_cdcp_component_label("reference") == "EVIDENCE"
    assert normalize_cdcp_component_label("reason") == "PREMISE"
    assert normalize_cdcp_component_label("unknown") is None


def test_qwen_sae_release_uses_hub_id_format_with_underscores() -> None:
    release = get_sae_release_spec("Qwen/Qwen2.5-7B-Instruct")
    assert release.sae_id_candidates_for_layer(3) == (
        "resid_post_layer_3_trainer_1",
        "resid_post_layer_3_trainer_0",
        "resid_post_layer_3_trainer_2",
        "resid_post_layer_3_trainer_3",
    )


def test_normalize_cdcp_span_record_labels_claim_evidence_and_premise_tokens() -> None:
    text = "Cats are mammals. Experts observed purring. Therefore cats make good pets."
    record = normalize_cdcp_span_record(
        "cdcp-1",
        text,
        proposition_starts=[0, 18, 44],
        proposition_ends=[17, 43, len(text)],
        proposition_labels=["value", "reference", "reason"],
    )
    assert record["task"] == "span_role"
    assert "CLAIM" in record["labels"]
    assert "EVIDENCE" in record["labels"]
    assert "PREMISE" in record["labels"]
    assert len(record["tokens"]) == len(record["labels"])


def test_extract_cdcp_proposition_offsets_supports_hf_nested_schema() -> None:
    row = {
        "propositions": {
            "start": [0, 18, 44],
            "end": [17, 43, 73],
            "label": [4, 2, 1],
            "url": ["", "", ""],
        }
    }
    assert extract_cdcp_proposition_offsets(row) == ([0, 18, 44], [17, 43, 73])


def test_derive_cdcp_component_labels_uses_relation_roles_from_hf_schema() -> None:
    text = "Cats are mammals. Experts observed purring. Therefore cats make good pets."
    row = {
        "id": "cdcp-hf-1",
        "text": text,
        "propositions": {
            "start": [0, 18, 44],
            "end": [17, 43, len(text)],
            "label": [4, 2, 1],
            "url": ["", "", ""],
        },
        "relations": {
            "head": [2, 2],
            "tail": [0, 1],
            "label": [0, 1],
        },
    }

    labels = derive_cdcp_component_labels(
        row,
        proposition_label_names=["fact", "policy", "reference", "testimony", "value"],
        relation_label_names=["evidence", "reason"],
    )

    assert labels == ["evidence", "reason", "policy"]

    record = normalize_cdcp_span_record(
        "cdcp-hf-1",
        text,
        proposition_starts=row["propositions"]["start"],
        proposition_ends=row["propositions"]["end"],
        proposition_labels=labels,
    )
    assert "CLAIM" in record["labels"]
    assert "EVIDENCE" in record["labels"]
    assert "PREMISE" in record["labels"]


def test_infer_span_label_list_uses_only_labels_present_in_records() -> None:
    pe_record = {
        "labels": ["CLAIM", "PREMISE", "OTHER"],
    }
    cdcp_record = {
        "labels": ["CLAIM", "EVIDENCE", "PREMISE", "OTHER"],
    }
    assert infer_span_label_list([pe_record]) == ("CLAIM", "PREMISE", "OTHER")
    assert infer_span_label_list([cdcp_record]) == ("CLAIM", "PREMISE", "EVIDENCE", "OTHER")


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


def test_merge_prepared_benchmark_dirs_combines_splits_and_tags_sources(tmp_path: Path) -> None:
    pe_dir = tmp_path / "pe"
    cdcp_dir = tmp_path / "cdcp"
    write_split_jsonl(
        pe_dir,
        {
            "train": [{"id": "pe-1", "labels": ["CLAIM"], "task": "span_role"}],
            "test": [{"id": "pe-2", "labels": ["OTHER"], "task": "span_role"}],
        },
    )
    write_split_jsonl(
        cdcp_dir,
        {
            "train": [{"id": "cdcp-1", "labels": ["EVIDENCE"], "task": "span_role"}],
            "test": [{"id": "cdcp-2", "labels": ["CLAIM"], "task": "span_role"}],
        },
    )
    output_dir = tmp_path / "combined"
    summary = merge_prepared_benchmark_dirs(
        dataset_inputs={"pe": pe_dir, "cdcp": cdcp_dir},
        output_dir=output_dir,
        shuffle=False,
    )
    merged = load_jsonl_splits(output_dir)
    assert summary["split_counts"]["train"] == 2
    assert summary["source_counts"]["pe"]["train"] == 1
    assert summary["source_counts"]["cdcp"]["test"] == 1
    source_datasets = {str(row["source_dataset"]) for row in merged["train"] + merged["test"]}
    assert source_datasets == {"pe", "cdcp"}
    merge_status = json.loads((output_dir / "merge_status.json").read_text())
    assert merge_status["phase"] == "complete"
    assert merge_status["split_counts"]["train"] == 2
    merge_events = (output_dir / "merge_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(merge_events) >= 2


def test_merge_prepared_benchmark_dirs_clears_stale_split_files(tmp_path: Path) -> None:
    pe_dir = tmp_path / "pe"
    write_split_jsonl(
        pe_dir,
        {
            "train": [{"id": "pe-1", "labels": ["CLAIM"], "task": "span_role"}],
            "test": [{"id": "pe-2", "labels": ["OTHER"], "task": "span_role"}],
        },
    )
    output_dir = tmp_path / "combined"
    output_dir.mkdir()
    (output_dir / "dev.jsonl").write_text('{"id":"stale"}\n', encoding="utf-8")
    merge_prepared_benchmark_dirs(
        dataset_inputs={"pe": pe_dir},
        output_dir=output_dir,
        shuffle=False,
    )
    assert not (output_dir / "dev.jsonl").exists()


def test_merge_prepared_benchmark_dirs_rejects_missing_input(tmp_path: Path) -> None:
    try:
        merge_prepared_benchmark_dirs(
            dataset_inputs={"pe": tmp_path / "missing"},
            output_dir=tmp_path / "combined",
        )
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing prepared benchmark directory to raise.")


def test_validate_dataset_specs_rejects_duplicates_and_missing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "pe"
    existing.mkdir()
    try:
        _validate_dataset_specs(
            [
                SpanDatasetSpec(name="pe", input_dir=existing),
                SpanDatasetSpec(name="pe", input_dir=existing),
            ]
        )
    except ValueError as exc:
        assert "Duplicate dataset name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected duplicate dataset names to raise.")

    try:
        _validate_dataset_specs(
            [
                SpanDatasetSpec(name="pe", input_dir=existing),
                SpanDatasetSpec(name="cdcp", input_dir=tmp_path / "missing"),
            ]
        )
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing dataset input path to raise.")


def test_target_training_steps_uses_ceiling_and_rejects_invalid_values() -> None:
    assert _target_training_steps(loader_length=4, num_train_epochs=0.25) == 1
    assert _target_training_steps(loader_length=4, num_train_epochs=1.25) == 5

    try:
        _target_training_steps(loader_length=0, num_train_epochs=1.0)
    except ValueError as exc:
        assert "non-empty training loader" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected empty loader length to raise.")

    try:
        _target_training_steps(loader_length=4, num_train_epochs=0.0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected non-positive epoch count to raise.")


def test_span_prediction_metrics_report_perfect_and_skip_unknown_labels() -> None:
    prediction_rows = [
        {
            "gold_labels": ["CLAIM", "PREMISE", "OTHER"],
            "predicted_labels": ["CLAIM", "PREMISE", "OTHER"],
        },
        {
            "gold_labels": ["CLAIM", "UNKNOWN"],
            "predicted_labels": ["CLAIM", "OTHER"],
        },
    ]
    metrics = compute_span_metrics(prediction_rows, label_names=("CLAIM", "PREMISE", "OTHER"))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_span_prediction_metrics_count_predictions_outside_metric_label_space_as_errors(
    monkeypatch,
) -> None:
    def fake_accuracy_score(gold: list[int], pred: list[int]) -> float:
        assert gold == [0]
        assert pred == [2]
        return 0.0

    def fake_precision_recall_fscore_support(
        gold: list[int],
        pred: list[int],
        *,
        labels: list[int],
        average: str,
        zero_division: int,
    ) -> tuple[float, float, float, None]:
        assert gold == [0]
        assert pred == [2]
        assert labels == [0, 1, 3]
        assert average == "macro"
        assert zero_division == 0
        return 0.0, 0.0, 0.0, None

    monkeypatch.setattr(
        train_span_module,
        "_require_training_stack",
        lambda: (
            None,
            None,
            (fake_accuracy_score, fake_precision_recall_fscore_support),
            None,
        ),
    )

    metrics = train_span_module._compute_metrics_from_prediction_rows(
        [
            {
                "gold_labels": ["CLAIM"],
                "predicted_labels": ["EVIDENCE"],
            }
        ],
        label_names=("CLAIM", "PREMISE", "OTHER"),
        prediction_label_space=("CLAIM", "PREMISE", "EVIDENCE", "OTHER"),
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["f1_macro"] == 0.0


def test_multi_span_prediction_metrics_respect_dataset_specific_label_lists() -> None:
    prediction_rows = [
        {
            "dataset_name": "pe",
            "gold_labels": ["CLAIM", "PREMISE", "OTHER"],
            "predicted_labels": ["CLAIM", "PREMISE", "OTHER"],
        },
        {
            "dataset_name": "cdcp",
            "gold_labels": ["CLAIM", "PREMISE", "EVIDENCE", "OTHER"],
            "predicted_labels": ["CLAIM", "PREMISE", "EVIDENCE", "OTHER"],
        },
    ]
    pe_metrics = compute_multi_span_metrics(prediction_rows[:1], label_list=("CLAIM", "PREMISE", "OTHER"))
    cdcp_metrics = compute_multi_span_metrics(
        prediction_rows[1:],
        label_list=("CLAIM", "PREMISE", "EVIDENCE", "OTHER"),
    )
    assert pe_metrics["f1_macro"] == 1.0
    assert cdcp_metrics["f1_macro"] == 1.0


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


def test_load_source_candidate_transfer_rows_requires_core_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "post_id": "post-1",
                "title": "Example",
                "slug": "example",
                "candidate_id": "cand-1",
                "name": "Example Source",
                "context": "According to Example Source, this is true.",
                "gold_label": "SOURCE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_source_candidate_transfer_rows(path)
    assert rows[0]["candidate_id"] == "cand-1"
    assert rows[0]["gold_label"] == "SOURCE"


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


def test_public_sae_model_is_supported() -> None:
    assert "Qwen/Qwen2.5-7B-Instruct" in SUPPORTED_SAE_MODELS
    spec = get_sae_release_spec("Qwen/Qwen2.5-7B-Instruct")
    assert spec.repo_id == "andyrdt/saes-qwen2.5-7b-instruct"
    assert spec.hidden_state_index_for_layer(3) == 4
    assert spec.sae_id_for_layer(7) == "resid_post_layer_7/trainer_1"
    assert spec.sae_id_candidates_for_layer(7) == (
        "resid_post_layer_7/trainer_1",
        "resid_post_layer_7/trainer_0",
        "resid_post_layer_7/trainer_2",
        "resid_post_layer_7/trainer_3",
    )


def test_create_logistic_regression_supports_newer_and_older_signatures() -> None:
    calls: list[dict[str, object]] = []

    class WithMultiClass:
        def __init__(self, max_iter: int, multi_class: str) -> None:
            calls.append({"max_iter": max_iter, "multi_class": multi_class})

    class WithoutMultiClass:
        def __init__(self, max_iter: int) -> None:
            calls.append({"max_iter": max_iter})

    _create_logistic_regression(WithMultiClass)
    _create_logistic_regression(WithoutMultiClass)

    assert calls[0] == {"max_iter": 1000, "multi_class": "auto"}
    assert calls[1] == {"max_iter": 1000}


def test_emit_probe_progress_writes_status_file(tmp_path: Path, capsys) -> None:
    _emit_probe_progress(
        tmp_path,
        phase="loading_model",
        model_name="dummy/model",
        task="source_materiality",
        message="Loading model",
        split_counts={"train": 1, "dev": 0, "test": 1},
    )
    status = json.loads((tmp_path / "probe_status.json").read_text())
    assert status["phase"] == "loading_model"
    assert status["split_counts"]["train"] == 1
    assert "[probe:loading_model] Loading model" in capsys.readouterr().out


def test_emit_run_status_writes_generic_status_file(tmp_path: Path, capsys) -> None:
    emit_run_status(
        tmp_path,
        filename="train_status.json",
        prefix="train",
        phase="loading_data",
        message="Loaded benchmark",
        dataset="pe",
    )
    status = json.loads((tmp_path / "train_status.json").read_text())
    assert status["phase"] == "loading_data"
    assert status["dataset"] == "pe"
    events = (tmp_path / "train_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 1
    assert "[train:loading_data] Loaded benchmark" in capsys.readouterr().out


def test_write_run_analytics_writes_timestamped_payload(tmp_path: Path) -> None:
    payload = write_run_analytics(
        tmp_path,
        filename="train_analytics.json",
        payload={"dataset": "pe", "counts": {"train": 8}},
    )
    saved = json.loads((tmp_path / "train_analytics.json").read_text())
    assert saved["dataset"] == "pe"
    assert saved["counts"]["train"] == 8
    assert "updated_at_utc" in payload


def test_event_compatible_callback_returns_noop_for_unknown_trainer_events() -> None:
    callback = EventCompatibleCallback()
    assert callback.on_save(None, None, None) is None
    try:
        _ = callback.not_a_callback
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected non-callback attribute access to raise AttributeError.")


def test_hf_trainer_heartbeat_callback_writes_step_and_predict_events(tmp_path: Path) -> None:
    callback = HFTrainerHeartbeatCallback(
        output_dir=tmp_path,
        task="span_role",
        dataset_name="pe",
        model_name="dummy/model",
        split_counts={"train": 8, "eval": 1, "test": 1},
        step_interval=2,
        prediction_interval=2,
    )

    class Args:
        num_train_epochs = 3.0

    class State:
        max_steps = 4
        global_step = 0
        epoch = 0.0

    callback.on_train_begin(Args(), State(), None)
    State.global_step = 2
    State.epoch = 0.5
    callback.on_step_end(Args(), State(), None)
    callback.begin_prediction_stage("predict", total_examples=5)
    callback.on_prediction_step(Args(), State(), None)
    callback.on_prediction_step(Args(), State(), None)
    callback.on_predict(Args(), State(), None, metrics={"test_loss": 0.1})

    status = json.loads((tmp_path / "train_status.json").read_text())
    assert status["phase"] == "predict_complete"
    events = [json.loads(line) for line in (tmp_path / "train_events.jsonl").read_text(encoding="utf-8").splitlines()]
    phases = [event["phase"] for event in events]
    assert "train_step" in phases
    assert "predict_step" in phases
    assert "predict_complete" in phases


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


def test_write_source_candidate_transfer_summary_reports_gold_metrics(tmp_path: Path) -> None:
    (tmp_path / "abc123.json").write_text(
        json.dumps(
            {
                "post_id": "abc123",
                "title": "Example",
                "slug": "example",
                "candidates": [
                    {"predicted_label": "SOURCE", "gold_label": "SOURCE"},
                    {"predicted_label": "NOT_SOURCE", "gold_label": "NOT_SOURCE"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_source_candidate_transfer_summary(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["candidate_count"] == 2
    assert summary["labeled_candidate_count"] == 2
    assert summary["gold_metrics"]["accuracy"] == 1.0


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


def test_write_run_inventory_reports_multi_dataset_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "pe_cdcp_shared_deberta"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"test_metrics": {"accuracy": 0.7, "f1_macro": 0.6}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {"task": "span_role_multidataset", "datasets": ["pe", "cdcp"], "model_name": "dummy/model"},
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "inventory"
    payload = write_run_inventory(output_dir, run_root=run_root)

    assert payload["runs"][0]["dataset"] == "pe, cdcp"
    assert payload["runs"][0]["datasets"] == ["pe", "cdcp"]


def test_write_run_inventory_reports_summary_only_probe_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "scicite_probe"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_layer": 12,
                "best_metrics": {
                    "layer": 12,
                    "test_metrics": {"accuracy": 0.9, "f1_macro": 0.8},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {"task": "source_materiality", "dataset": "scicite", "model_name": "dummy/model"},
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "inventory"
    write_run_inventory(output_dir, run_root=run_root)
    inventory_md = (output_dir / "inventory.md").read_text(encoding="utf-8")

    assert "best_layer: `12`" in inventory_md
    assert "f1_macro: `0.8000`" in inventory_md


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


def test_package_run_artifacts_includes_summary_and_probe_model_files(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "scicite_probe"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"best_layer": 12}), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({"task": "source_materiality"}), encoding="utf-8")
    (run_dir / "layer_12.metrics.json").write_text(json.dumps({"layer": 12}), encoding="utf-8")
    (run_dir / "layer_12.probe.joblib").write_bytes(b"probe")

    output_path = tmp_path / "packaged" / "runs.tar.gz"
    manifest = package_run_artifacts(run_root=run_root, output_path=output_path)

    assert "summary.json" in manifest["runs"][0]["included"]
    assert "layer_12.metrics.json" in manifest["runs"][0]["included"]
    assert "layer_12.probe.joblib" in manifest["runs"][0]["included"]
    with tarfile.open(output_path, "r:gz") as archive:
        archive_names = set(archive.getnames())
    assert "scicite_probe/summary.json" in archive_names
    assert "scicite_probe/layer_12.metrics.json" in archive_names
    assert "scicite_probe/layer_12.probe.joblib" in archive_names
