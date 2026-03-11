from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any


def _json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _dataset_fields(payload: dict[str, Any] | None) -> tuple[str | None, list[str] | None]:
    if not isinstance(payload, dict):
        return None, None
    dataset = payload.get("dataset")
    if isinstance(dataset, str) and dataset:
        return dataset, [dataset]
    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        normalized = [str(item) for item in datasets if str(item)]
        if normalized:
            return ", ".join(normalized), normalized
    return None, None


def summarize_run_directory(run_dir: Path) -> dict[str, Any]:
    metrics = _json_if_exists(run_dir / "metrics.json")
    summary = _json_if_exists(run_dir / "summary.json")
    run_config = _json_if_exists(run_dir / "run_config.json")
    diagnostics = _json_if_exists(run_dir / "diagnostics.json")
    transfer_summary = _json_if_exists(run_dir / "transfer" / "summary.json")
    primary_payload = run_config or metrics or summary or {}
    dataset, datasets = _dataset_fields(primary_payload)

    return {
        "run_name": run_dir.name,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "files": {
            "metrics_json": (run_dir / "metrics.json").exists(),
            "summary_json": (run_dir / "summary.json").exists(),
            "run_config_json": (run_dir / "run_config.json").exists(),
            "diagnostics_json": (run_dir / "diagnostics.json").exists(),
            "report_md": (run_dir / "report.md").exists(),
            "predictions_jsonl": (run_dir / "predictions.jsonl").exists(),
            "model_dir": (run_dir / "model").exists(),
            "trainer_dir": (run_dir / "trainer").exists(),
            "transfer_dir": (run_dir / "transfer").exists(),
        },
        "sizes": {
            "model_bytes": _file_size(run_dir / "model") if (run_dir / "model").exists() else 0,
            "trainer_bytes": _file_size(run_dir / "trainer") if (run_dir / "trainer").exists() else 0,
            "transfer_bytes": _file_size(run_dir / "transfer") if (run_dir / "transfer").exists() else 0,
            "predictions_bytes": (run_dir / "predictions.jsonl").stat().st_size
            if (run_dir / "predictions.jsonl").exists()
            else 0,
        },
        "task": primary_payload.get("task"),
        "dataset": dataset,
        "datasets": datasets,
        "model_name": primary_payload.get("model_name"),
        "metrics": metrics,
        "summary": summary,
        "transfer_summary": transfer_summary,
        "diagnostics": diagnostics,
    }


def collect_run_inventory(run_root: Path) -> list[dict[str, Any]]:
    if not run_root.exists():
        return []
    return [summarize_run_directory(path) for path in sorted(run_root.iterdir()) if path.is_dir()]


def _append_metric_lines(lines: list[str], metrics: dict[str, Any]) -> bool:
    test_metrics = metrics.get("test_metrics") if isinstance(metrics, dict) else None
    if isinstance(test_metrics, dict):
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            value = test_metrics.get(key)
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: `{value:.4f}`")
        return True
    return False


def _report_lines(rows: list[dict[str, Any]], *, title: str) -> list[str]:
    lines = [f"# {title}", "", f"- run_count: `{len(rows)}`", ""]
    for row in rows:
        lines.append(f"## {row['run_name']}")
        lines.append("")
        lines.append(f"- task: `{row.get('task')}`")
        lines.append(f"- dataset: `{row.get('dataset')}`")
        lines.append(f"- model: `{row.get('model_name')}`")
        sizes = row["sizes"]
        lines.append(f"- model_bytes: `{sizes['model_bytes']}`")
        lines.append(f"- trainer_bytes: `{sizes['trainer_bytes']}`")
        lines.append(f"- transfer_bytes: `{sizes['transfer_bytes']}`")
        metrics = row.get("metrics", {})
        wrote_primary_metrics = _append_metric_lines(lines, metrics if isinstance(metrics, dict) else {})
        summary = row.get("summary")
        if isinstance(summary, dict) and not wrote_primary_metrics:
            best_metrics = summary.get("best_metrics")
            if isinstance(best_metrics, dict):
                best_layer = best_metrics.get("layer")
                if isinstance(best_layer, int):
                    lines.append(f"- best_layer: `{best_layer}`")
                _append_metric_lines(lines, best_metrics)
        transfer_summary = row.get("transfer_summary")
        if isinstance(transfer_summary, dict):
            lines.append(
                f"- transfer_posts_with_candidates: `{transfer_summary.get('posts_with_candidates', transfer_summary.get('post_count'))}`"
            )
            lines.append(f"- transfer_total_candidates: `{transfer_summary.get('total_candidates', 0)}`")
        lines.append("")
    return lines


def write_run_inventory(output_dir: Path, *, run_root: Path, title: str = "Run Inventory") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_run_inventory(run_root)
    payload = {
        "run_root": str(run_root),
        "run_count": len(rows),
        "runs": rows,
    }
    (output_dir / "inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_dir / "inventory.md").write_text("\n".join(_report_lines(rows, title=title)) + "\n")
    return payload


def _iter_top_level_report_artifacts(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".md"}
    )


def _iter_top_level_model_artifacts(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.suffix == ".joblib"
    )


def package_run_artifacts(
    *,
    run_root: Path,
    output_path: Path,
    run_names: list[str] | None = None,
    include_models: bool = True,
    include_transfer: bool = True,
    include_predictions: bool = True,
    include_reports: bool = True,
    include_trainer: bool = False,
) -> dict[str, Any]:
    selected = collect_run_inventory(run_root)
    if run_names is not None:
        wanted = set(run_names)
        selected = [row for row in selected if row["run_name"] in wanted]

    manifest_rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for row in selected:
            run_dir = Path(str(row["path"]))
            included: list[str] = []
            if include_reports:
                for path in _iter_top_level_report_artifacts(run_dir):
                    archive.add(path, arcname=f"{run_dir.name}/{path.name}")
                    included.append(path.name)
            if include_predictions and (run_dir / "predictions.jsonl").exists():
                archive.add(run_dir / "predictions.jsonl", arcname=f"{run_dir.name}/predictions.jsonl")
                included.append("predictions.jsonl")
            if include_models and (run_dir / "model").exists():
                archive.add(run_dir / "model", arcname=f"{run_dir.name}/model")
                included.append("model/")
            if include_models:
                for path in _iter_top_level_model_artifacts(run_dir):
                    archive.add(path, arcname=f"{run_dir.name}/{path.name}")
                    included.append(path.name)
            if include_transfer and (run_dir / "transfer").exists():
                archive.add(run_dir / "transfer", arcname=f"{run_dir.name}/transfer")
                included.append("transfer/")
            if include_trainer and (run_dir / "trainer").exists():
                archive.add(run_dir / "trainer", arcname=f"{run_dir.name}/trainer")
                included.append("trainer/")
            manifest_rows.append(
                {
                    "run_name": run_dir.name,
                    "source_path": str(run_dir),
                    "included": included,
                }
            )

    manifest = {
        "run_root": str(run_root),
        "output_path": str(output_path),
        "run_count": len(manifest_rows),
        "include_models": include_models,
        "include_transfer": include_transfer,
        "include_predictions": include_predictions,
        "include_reports": include_reports,
        "include_trainer": include_trainer,
        "runs": manifest_rows,
    }
    (output_path.with_suffix(output_path.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest
