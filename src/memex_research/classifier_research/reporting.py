from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _safe_git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except Exception:  # pragma: no cover - package may be absent in local non-ML installs
        return None


def collect_run_diagnostics(*, repo_root: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_head": _safe_git_head(repo_root),
        "package_versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "scikit-learn": _package_version("scikit-learn"),
            "accelerate": _package_version("accelerate"),
            "protobuf": _package_version("protobuf"),
        },
    }
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        diagnostics["cuda"] = {"available": False}
    else:
        available = bool(torch.cuda.is_available())
        diagnostics["cuda"] = {
            "available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
        }
    return diagnostics


def _metrics_summary_lines(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    metrics = payload.get("test_metrics")
    if isinstance(metrics, Mapping):
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                lines.append(f"- `{key}`: `{value:.4f}`")
    elif payload.get("best_metrics") and isinstance(payload["best_metrics"], Mapping):
        best = payload["best_metrics"]
        metrics = best.get("test_metrics") if isinstance(best, Mapping) else None
        if isinstance(metrics, Mapping):
            layer = best.get("layer")
            if layer is not None:
                lines.append(f"- `best_layer`: `{layer}`")
            for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    lines.append(f"- `{key}`: `{value:.4f}`")
    return lines


def write_run_reports(
    *,
    output_dir: Path,
    repo_root: Path,
    title: str,
    payload: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = collect_run_diagnostics(repo_root=repo_root)
    if extra:
        diagnostics["extra"] = dict(extra)
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))

    lines = [f"# {title}", "", "## Summary", ""]
    lines.extend(_metrics_summary_lines(payload))
    if not any(line.startswith("- `") for line in lines):
        lines.append("- See the JSON artifacts for full details.")
    if extra:
        lines.extend(["", "## Run Metadata", ""])
        for key, value in extra.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json` or `summary.json`",
            "- `run_config.json`",
            "- `diagnostics.json`",
        ]
    )
    if (output_dir / "predictions.jsonl").exists():
        lines.append("- `predictions.jsonl`")
    if (output_dir / "transfer").exists():
        lines.append("- `transfer/`")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
