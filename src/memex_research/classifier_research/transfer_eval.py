from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from memex_research.classifier_research.hf_models import create_probe_components
from memex_research.classifier_research.reporting import write_run_reports


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_TRANSFER_MANIFEST = (
    _repo_root() / "analysis" / "memex_transfer_seed_10" / "manifest.json"
)
DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH = (
    _repo_root() / "analysis" / "memex_transfer_seed_10" / "source_candidates.jsonl"
)
_NUMBERED_ENTRY_RE = re.compile(r"(?:^|\s)(\d+)\.\s+(.+?)(?=(?:\s+\d+\.\s+)|$)")
_URL_RE = re.compile(r"https?://\S+")
_QUOTED_TITLE_RE = re.compile(r"[“\"]([^”\"]{5,200})[”\"]")


def _require_ml_stack() -> tuple[Any, Any, Any]:
    try:
        import joblib  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Transfer evaluation dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return joblib, np, (torch, transformers)


def _load_local_tokenizer(model_dir: Path) -> Any:
    _joblib, _np, stack = _require_ml_stack()
    _torch, transformers = stack
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_transfer_manifest(path: Path = DEFAULT_TRANSFER_MANIFEST) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    rows = manifest.get("posts", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid transfer manifest at {path}")
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        resolved = dict(row)
        for key in ("raw_path", "text_path"):
            candidate = Path(str(resolved[key]))
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            resolved[key] = str(candidate)
        resolved_rows.append(resolved)
    return resolved_rows


def _load_transfer_posts(path: Path = DEFAULT_TRANSFER_MANIFEST) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    for row in load_transfer_manifest(path):
        raw_path = Path(str(row["raw_path"]))
        text_path = Path(str(row["text_path"]))
        raw = json.loads(raw_path.read_text())
        posts.append(
            {
                "post_id": str(row["post_id"]),
                "title": str(row["title"]),
                "slug": str(row["slug"]),
                "text": text_path.read_text(encoding="utf-8"),
                "post": raw,
            }
        )
    return posts


def load_source_candidate_transfer_rows(
    path: Path = DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = line.strip()
            if not clean:
                continue
            row = json.loads(clean)
            if not isinstance(row, dict):
                raise RuntimeError(f"Invalid candidate-only transfer row at {path}:{line_number}")
            context = str(row.get("context") or "").strip()
            name = str(row.get("name") or "").strip()
            post_id = str(row.get("post_id") or "").strip()
            if not context or not name or not post_id:
                raise RuntimeError(
                    f"Candidate-only transfer rows require post_id, name, and context at {path}:{line_number}"
                )
            rows.append(
                {
                    "post_id": post_id,
                    "title": str(row.get("title") or ""),
                    "slug": str(row.get("slug") or ""),
                    "candidate_id": str(row.get("candidate_id") or f"{post_id}:{len(rows)}"),
                    "name": name,
                    "context": context,
                    "url": str(row.get("url") or "").strip() or None,
                    "origin": str(row.get("origin") or "curated_candidate"),
                    "gold_label": str(row.get("gold_label") or "").strip() or None,
                }
            )
    return rows


def _source_key(name: str, url: str | None) -> tuple[str, str]:
    return ((url or "").strip().lower(), " ".join(name.lower().split()))


def _derive_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_bits = [bit for bit in parsed.path.split("/") if bit]
    if path_bits:
        name = path_bits[-1].replace("-", " ").replace("_", " ").strip()
        if name:
            return name
    return (parsed.netloc or url).replace("www.", "").strip()


def _candidate_name_tokens(name: str) -> list[str]:
    return sorted(
        {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", name)
        },
        key=len,
        reverse=True,
    )


def _looks_like_title_candidate(name: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", name)
    if not 1 <= len(tokens) <= 12:
        return False
    uppercase_initial = sum(1 for token in tokens if token[0].isupper())
    if len(tokens) == 1:
        return uppercase_initial == 1 and len(tokens[0]) >= 4
    return uppercase_initial >= max(2, len(tokens) // 2)


def _parse_bibliography_entry(entry_text: str) -> dict[str, Any] | None:
    clean = " ".join(entry_text.split()).strip()
    if not clean:
        return None
    url_match = _URL_RE.search(clean)
    url = url_match.group(0) if url_match else None
    title_match = _QUOTED_TITLE_RE.search(clean)
    title = title_match.group(1).strip() if title_match else ""
    author = ""
    comma_parts = [part.strip() for part in clean.split(",") if part.strip()]
    if comma_parts:
        author = comma_parts[0]
        if not title and len(comma_parts) >= 2:
            title = comma_parts[1]
    if not title:
        title = clean[:160]
    if len(title) < 4:
        return None
    return {"name": title, "url": url, "author": author or None}


def _extract_bibliography_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in _NUMBERED_ENTRY_RE.finditer(text):
        parsed = _parse_bibliography_entry(match.group(2))
        if parsed is not None:
            candidates.append(parsed)
    return candidates


def _extract_external_link_candidates(raw_post: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    external_links = raw_post.get("external_links") or []
    if not isinstance(external_links, list):
        return candidates
    for value in external_links:
        url = str(value or "").strip()
        if not url:
            continue
        name = _derive_name_from_url(url)
        candidates.append({"name": name, "url": url, "author": None, "origin": "external_link"})
    return candidates


def _extract_inline_title_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in _QUOTED_TITLE_RE.finditer(text):
        title = " ".join(match.group(1).split()).strip()
        if len(title) < 5:
            continue
        if not _looks_like_title_candidate(title):
            continue
        candidates.append({"name": title, "url": None, "author": None, "origin": "quoted_title"})
    return candidates


def _candidate_context_window(text: str, candidate_name: str, *, radius: int = 220) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    needle = candidate_name.lower()
    start = lowered.find(needle)
    if start == -1:
        for token in _candidate_name_tokens(candidate_name):
            start = lowered.find(token.lower())
            if start != -1:
                break
    if start == -1:
        return None
    left = max(0, start - radius)
    right = min(len(text), start + len(candidate_name) + radius)
    return text[left:right].strip()


def build_transfer_source_candidates(*, raw_post: dict[str, Any], text: str) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in (
        _extract_bibliography_candidates(text)
        + _extract_inline_title_candidates(text)
        + _extract_external_link_candidates(raw_post)
    ):
        name = str(candidate.get("name") or "").strip()
        url = str(candidate.get("url") or "").strip() or None
        if not name:
            continue
        if candidate.get("origin") == "external_link":
            has_text_anchor = _candidate_context_window(text, name) is not None
            if not _looks_like_title_candidate(name) and not has_text_anchor:
                continue
        deduped.setdefault(
            _source_key(name, url),
            {
                "name": name,
                "url": url,
                "author": candidate.get("author"),
                "origin": candidate.get("origin"),
            },
        )
    return list(deduped.values())


def _write_source_transfer_summary(output_dir: Path) -> None:
    rows = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("*.json"))
        if path.name not in {"summary.json", "diagnostics.json", "run_config.json", "metrics.json"}
    ]
    label_counts: Counter[str] = Counter()
    generated_origin_counts: Counter[str] = Counter()
    kept_origin_counts: Counter[str] = Counter()
    total_candidates = 0
    generated_candidates = 0
    skipped_no_context = 0
    posts_with_candidates = 0
    for row in rows:
        candidates = row.get("candidates", [])
        if candidates:
            posts_with_candidates += 1
        total_candidates += len(candidates)
        generated_candidates += int(row.get("generated_candidate_count", len(candidates)))
        skipped_no_context += int(row.get("skipped_no_context_count", 0))
        for origin, count in dict(row.get("generated_origin_counts") or {}).items():
            generated_origin_counts[str(origin)] += int(count)
        for origin, count in dict(row.get("kept_origin_counts") or {}).items():
            kept_origin_counts[str(origin)] += int(count)
        for candidate in candidates:
            label = str(candidate.get("predicted_label") or "")
            if label:
                label_counts[label] += 1
    summary = {
        "post_count": len(rows),
        "posts_with_candidates": posts_with_candidates,
        "generated_candidates": generated_candidates,
        "total_candidates": total_candidates,
        "skipped_no_context": skipped_no_context,
        "generated_origin_counts": dict(sorted(generated_origin_counts.items())),
        "kept_origin_counts": dict(sorted(kept_origin_counts.items())),
        "predicted_label_counts": dict(sorted(label_counts.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_run_reports(
        output_dir=output_dir,
        repo_root=_repo_root(),
        title="Source Transfer Evaluation",
        payload=summary,
        extra={"problem_type": "qualitative_transfer"},
    )


def _write_source_candidate_transfer_summary(output_dir: Path) -> None:
    rows = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("*.json"))
        if path.name not in {"summary.json", "diagnostics.json", "run_config.json", "metrics.json"}
    ]
    label_counts: Counter[str] = Counter()
    gold_rows: list[tuple[str, str]] = []
    candidate_count = 0
    for row in rows:
        candidates = row.get("candidates", [])
        candidate_count += len(candidates)
        for candidate in candidates:
            predicted = str(candidate.get("predicted_label") or "")
            gold = str(candidate.get("gold_label") or "")
            if predicted:
                label_counts[predicted] += 1
            if gold:
                gold_rows.append((gold, predicted))
    summary: dict[str, Any] = {
        "post_count": len(rows),
        "candidate_count": candidate_count,
        "predicted_label_counts": dict(sorted(label_counts.items())),
    }
    if gold_rows:
        try:
            from sklearn.metrics import (  # type: ignore[import-not-found]
                accuracy_score,
                precision_recall_fscore_support,
            )
        except ImportError:
            pass
        else:
            target_labels = ["SOURCE", "NOT_SOURCE"]
            gold_labels = [item[0] for item in gold_rows]
            predicted_labels = [item[1] for item in gold_rows]
            precision, recall, f1, _support = precision_recall_fscore_support(
                gold_labels,
                predicted_labels,
                labels=target_labels,
                average="macro",
                zero_division=0,
            )
            summary["labeled_candidate_count"] = len(gold_rows)
            summary["gold_metrics"] = {
                "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_run_reports(
        output_dir=output_dir,
        repo_root=_repo_root(),
        title="Candidate-Only Source Transfer Evaluation",
        payload=summary,
        extra={"problem_type": "candidate_only_transfer"},
    )


def _write_span_transfer_summary(output_dir: Path) -> None:
    rows = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("*.json"))
        if path.name not in {"summary.json", "diagnostics.json", "run_config.json", "metrics.json"}
    ]
    label_counts: Counter[str] = Counter()
    total_spans = 0
    for row in rows:
        spans = row.get("spans", [])
        total_spans += len(spans)
        for span in spans:
            label = str(span.get("label") or "")
            if label:
                label_counts[label] += 1
    summary = {
        "post_count": len(rows),
        "total_spans": total_spans,
        "span_label_counts": dict(sorted(label_counts.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_run_reports(
        output_dir=output_dir,
        repo_root=_repo_root(),
        title="Span Role Transfer Evaluation",
        payload=summary,
        extra={"problem_type": "qualitative_transfer"},
    )


def _group_labeled_tokens(tokens: list[str], labels: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current_tokens: list[str] = []
    current_label: str | None = None
    start_index = 0
    for index, (token, label) in enumerate(zip(tokens, labels, strict=True)):
        if label == "OTHER":
            if current_label is not None:
                spans.append(
                    {
                        "label": current_label,
                        "text": " ".join(current_tokens),
                        "start_token": start_index,
                        "end_token": index - 1,
                    }
                )
                current_tokens = []
                current_label = None
            continue
        if current_label == label:
            current_tokens.append(token)
            continue
        if current_label is not None:
            spans.append(
                {
                    "label": current_label,
                    "text": " ".join(current_tokens),
                    "start_token": start_index,
                    "end_token": index - 1,
                }
            )
        current_label = label
        current_tokens = [token]
        start_index = index
    if current_label is not None:
        spans.append(
            {
                "label": current_label,
                "text": " ".join(current_tokens),
                "start_token": start_index,
                "end_token": len(tokens) - 1,
            }
        )
    return spans


def run_source_transfer_hf_model(
    *,
    model_dir: Path,
    output_dir: Path,
    manifest_path: Path = DEFAULT_TRANSFER_MANIFEST,
    max_length: int = 256,
) -> None:
    _joblib, _np, stack = _require_ml_stack()
    torch, transformers = stack
    tokenizer = _load_local_tokenizer(model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        candidates = build_transfer_source_candidates(raw_post=row["post"], text=row["text"])
        candidate_outputs: list[dict[str, Any]] = []
        generated_origin_counts: Counter[str] = Counter(
            str(candidate.get("origin") or "unknown") for candidate in candidates
        )
        kept_origin_counts: Counter[str] = Counter()
        skipped_no_context = 0
        for candidate in candidates:
            context = _candidate_context_window(row["text"], str(candidate.get("name") or ""))
            if not context:
                skipped_no_context += 1
                continue
            encoded = tokenizer(context, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                result = model(**encoded)
            logits = result.logits.detach().cpu().numpy()[0]
            pred_id = int(logits.argmax())
            origin = str(candidate.get("origin") or "unknown")
            kept_origin_counts[origin] += 1
            candidate_outputs.append(
                {
                    "name": candidate.get("name"),
                    "origin": origin,
                    "url": candidate.get("url"),
                    "context": context,
                    "predicted_label": model.config.id2label[pred_id],
                    "logits": logits.tolist(),
                }
            )
        (output_dir / f"{row['post_id']}.json").write_text(
            json.dumps(
                {
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "slug": row["slug"],
                    "generated_candidate_count": len(candidates),
                    "skipped_no_context_count": skipped_no_context,
                    "generated_origin_counts": dict(sorted(generated_origin_counts.items())),
                    "kept_origin_counts": dict(sorted(kept_origin_counts.items())),
                    "candidates": candidate_outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
    _write_source_transfer_summary(output_dir)


def run_source_candidate_transfer_hf_model(
    *,
    model_dir: Path,
    output_dir: Path,
    candidates_path: Path = DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH,
    max_length: int = 256,
) -> None:
    _joblib, _np, stack = _require_ml_stack()
    torch, transformers = stack
    tokenizer = _load_local_tokenizer(model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows: dict[str, dict[str, Any]] = {}
    for row in load_source_candidate_transfer_rows(candidates_path):
        bucket = grouped_rows.setdefault(
            row["post_id"],
            {
                "post_id": row["post_id"],
                "title": row["title"],
                "slug": row["slug"],
                "candidates": [],
            },
        )
        encoded = tokenizer(str(row["context"]), truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**encoded)
        logits = result.logits.detach().cpu().numpy()[0]
        pred_id = int(logits.argmax())
        bucket["candidates"].append(
            {
                "candidate_id": row["candidate_id"],
                "name": row["name"],
                "origin": row["origin"],
                "url": row["url"],
                "context": row["context"],
                "gold_label": row["gold_label"],
                "predicted_label": model.config.id2label[pred_id],
                "logits": logits.tolist(),
            }
        )
    for post_id, row in grouped_rows.items():
        (output_dir / f"{post_id}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    _write_source_candidate_transfer_summary(output_dir)


def run_span_transfer_hf_model(
    *,
    model_dir: Path,
    output_dir: Path,
    manifest_path: Path = DEFAULT_TRANSFER_MANIFEST,
    max_length: int = 256,
) -> None:
    _joblib, _np, stack = _require_ml_stack()
    torch, transformers = stack
    tokenizer = _load_local_tokenizer(model_dir)
    model = transformers.AutoModelForTokenClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        tokens = re.findall(r"\S+", row["text"])
        encoded: Any = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**encoded)
        pred_ids = result.logits.detach().cpu().numpy().argmax(axis=-1)[0].tolist()
        projected_labels = ["OTHER"] * len(tokens)
        seen_word_ids: set[int] = set()
        for position, word_id in enumerate(encoded.word_ids(0)):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            projected_labels[word_id] = model.config.id2label[int(pred_ids[position])]
        (output_dir / f"{row['post_id']}.json").write_text(
            json.dumps(
                {
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "slug": row["slug"],
                    "spans": _group_labeled_tokens(tokens, projected_labels),
                },
                indent=2,
                sort_keys=True,
            )
        )
    _write_span_transfer_summary(output_dir)


def run_source_transfer_probe(
    *,
    probe_dir: Path,
    output_dir: Path,
    model_name: str,
    manifest_path: Path = DEFAULT_TRANSFER_MANIFEST,
    max_length: int = 256,
) -> None:
    joblib, _np, stack = _require_ml_stack()
    torch, _transformers = stack
    summary = json.loads((probe_dir / "summary.json").read_text())
    best_layer = int(summary["best_layer"])
    classifier = joblib.load(probe_dir / f"layer_{best_layer:02d}.probe.joblib")
    tokenizer, model = create_probe_components(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        candidates = build_transfer_source_candidates(raw_post=row["post"], text=row["text"])
        candidate_outputs: list[dict[str, Any]] = []
        generated_origin_counts: Counter[str] = Counter(
            str(candidate.get("origin") or "unknown") for candidate in candidates
        )
        kept_origin_counts: Counter[str] = Counter()
        skipped_no_context = 0
        for candidate in candidates:
            context = _candidate_context_window(row["text"], str(candidate.get("name") or ""))
            if not context:
                skipped_no_context += 1
                continue
            encoded = tokenizer(context, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                result = model(**encoded)
            hidden_state = result.hidden_states[best_layer][0].detach().cpu().numpy()
            attention_mask = encoded["attention_mask"][0].detach().cpu().numpy().astype(bool)
            pooled = hidden_state[attention_mask].mean(axis=0).reshape(1, -1)
            pred_id = int(classifier.predict(pooled)[0])
            origin = str(candidate.get("origin") or "unknown")
            kept_origin_counts[origin] += 1
            candidate_outputs.append(
                {
                    "name": candidate.get("name"),
                    "origin": origin,
                    "url": candidate.get("url"),
                    "context": context,
                    "predicted_label": summary["best_metrics"]["label_list"][pred_id],
                }
            )
        (output_dir / f"{row['post_id']}.json").write_text(
            json.dumps(
                {
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "slug": row["slug"],
                    "generated_candidate_count": len(candidates),
                    "skipped_no_context_count": skipped_no_context,
                    "generated_origin_counts": dict(sorted(generated_origin_counts.items())),
                    "kept_origin_counts": dict(sorted(kept_origin_counts.items())),
                    "candidates": candidate_outputs,
                    "best_layer": best_layer,
                },
                indent=2,
                sort_keys=True,
            )
        )
    _write_source_transfer_summary(output_dir)


def run_source_candidate_transfer_probe(
    *,
    probe_dir: Path,
    output_dir: Path,
    model_name: str,
    candidates_path: Path = DEFAULT_SOURCE_CANDIDATE_TRANSFER_PATH,
    max_length: int = 256,
) -> None:
    joblib, _np, stack = _require_ml_stack()
    torch, _transformers = stack
    summary = json.loads((probe_dir / "summary.json").read_text())
    best_layer = int(summary["best_layer"])
    classifier = joblib.load(probe_dir / f"layer_{best_layer:02d}.probe.joblib")
    tokenizer, model = create_probe_components(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    grouped_rows: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in load_source_candidate_transfer_rows(candidates_path):
        bucket = grouped_rows.setdefault(
            row["post_id"],
            {
                "post_id": row["post_id"],
                "title": row["title"],
                "slug": row["slug"],
                "candidates": [],
                "best_layer": best_layer,
            },
        )
        encoded = tokenizer(str(row["context"]), truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**encoded)
        hidden_state = result.hidden_states[best_layer][0].detach().cpu().numpy()
        attention_mask = encoded["attention_mask"][0].detach().cpu().numpy().astype(bool)
        pooled = hidden_state[attention_mask].mean(axis=0).reshape(1, -1)
        pred_id = int(classifier.predict(pooled)[0])
        bucket["candidates"].append(
            {
                "candidate_id": row["candidate_id"],
                "name": row["name"],
                "origin": row["origin"],
                "url": row["url"],
                "context": row["context"],
                "gold_label": row["gold_label"],
                "predicted_label": summary["best_metrics"]["label_list"][pred_id],
            }
        )
    for post_id, row in grouped_rows.items():
        (output_dir / f"{post_id}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    _write_source_candidate_transfer_summary(output_dir)


def run_span_transfer_probe(
    *,
    probe_dir: Path,
    output_dir: Path,
    model_name: str,
    manifest_path: Path = DEFAULT_TRANSFER_MANIFEST,
    max_length: int = 256,
) -> None:
    joblib, _np, stack = _require_ml_stack()
    torch, _transformers = stack
    summary = json.loads((probe_dir / "summary.json").read_text())
    best_layer = int(summary["best_layer"])
    classifier = joblib.load(probe_dir / f"layer_{best_layer:02d}.probe.joblib")
    tokenizer, model = create_probe_components(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        tokens = re.findall(r"\S+", row["text"])
        encoded: Any = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**encoded)
        hidden_state = result.hidden_states[best_layer][0].detach().cpu().numpy()
        projected_labels = ["OTHER"] * len(tokens)
        seen_word_ids: set[int] = set()
        for position, word_id in enumerate(encoded.word_ids(0)):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            pred_id = int(classifier.predict(hidden_state[position].reshape(1, -1))[0])
            projected_labels[word_id] = summary["best_metrics"]["label_list"][pred_id]
        (output_dir / f"{row['post_id']}.json").write_text(
            json.dumps(
                {
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "slug": row["slug"],
                    "best_layer": best_layer,
                    "spans": _group_labeled_tokens(tokens, projected_labels),
                },
                indent=2,
                sort_keys=True,
            )
        )
    _write_span_transfer_summary(output_dir)
