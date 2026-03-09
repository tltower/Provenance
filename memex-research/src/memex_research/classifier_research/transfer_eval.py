from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from memex_research.classifier_research.hf_models import create_probe_components

DEFAULT_TRANSFER_MANIFEST = Path(
    "/Users/tatetower/Codex/memex-research/analysis/memex_transfer_seed_10/manifest.json"
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
    return [row for row in rows if isinstance(row, dict)]


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
        candidates.append({"name": _derive_name_from_url(url), "url": url, "author": None})
    return candidates


def build_transfer_source_candidates(*, raw_post: dict[str, Any], text: str) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in _extract_bibliography_candidates(text) + _extract_external_link_candidates(raw_post):
        name = str(candidate.get("name") or "").strip()
        url = str(candidate.get("url") or "").strip() or None
        if not name:
            continue
        deduped.setdefault(
            _source_key(name, url),
            {
                "name": name,
                "url": url,
                "author": candidate.get("author"),
            },
        )
    return list(deduped.values())


def _candidate_context_window(text: str, candidate_name: str, *, radius: int = 220) -> str:
    if not text:
        return ""
    lowered = text.lower()
    needle = candidate_name.lower()
    start = lowered.find(needle)
    if start == -1:
        for token in sorted(re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", candidate_name), key=len, reverse=True):
            start = lowered.find(token.lower())
            if start != -1:
                break
    if start == -1:
        return text[: min(len(text), radius * 2)].strip()
    left = max(0, start - radius)
    right = min(len(text), start + len(candidate_name) + radius)
    return text[left:right].strip()


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
    model.eval()
    device = next(model.parameters()).device

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        candidates = build_transfer_source_candidates(raw_post=row["post"], text=row["text"])
        candidate_outputs: list[dict[str, Any]] = []
        for candidate in candidates:
            context = _candidate_context_window(row["text"], str(candidate.get("name") or ""))
            encoded = tokenizer(context, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                result = model(**encoded)
            logits = result.logits.detach().cpu().numpy()[0]
            pred_id = int(logits.argmax())
            candidate_outputs.append(
                {
                    "name": candidate.get("name"),
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
                    "candidates": candidate_outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )


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
    model.eval()
    device = next(model.parameters()).device

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
    model.eval()
    device = next(model.parameters()).device

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in _load_transfer_posts(manifest_path):
        candidates = build_transfer_source_candidates(raw_post=row["post"], text=row["text"])
        candidate_outputs: list[dict[str, Any]] = []
        for candidate in candidates:
            context = _candidate_context_window(row["text"], str(candidate.get("name") or ""))
            encoded = tokenizer(context, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                result = model(**encoded)
            hidden_state = result.hidden_states[best_layer][0].detach().cpu().numpy()
            attention_mask = encoded["attention_mask"][0].detach().cpu().numpy().astype(bool)
            pooled = hidden_state[attention_mask].mean(axis=0).reshape(1, -1)
            pred_id = int(classifier.predict(pooled)[0])
            candidate_outputs.append(
                {
                    "name": candidate.get("name"),
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
                    "candidates": candidate_outputs,
                    "best_layer": best_layer,
                },
                indent=2,
                sort_keys=True,
            )
        )


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
    model.eval()
    device = next(model.parameters()).device

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
