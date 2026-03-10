from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memex_research.classifier_research.hf_models import (
    create_probe_components,
    get_sae_release_spec,
)
from memex_research.classifier_research.probes import (
    _compute_classification_metrics,
    _create_logistic_regression,
    _device_for_model,
    _load_cached_label_strings,
    _load_cached_layer_vectors,
    _require_probe_stack,
    _sequence_hidden_state_cache,
    _token_hidden_state_cache,
)
from memex_research.classifier_research.reporting import write_run_reports
from memex_research.classifier_research.splits import load_jsonl_splits
from memex_research.classifier_research.tasks import (
    SOURCE_MATERIALITY_LABELS,
    SPAN_ROLE_LABELS,
    TASK_SOURCE_MATERIALITY,
    TASK_SPAN_ROLE,
)


def _require_sae_stack() -> tuple[Any, Any, Any, Any, Any]:
    joblib, np, torch, libs = _require_probe_stack()
    try:
        from sae_lens import SAE  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SAE dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return joblib, np, torch, libs, SAE


def _emit_sae_progress(
    output_dir: Path,
    *,
    phase: str,
    model_name: str,
    task: str,
    message: str,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "task": task,
        "model_name": model_name,
        "message": message,
    }
    if extra:
        payload.update(extra)
    (output_dir / "sae_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[sae:{phase}] {message}", flush=True)


def _load_pretrained_sae(*, model_name: str, sae_layer: int, device: Any) -> tuple[Any, str, str]:
    _joblib, _np, _torch, _libs, SAE = _require_sae_stack()
    release_spec = get_sae_release_spec(model_name)
    sae_id = release_spec.sae_id_for_layer(sae_layer)
    loaded = SAE.from_pretrained(
        release=release_spec.release,
        sae_id=sae_id,
        device=str(device),
    )
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.to(device)
    sae.eval()
    return sae, release_spec.release, sae_id


def _batched_sae_feature_scores(
    *,
    sae: Any,
    vectors: Any,
    batch_size: int,
) -> tuple[Any, Any]:
    _joblib, np, torch, _libs, _SAE = _require_sae_stack()
    feature_sum = None
    feature_count = None
    for start in range(0, vectors.shape[0], batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(next(sae.parameters()).device)
        with torch.no_grad():
            features = sae.encode(batch)
        if isinstance(features, tuple):
            features = features[0]
        features = features.detach()
        positive = features > 0
        batch_sum = features.sum(dim=0).cpu().numpy()
        batch_count = positive.sum(dim=0).cpu().numpy()
        if feature_sum is None:
            feature_sum = batch_sum
            feature_count = batch_count
        else:
            feature_sum += batch_sum
            feature_count += batch_count
    if feature_sum is None or feature_count is None:
        raise RuntimeError("No SAE features produced from cached hidden states.")
    return feature_sum, feature_count


def _select_sae_features(
    *,
    sae: Any,
    train_vectors: Any,
    feature_cap: int,
    batch_size: int,
) -> list[int]:
    _joblib, np, _torch, _libs, _SAE = _require_sae_stack()
    feature_sum, feature_count = _batched_sae_feature_scores(
        sae=sae,
        vectors=train_vectors,
        batch_size=batch_size,
    )
    valid = feature_count > 0
    if not valid.any():
        raise RuntimeError("SAE produced no active features on the training split.")
    scores = feature_sum.astype(np.float64)
    candidate_indices = np.flatnonzero(valid)
    ranking = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    return ranking[:feature_cap].tolist()


def _encode_selected_features(
    *,
    sae: Any,
    vectors: Any,
    selected_features: list[int],
    batch_size: int,
) -> Any:
    _joblib, np, torch, _libs, _SAE = _require_sae_stack()
    batches: list[Any] = []
    feature_index = np.array(selected_features, dtype=np.int64)
    for start in range(0, vectors.shape[0], batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(next(sae.parameters()).device)
        with torch.no_grad():
            features = sae.encode(batch)
        if isinstance(features, tuple):
            features = features[0]
        encoded = features.detach().cpu().numpy()[:, feature_index].astype(np.float32)
        batches.append(encoded)
    return np.vstack(batches)


def run_sae_experiment(
    *,
    task: str,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    max_length: int = 256,
    layers: tuple[int, ...] | None = None,
    feature_cap: int = 1024,
    batch_size: int = 128,
) -> dict[str, Any]:
    joblib, _np, _torch, libs, _SAE = _require_sae_stack()
    LogisticRegression, _accuracy_score, _prf = libs

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = load_jsonl_splits(input_dir)
    train_rows = split_rows["train"]
    dev_rows = split_rows.get("dev")
    test_rows = split_rows["test"]

    release_spec = get_sae_release_spec(model_name)
    selected_sae_layers = (
        tuple(release_spec.available_layers)
        if layers is None
        else tuple(dict.fromkeys(layers))
    )
    invalid_layers = [layer for layer in selected_sae_layers if layer not in release_spec.available_layers]
    if invalid_layers:
        supported = ", ".join(str(layer) for layer in release_spec.available_layers)
        raise ValueError(
            f"Unsupported SAE layers {invalid_layers} for {model_name!r}. "
            f"Expected one of: {supported}"
        )
    hidden_state_layers = tuple(
        release_spec.hidden_state_index_for_layer(layer) for layer in selected_sae_layers
    )

    _emit_sae_progress(
        output_dir,
        phase="loading_model",
        task=task,
        model_name=model_name,
        message="Loading tokenizer and base model for SAE experiment",
        split_counts={
            "train": len(train_rows),
            "dev": len(dev_rows) if dev_rows else 0,
            "test": len(test_rows),
        },
        max_length=max_length,
        sae_release=release_spec.release,
        sae_layers=list(selected_sae_layers),
    )
    tokenizer, model = create_probe_components(model_name)
    _emit_sae_progress(
        output_dir,
        phase="model_loaded",
        task=task,
        model_name=model_name,
        message="Tokenizer and model loaded; starting hidden-state caching",
    )
    model_device = _device_for_model(model)

    with tempfile.TemporaryDirectory(prefix="sae-cache-", dir=output_dir) as cache_root:
        cache_dir = Path(cache_root)

        if task == TASK_SOURCE_MATERIALITY:
            label_list = list(SOURCE_MATERIALITY_LABELS)
            cached_layers = _sequence_hidden_state_cache(
                train_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="train",
                layers=hidden_state_layers,
                progress_callback=lambda **kwargs: _emit_sae_progress(
                    output_dir,
                    phase="caching_hidden_states",
                    task=task,
                    model_name=model_name,
                    message="Caching hidden states for source-materiality split",
                    **kwargs,
                ),
            )
            if dev_rows:
                _sequence_hidden_state_cache(
                    dev_rows,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    split_name="dev",
                    layers=cached_layers,
                    progress_callback=lambda **kwargs: _emit_sae_progress(
                        output_dir,
                        phase="caching_hidden_states",
                        task=task,
                        model_name=model_name,
                        message="Caching hidden states for source-materiality split",
                        **kwargs,
                    ),
                )
            _sequence_hidden_state_cache(
                test_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="test",
                layers=cached_layers,
                progress_callback=lambda **kwargs: _emit_sae_progress(
                    output_dir,
                    phase="caching_hidden_states",
                    task=task,
                    model_name=model_name,
                    message="Caching hidden states for source-materiality split",
                    **kwargs,
                ),
            )
            gold_train = [label_list.index(str(row["label"])) for row in train_rows]
            gold_dev = [label_list.index(str(row["label"])) for row in dev_rows] if dev_rows else None
            gold_test = [label_list.index(str(row["label"])) for row in test_rows]
        elif task == TASK_SPAN_ROLE:
            label_list = list(SPAN_ROLE_LABELS)
            cached_layers = _token_hidden_state_cache(
                train_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="train",
                layers=hidden_state_layers,
                progress_callback=lambda **kwargs: _emit_sae_progress(
                    output_dir,
                    phase="caching_hidden_states",
                    task=task,
                    model_name=model_name,
                    message="Caching hidden states for span-role split",
                    **kwargs,
                ),
            )
            if dev_rows:
                _token_hidden_state_cache(
                    dev_rows,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    split_name="dev",
                    layers=cached_layers,
                    progress_callback=lambda **kwargs: _emit_sae_progress(
                        output_dir,
                        phase="caching_hidden_states",
                        task=task,
                        model_name=model_name,
                        message="Caching hidden states for span-role split",
                        **kwargs,
                    ),
                )
            _token_hidden_state_cache(
                test_rows,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                cache_dir=cache_dir,
                split_name="test",
                layers=cached_layers,
                progress_callback=lambda **kwargs: _emit_sae_progress(
                    output_dir,
                    phase="caching_hidden_states",
                    task=task,
                    model_name=model_name,
                    message="Caching hidden states for span-role split",
                    **kwargs,
                ),
            )
            gold_train = [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="train")]
            gold_dev = (
                [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="dev")]
                if dev_rows
                else None
            )
            gold_test = [label_list.index(label) for label in _load_cached_label_strings(cache_dir, split_name="test")]
        else:
            raise ValueError(f"Unsupported SAE task: {task!r}")

        layer_summaries: list[dict[str, Any]] = []
        best_summary: dict[str, Any] | None = None
        best_score = float("-inf")
        selection_key = "dev_f1_macro" if gold_dev is not None else "test_f1_macro"

        _emit_sae_progress(
            output_dir,
            phase="fitting_layers",
            task=task,
            model_name=model_name,
            message="Hidden-state caching complete; fitting per-layer SAE feature classifiers",
            selected_layers=list(selected_sae_layers),
            hidden_state_layers=list(hidden_state_layers),
            feature_cap=feature_cap,
        )
        total_layers = len(selected_sae_layers)
        for layer_position, sae_layer in enumerate(selected_sae_layers, start=1):
            hidden_state_index = release_spec.hidden_state_index_for_layer(sae_layer)
            _emit_sae_progress(
                output_dir,
                phase="fitting_layers",
                task=task,
                model_name=model_name,
                message=f"Loading SAE and fitting classifier for layer {sae_layer}",
                current_layer=sae_layer,
                hidden_state_index=hidden_state_index,
                layer_position=layer_position,
                total_layers=total_layers,
                feature_cap=feature_cap,
            )
            train_vectors = _load_cached_layer_vectors(
                cache_dir, split_name="train", layer_index=hidden_state_index
            )
            sae, sae_release, sae_id = _load_pretrained_sae(
                model_name=model_name,
                sae_layer=sae_layer,
                device=model_device,
            )
            selected_features = _select_sae_features(
                sae=sae,
                train_vectors=train_vectors,
                feature_cap=feature_cap,
                batch_size=batch_size,
            )
            train_features = _encode_selected_features(
                sae=sae,
                vectors=train_vectors,
                selected_features=selected_features,
                batch_size=batch_size,
            )
            clf = _create_logistic_regression(LogisticRegression)
            clf.fit(train_features, gold_train)

            test_features = _encode_selected_features(
                sae=sae,
                vectors=_load_cached_layer_vectors(cache_dir, split_name="test", layer_index=hidden_state_index),
                selected_features=selected_features,
                batch_size=batch_size,
            )
            test_pred = clf.predict(test_features).tolist()
            test_metrics = _compute_classification_metrics(
                gold_test,
                test_pred,
                labels=list(range(len(label_list))),
            )

            summary: dict[str, Any] = {
                "layer": sae_layer,
                "hidden_state_index": hidden_state_index,
                "task": task,
                "model_name": model_name,
                "sae_release": sae_release,
                "sae_id": sae_id,
                "selected_feature_count": len(selected_features),
                "feature_cap": feature_cap,
                "test_metrics": test_metrics,
                "test_f1_macro": test_metrics["f1_macro"],
            }
            if gold_dev is not None:
                dev_features = _encode_selected_features(
                    sae=sae,
                    vectors=_load_cached_layer_vectors(
                        cache_dir, split_name="dev", layer_index=hidden_state_index
                    ),
                    selected_features=selected_features,
                    batch_size=batch_size,
                )
                dev_pred = clf.predict(dev_features).tolist()
                dev_metrics = _compute_classification_metrics(
                    gold_dev,
                    dev_pred,
                    labels=list(range(len(label_list))),
                )
                summary["dev_metrics"] = dev_metrics
                summary["dev_f1_macro"] = dev_metrics["f1_macro"]

            (output_dir / f"sae_layer_{sae_layer:02d}.metrics.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
            (output_dir / f"sae_layer_{sae_layer:02d}.features.json").write_text(
                json.dumps(
                    {
                        "layer": sae_layer,
                        "hidden_state_index": hidden_state_index,
                        "sae_release": sae_release,
                        "sae_id": sae_id,
                        "selected_feature_count": len(selected_features),
                        "selected_features": selected_features,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            joblib.dump(clf, output_dir / f"sae_layer_{sae_layer:02d}.classifier.joblib")
            layer_summaries.append(summary)

            score = float(summary[selection_key])
            if score > best_score:
                best_score = score
                best_summary = summary

            _emit_sae_progress(
                output_dir,
                phase="fitting_layers",
                task=task,
                model_name=model_name,
                message=f"Completed SAE layer {sae_layer}",
                current_layer=sae_layer,
                hidden_state_index=hidden_state_index,
                layer_position=layer_position,
                total_layers=total_layers,
                current_score=score,
                selection_key=selection_key,
                best_layer=best_summary["layer"] if best_summary is not None else None,
                best_score=best_score if best_summary is not None else None,
            )

    if best_summary is None:
        raise RuntimeError("SAE experiment produced no layer summaries.")

    summary = {
        "task": task,
        "model_name": model_name,
        "sae_release": release_spec.release,
        "selection_key": selection_key,
        "best_layer": best_summary["layer"],
        "hidden_state_layers": list(hidden_state_layers),
        "sae_layers": list(selected_sae_layers),
        "feature_cap": feature_cap,
        "best_metrics": best_summary,
        "layer_rankings": sorted(
            layer_summaries,
            key=lambda row: float(row[selection_key]),
            reverse=True,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": task,
                "model_name": model_name,
                "sae_release": release_spec.release,
                "max_length": max_length,
                "sae_layers": list(selected_sae_layers),
                "hidden_state_layers": list(hidden_state_layers),
                "feature_cap": feature_cap,
                "batch_size": batch_size,
                "label_list": label_list,
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_run_reports(
        output_dir=output_dir,
        repo_root=Path(__file__).resolve().parents[3],
        title="SAE Experiment Run",
        payload=summary,
        extra={
            "input_dir": str(input_dir),
            "selection_key": selection_key,
            "max_length": max_length,
            "feature_cap": feature_cap,
            "batch_size": batch_size,
            "sae_release": release_spec.release,
        },
    )
    _emit_sae_progress(
        output_dir,
        phase="complete",
        task=task,
        model_name=model_name,
        message="SAE experiment complete",
        best_layer=summary["best_layer"],
        selection_key=selection_key,
        best_score=float(best_summary[selection_key]),
    )
    return summary
