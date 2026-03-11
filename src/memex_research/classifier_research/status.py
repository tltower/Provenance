from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _noop_callback_event(*args: Any, **kwargs: Any) -> None:
    return None


class EventCompatibleCallback:
    """Provide no-op handlers for Trainer callback events we do not override."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return _noop_callback_event
        raise AttributeError(name)


class HFTrainerHeartbeatCallback(EventCompatibleCallback):
    def __init__(
        self,
        *,
        output_dir: Path,
        task: str,
        dataset_name: str,
        model_name: str,
        split_counts: dict[str, int],
        prefix: str = "train",
        status_filename: str = "train_status.json",
        events_filename: str = "train_events.jsonl",
        step_interval: int | None = None,
        prediction_interval: int = 10,
    ) -> None:
        self.output_dir = output_dir
        self.task = task
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.split_counts = split_counts
        self.prefix = prefix
        self.status_filename = status_filename
        self.events_filename = events_filename
        self._configured_step_interval = step_interval
        self._active_step_interval = step_interval or 1
        self.prediction_interval = max(1, prediction_interval)
        self.prediction_steps_seen = 0
        self.prediction_stage = "eval"

    def _emit(self, *, phase: str, message: str, **extra: Any) -> None:
        emit_run_status(
            self.output_dir,
            filename=self.status_filename,
            events_filename=self.events_filename,
            prefix=self.prefix,
            phase=phase,
            message=message,
            task=self.task,
            dataset=self.dataset_name,
            model_name=self.model_name,
            split_counts=self.split_counts,
            **extra,
        )

    def begin_prediction_stage(self, stage_name: str, *, total_examples: int | None = None) -> None:
        self.prediction_stage = stage_name
        self.prediction_steps_seen = 0
        self._emit(
            phase=f"{stage_name}_begin",
            message=f"Starting {stage_name} loop",
            total_examples=total_examples,
        )

    def _prediction_step_phase(self) -> str:
        return "predict_step" if self.prediction_stage == "predict" else "eval_step"

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        max_steps = max(1, int(state.max_steps))
        self._active_step_interval = self._configured_step_interval or max(1, max_steps // 10)
        self._emit(
            phase="train_begin",
            message="Starting trainer optimization",
            total_epochs=float(args.num_train_epochs),
            max_steps=max_steps,
            step_interval=self._active_step_interval,
            prediction_interval=self.prediction_interval,
        )

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._emit(
            phase="epoch_begin",
            message=f"Starting epoch {int(state.epoch or 0) + 1}",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
        )

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        max_steps = max(1, int(state.max_steps))
        global_step = int(state.global_step)
        if (
            global_step == 1
            or global_step == max_steps
            or global_step % self._active_step_interval == 0
        ):
            self._emit(
                phase="train_step",
                message="Completed trainer step",
                epoch=float(state.epoch or 0.0),
                global_step=global_step,
                max_steps=max_steps,
                train_progress=float(global_step / max_steps),
            )

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._emit(
            phase="epoch_end",
            message=f"Completed epoch {int(round(state.epoch or 0.0))}",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        self._emit(
            phase="train_log",
            message="Trainer emitted metrics",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
            logs=logs,
        )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._emit(
            phase="checkpoint_saved",
            message="Trainer saved a checkpoint",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
        )

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.prediction_steps_seen += 1
        if self.prediction_steps_seen == 1 or self.prediction_steps_seen % self.prediction_interval == 0:
            self._emit(
                phase=self._prediction_step_phase(),
                message=f"Processed {self.prediction_stage} batch",
                epoch=float(state.epoch or 0.0),
                global_step=int(state.global_step),
                prediction_steps_seen=self.prediction_steps_seen,
                prediction_stage=self.prediction_stage,
            )

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            phase="eval_complete",
            message="Completed evaluation pass",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
            prediction_steps_seen=self.prediction_steps_seen,
            metrics=metrics or {},
        )
        self.prediction_steps_seen = 0
        self.prediction_stage = "eval"

    def on_predict(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            phase="predict_complete",
            message="Completed prediction loop",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
            prediction_steps_seen=self.prediction_steps_seen,
            metrics=metrics or {},
        )
        self.prediction_steps_seen = 0
        self.prediction_stage = "eval"

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._emit(
            phase="train_end",
            message="Trainer finished optimization",
            epoch=float(state.epoch or 0.0),
            global_step=int(state.global_step),
        )


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_events_filename(filename: str) -> str:
    suffix = "".join(Path(filename).suffixes)
    stem = filename[: -len(suffix)] if suffix else filename
    if stem.endswith("_status"):
        stem = stem[: -len("_status")]
    return f"{stem}_events.jsonl"


def append_run_event(
    output_dir: Path,
    *,
    filename: str,
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_run_analytics(
    output_dir: Path,
    *,
    filename: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    enriched = {
        "updated_at_utc": _timestamp_utc(),
        **payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(enriched, indent=2, sort_keys=True))
    return enriched


def emit_run_status(
    output_dir: Path,
    *,
    filename: str,
    prefix: str,
    phase: str,
    message: str,
    events_filename: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "updated_at_utc": _timestamp_utc(),
        "phase": phase,
        "message": message,
    }
    if extra:
        payload.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True))
    append_run_event(
        output_dir,
        filename=events_filename or _default_events_filename(filename),
        payload=payload,
    )
    print(f"[{prefix}:{phase}] {message}", flush=True)
    return payload
