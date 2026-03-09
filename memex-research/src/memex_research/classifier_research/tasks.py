from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

TaskName = Literal["span_role", "source_materiality"]
DatasetName = Literal["pe", "scicite"]

TASK_SPAN_ROLE: TaskName = "span_role"
TASK_SOURCE_MATERIALITY: TaskName = "source_materiality"

SPAN_ROLE_LABELS = ("CLAIM", "PREMISE", "EVIDENCE", "OTHER")
SOURCE_MATERIALITY_LABELS = ("SOURCE", "NOT_SOURCE")


class SpanRoleExample(TypedDict):
    id: str
    text: str
    tokens: list[str]
    labels: list[str]
    dataset: str
    task: str


class SourceMaterialityExample(TypedDict):
    id: str
    context: str
    label: str
    section_name: str
    source_type: str
    dataset: str
    task: str


@dataclass(frozen=True)
class TaskSpec:
    name: TaskName
    labels: tuple[str, ...]
    allowed_datasets: tuple[DatasetName, ...]
    problem_type: str

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.labels)}


TASK_SPECS: dict[TaskName, TaskSpec] = {
    TASK_SPAN_ROLE: TaskSpec(
        name=TASK_SPAN_ROLE,
        labels=SPAN_ROLE_LABELS,
        allowed_datasets=("pe",),
        problem_type="token_classification",
    ),
    TASK_SOURCE_MATERIALITY: TaskSpec(
        name=TASK_SOURCE_MATERIALITY,
        labels=SOURCE_MATERIALITY_LABELS,
        allowed_datasets=("scicite",),
        problem_type="sequence_classification",
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    if task not in TASK_SPECS:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported task {task!r}. Expected one of: {supported}")
    return TASK_SPECS[cast(TaskName, task)]


def validate_task_dataset(task: str, dataset: str) -> None:
    spec = get_task_spec(task)
    if dataset not in spec.allowed_datasets:
        allowed = ", ".join(spec.allowed_datasets)
        raise ValueError(f"Dataset {dataset!r} is not valid for task {task!r}. Expected one of: {allowed}")
