from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_REGISTRY = {
    "sequence-classification": {
        "microsoft/deberta-v3-base",
        "allenai/scibert_scivocab_uncased",
    },
    "token-classification": {
        "microsoft/deberta-v3-base",
    },
    "probe": {
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-v0.3",
    },
}
SUPPORTED_SEQUENCE_MODELS = MODEL_REGISTRY["sequence-classification"]
SUPPORTED_TOKEN_MODELS = MODEL_REGISTRY["token-classification"]
SUPPORTED_PROBE_MODELS = MODEL_REGISTRY["probe"]


@dataclass(frozen=True)
class SAEReleaseSpec:
    model_name: str
    release: str
    repo_id: str
    available_layers: tuple[int, ...]
    hidden_state_offset: int
    trainer_candidates: tuple[int, ...]

    def hidden_state_index_for_layer(self, sae_layer: int) -> int:
        return sae_layer + self.hidden_state_offset

    def sae_id_candidates_for_layer(self, sae_layer: int) -> tuple[str, ...]:
        return tuple(
            f"resid_post_layer_{sae_layer}/trainer_{trainer}"
            for trainer in self.trainer_candidates
        )

    def sae_id_for_layer(self, sae_layer: int) -> str:
        return self.sae_id_candidates_for_layer(sae_layer)[0]


SAE_RELEASES: dict[str, SAEReleaseSpec] = {
    "Qwen/Qwen2.5-7B-Instruct": SAEReleaseSpec(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        release="qwen2.5-7b-instruct-andyrdt",
        repo_id="andyrdt/saes-qwen2.5-7b-instruct",
        available_layers=(3, 7, 11, 15, 19, 23, 27),
        hidden_state_offset=1,
        trainer_candidates=(1, 0, 2, 3),
    ),
}
SUPPORTED_SAE_MODELS = set(SAE_RELEASES)


def _require_ml_stack() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency is optional locally
        raise RuntimeError(
            "ML research dependencies are not installed. Install the research extra with "
            "`pip install -e .[research]` in Colab or a local ML environment."
        ) from exc
    return torch, transformers


def _validate_model_name(model_name: str, *, allowed: set[str], purpose: str) -> None:
    if model_name not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {purpose} model {model_name!r}. Expected one of: {supported}")


def get_tokenizer(model_name: str) -> Any:
    _torch, transformers = _require_ml_stack()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def create_sequence_classification_components(
    *,
    model_name: str,
    labels: tuple[str, ...],
) -> tuple[Any, Any]:
    _validate_model_name(model_name, allowed=SUPPORTED_SEQUENCE_MODELS, purpose="sequence-classification")
    _torch, transformers = _require_ml_stack()
    tokenizer = get_tokenizer(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
    )
    return tokenizer, model


def create_token_classification_components(
    *,
    model_name: str,
    labels: tuple[str, ...],
) -> tuple[Any, Any]:
    _validate_model_name(model_name, allowed=SUPPORTED_TOKEN_MODELS, purpose="token-classification")
    _torch, transformers = _require_ml_stack()
    tokenizer = get_tokenizer(model_name)
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
    )
    return tokenizer, model


def create_probe_components(model_name: str) -> tuple[Any, Any]:
    _validate_model_name(model_name, allowed=SUPPORTED_PROBE_MODELS, purpose="probe")
    _torch, transformers = _require_ml_stack()
    tokenizer = get_tokenizer(model_name)
    model = transformers.AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model


def get_sae_release_spec(model_name: str) -> SAEReleaseSpec:
    _validate_model_name(model_name, allowed=SUPPORTED_SAE_MODELS, purpose="sae")
    return SAE_RELEASES[model_name]
