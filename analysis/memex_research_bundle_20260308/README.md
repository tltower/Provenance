# Memex Research Bundle (2026-03-08)

This bundle is the Sprint 1 review surface for the Memex classifier/probe pilot.

Current status:

- implementation scaffold complete
- benchmark adapters complete for:
  - Persuasive Essays / UKP -> `span_role`
  - SciCite -> `source_materiality`
- training and probe runners complete
- transfer-set manifest complete
- benchmark runs not yet executed in this workspace

This bundle is intentionally split into two phases:

1. implementation state
2. benchmark and transfer results after Colab runs

## Implementation State

Core research package:

- [`src/lesswrong_provenance/experimental/classifier_research/tasks.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/tasks.py)
- [`src/lesswrong_provenance/experimental/classifier_research/datasets.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/datasets.py)
- [`src/lesswrong_provenance/experimental/classifier_research/splits.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/splits.py)
- [`src/lesswrong_provenance/experimental/classifier_research/hf_models.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/hf_models.py)
- [`src/lesswrong_provenance/experimental/classifier_research/train_span.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/train_span.py)
- [`src/lesswrong_provenance/experimental/classifier_research/train_source.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/train_source.py)
- [`src/lesswrong_provenance/experimental/classifier_research/probes.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/probes.py)
- [`src/lesswrong_provenance/experimental/classifier_research/transfer_eval.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/transfer_eval.py)

Benchmark prep / execution scripts:

- [`scripts/prepare_pe_span_benchmark.py`](/Users/tatetower/Codex/memex-research/scripts/prepare_pe_span_benchmark.py)
- [`scripts/prepare_scicite_material_use_benchmark.py`](/Users/tatetower/Codex/memex-research/scripts/prepare_scicite_material_use_benchmark.py)
- [`scripts/run_classifier_train.py`](/Users/tatetower/Codex/memex-research/scripts/run_classifier_train.py)
- [`scripts/run_probe_experiment.py`](/Users/tatetower/Codex/memex-research/scripts/run_probe_experiment.py)

Transfer-set definition:

- [`analysis/memex_transfer_seed_10/manifest.json`](/Users/tatetower/Codex/memex-research/analysis/memex_transfer_seed_10/manifest.json)

Validation completed locally:

- `mypy`: `Success: no issues found in 16 source files`
- `ruff check src tests scripts`: passed
- `pytest -q`: `10 passed in 0.06s`

## Expected Sprint 1 Runs

Classifier runs:

- PE + `microsoft/deberta-v3-base`
- SciCite + `microsoft/deberta-v3-base`
- SciCite + `allenai/scibert_scivocab_uncased`

Probe runs:

- PE + `meta-llama/Llama-3.1-8B`
- SciCite + `meta-llama/Llama-3.1-8B`

Each run should write:

- `metrics.json` or `summary.json`
- `run_config.json`
- prediction artifacts
- transfer outputs under `transfer/`

## Output Layout

Suggested output roots:

- `analysis/memex_runs/pe_deberta/`
- `analysis/memex_runs/scicite_deberta/`
- `analysis/memex_runs/scicite_scibert/`
- `analysis/memex_runs/pe_probe_llama31_8b/`
- `analysis/memex_runs/scicite_probe_llama31_8b/`

## Sprint 1 Decision Gate

Probe continuation gate:

- compare the best probe F1 to the best DeBERTa F1 on the same benchmark task
- continue probe/SAE work only if best probe F1 is at least `90%` of the matching DeBERTa F1

Transfer gate:

- if both benchmark-trained classifiers look obviously unusable on the fixed 10-post LessWrong transfer set, do not expand the benchmark set yet
- pivot to a small in-domain labeling plan instead

## Pending Review Questions

- Does PE-trained span detection transfer plausibly enough to LessWrong prose to justify adding more argument datasets?
- Does SciCite-trained source materiality transfer plausibly enough to LessWrong citation behavior to justify a source classifier before in-domain labels?
- Are probes close enough to the practical baselines to justify later SAE work?

## Notes

This bundle is intentionally not claiming benchmark results yet. It is the implementation and review scaffold for the first Colab training/probe round.
