# Memex Sprint 1 Remote Run Bundle (2026-03-09)

This bundle packages the first real remote benchmark run of `memex-research` so Claude can review the state of the project and advise on next steps.

This is **not** a full artifact sync from the remote box. The trained checkpoints and transfer JSON files currently still live on the RunPod pod. The bundle below records the benchmark results, the transfer behavior we observed, the runtime issues we hit, the failed downlink/export attempt, and the concrete follow-up questions.

## Executive Summary

The practical classifier path is now real.

Benchmark results on SciCite:

- `microsoft/deberta-v3-base`
  - macro F1: `0.8698`
  - accuracy: `0.8700`
- `allenai/scibert_scivocab_uncased`
  - macro F1: `0.8738`
  - accuracy: `0.8743`

So:

- the source-materiality classifier path is viable
- SciBERT is slightly better than DeBERTa on the academic benchmark
- the main weak point is now **LessWrong transfer candidate generation**, not benchmark training

The first qualitative transfer pass on the fixed 10-post LessWrong seed set was **not** a fair end-to-end evaluation of source-materiality, because the current transfer adapter only proposes candidates from:

- numbered bibliography entries in the post text
- `external_links` in the raw post record

That produced:

- `posts_with_candidates = 6 / 10`
- `total_candidates = 69`

Per-post candidate counts:

- `0`  `6hfGNLf4Hg5DXqJCF`  `A Fable of Science and Politics`
- `6`  `7ZqGiPHTpiDMwqMN2`  `Twelve Virtues of Rationality`
- `4`  `GrDqnMjhqoxiqpQPw`  `The Proper Use of Humility`
- `6`  `NKECtGX4RZPd7SqYp`  `The Modesty Argument`
- `0`  `Pm83rA8MTYYeR4Ci4`  `"I don't know."`
- `7`  `XTXWPQSEgoMkAupKt`  `An Intuitive Explanation of Bayes's Theorem`
- `0`  `YshRbqZHYFoEMqFAu`  `Why Truth?`
- `45` `afmj8TKAqH6F2QMfZ`  `A Technical Explanation of Technical Explanation`
- `0`  `jnZbHi873v9vcpGpZ`  `What's a Bias?`
- `1`  `teaxCFgtmCQ3E9fy8`  `The Martial Art of Rationality`

This is enough to conclude that the current transfer adapter is too weak and too uneven to judge transfer quality yet.

## What Actually Ran

Remote environment:

- provider: RunPod pod
- GPU: `NVIDIA A100 80GB PCIe`
- runtime: Ubuntu container with Python 3.11

Project root on the pod:

- `/workspace/memex-research/memex-research/memex-research`

Benchmark prepared:

- SciCite official `train/dev/test` splits

Executed benchmark runs:

1. `SciCite + DeBERTa-v3-base`
2. `SciCite + SciBERT`

Executed qualitative transfer:

1. `SciCite + DeBERTa-v3-base` transfer-only pass on the fixed 10-post LessWrong seed set

Attempted but blocked:

1. `SciCite + Llama-3.1-8B` probe run
   - failed because the model is gated on Hugging Face
2. `SciCite + Qwen/Qwen2.5-7B` probe run
   - failed because the current code only allows `meta-llama/Llama-3.1-8B` as a probe model

## Benchmark Results

### 1. SciCite + DeBERTa-v3-base

Recorded `metrics.json` content:

```json
{
  "counts": {
    "eval": 916,
    "test": 1861,
    "train": 8243
  },
  "dataset": "scicite",
  "model_name": "microsoft/deberta-v3-base",
  "task": "source_materiality",
  "test_metrics": {
    "accuracy": 0.8699623858140785,
    "f1_macro": 0.8698225228350098,
    "precision_macro": 0.8701865326865328,
    "recall_macro": 0.8720739765593075
  }
}
```

Training behavior:

- completed successfully
- training time about `398.7s` (`~6m39s`)
- no VRAM pressure issues
- compute throughput, not memory, was the limiting factor

### 2. SciCite + SciBERT

Recorded `metrics.json` content:

```json
{
  "counts": {
    "eval": 916,
    "test": 1861,
    "train": 8243
  },
  "dataset": "scicite",
  "model_name": "allenai/scibert_scivocab_uncased",
  "task": "source_materiality",
  "test_metrics": {
    "accuracy": 0.8742611499193982,
    "f1_macro": 0.8738283151380293,
    "precision_macro": 0.8734080303324907,
    "recall_macro": 0.8745420288643708
  }
}
```

Conclusion:

- SciBERT is slightly better than DeBERTa on the academic citation benchmark
- the gain is modest but real

## Transfer Findings

The first transfer pass used the benchmark-trained DeBERTa classifier plus the current transfer candidate adapter.

Example transfer file:

- post: `6hfGNLf4Hg5DXqJCF`
- title: `A Fable of Science and Politics`
- result:

```json
{
  "candidates": [],
  "post_id": "6hfGNLf4Hg5DXqJCF",
  "slug": "a-fable-of-science-and-politics",
  "title": "A Fable of Science and Politics"
}
```

Interpretation:

- the classifier is not the immediate issue here
- the transfer adapter is failing to surface source candidates for many posts
- on at least one post it surfaced far too many (`45`) candidates

That means:

- benchmark training worked
- transfer candidate generation is currently the bottleneck
- we should not over-interpret the transfer results yet

## Remote Runtime Issues Encountered

### 1. Absolute local paths in tests and transfer manifest

The repo initially assumed local Mac paths like:

- `/Users/tatetower/Codex/memex-research/...`

This broke on the remote box.

We fixed:

- transfer manifest loading to be repo-relative
- test script paths to be repo-relative
- the transfer manifest JSON payload to point to the copied local seed paths

### 2. `transformers` API mismatch

The trainer code initially used `evaluation_strategy=...`.

On the remote environment, the installed `transformers` expected:

- `eval_strategy=...`

This was patched so the run could proceed.

### 3. Missing runtime dependencies

The remote box exposed missing dependencies not yet included in the original `research` extra:

- `protobuf`
- `accelerate`

These are now added locally to the standalone repo’s `research` extra.

### 4. Transfer pass defaulted to CPU

The benchmark training used GPU, but the post-training transfer pass reloaded the model on CPU because the transfer helper never moved the model to CUDA.

This caused:

- `100%` CPU usage
- no GPU usage
- an apparently “hung” post-training phase

This was patched locally. The current remote transfer run completed after applying that fix.

### 5. Probe model access failure

The first probe attempt failed for external reasons:

- `meta-llama/Llama-3.1-8B` is gated on Hugging Face
- the remote box was not authenticated for gated access

The second probe attempt failed for internal reasons:

- the code hardcoded the probe model allowlist to only `meta-llama/Llama-3.1-8B`

So we do **not** have a first probe result yet.

## Downlink / Artifact Export Status

The first remote downlink attempt should be treated as failed.

- I tried to copy a tarball of the run tree back from the pod.
- The archive grew into the multi-GB range and was aborted.
- That strongly suggests we were packaging too much, most likely full trainer/checkpoint state rather than only the benchmark artifacts we actually need.

So at the moment:

- the benchmark numbers above are trustworthy because they were captured from successful remote terminal output
- the previous downloaded artifact payload is **not** trustworthy and should be ignored

This is now fixed locally with explicit artifact tooling:

- inventory helper:
  - [`report_run_inventory.py`](/Users/tatetower/Codex/memex-research/scripts/report_run_inventory.py)
- packaging helper:
  - [`package_run_artifacts.py`](/Users/tatetower/Codex/memex-research/scripts/package_run_artifacts.py)
- implementation:
  - [`artifacts.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/artifacts.py)

Current export design:

- produce `inventory.json` and `inventory.md` first
- package only selected runs
- include:
  - `metrics.json`
  - `run_config.json`
  - `diagnostics.json`
  - `report.md`
  - `predictions.jsonl`
  - `model/`
  - `transfer/` if present
- exclude `trainer/` by default

That is the right remote export shape for the next pod session.

## Better Workflow Added Locally

To avoid another session of hand-pasted long shell blocks, the local standalone repo now has:

- [`Makefile`](/Users/tatetower/Codex/memex-research/Makefile)
- [`scripts/bootstrap_remote.sh`](/Users/tatetower/Codex/memex-research/scripts/bootstrap_remote.sh)
- [`scripts/run_scicite_source_experiment.py`](/Users/tatetower/Codex/memex-research/scripts/run_scicite_source_experiment.py)
- [`scripts/report_run_inventory.py`](/Users/tatetower/Codex/memex-research/scripts/report_run_inventory.py)
- [`scripts/package_run_artifacts.py`](/Users/tatetower/Codex/memex-research/scripts/package_run_artifacts.py)

Intended next-session workflow:

```bash
make bootstrap
make scicite-deberta
make scicite-scibert
make run-inventory
make package-scicite-runs
make scicite-deberta-transfer
```

This is implemented locally, but not yet synced back to the remote box.

There are also local repo fixes since the remote run:

- public probe models are now allowed:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `mistralai/Mistral-7B-v0.3`
- transfer is benchmark-first / opt-in
- every run now writes:
  - `diagnostics.json`
  - `report.md`
- transfer summaries ignore metadata files on reruns

## Current State Of The Decision

What is now justified:

- the source-materiality classifier path is real
- benchmark-based fine-tuning is working
- SciBERT is the strongest current source-materiality baseline

What is still unresolved:

- whether the benchmark-trained classifier transfers plausibly to LessWrong once candidate generation is reasonable
- whether the probe path can get close enough to the classifier baseline to justify deeper SAE work
- whether the new export/package path is sufficient, or whether `model/` also needs to be separated from metrics/report artifacts

## Concrete Next Questions For Claude

1. Is the current conclusion right that **candidate generation**, not classification, is the main transfer bottleneck after the SciCite runs?
2. Should the next transfer iteration use:
   - richer deterministic LessWrong-specific candidates, or
   - the existing provenance extractor’s source candidates directly?
3. Should the first probe retry use a public model like `Qwen/Qwen2.5-7B`, and if so, what is the best public probe model to compare against SciBERT?
4. Is the `SciBERT > DeBERTa` delta on SciCite meaningful enough to change the primary baseline choice, or is DeBERTa still the better cross-domain default?
5. Given the uneven 10-post transfer candidate counts, what is the cleanest next evaluation design before adding any in-domain labels?
6. Is there a better first probe task than source-materiality, or is SciCite still the right initial probe benchmark?
7. Is the new export/package design the right granularity for remote runs, or should `model/` be exported separately from metrics/report artifacts?

## Related Local Files

Standalone project:

- [`README.md`](/Users/tatetower/Codex/memex-research/README.md)
- [`scripts/README.md`](/Users/tatetower/Codex/memex-research/scripts/README.md)
- [`pyproject.toml`](/Users/tatetower/Codex/memex-research/pyproject.toml)

Classifier / transfer code:

- [`train_source.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/train_source.py)
- [`transfer_eval.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/transfer_eval.py)
- [`hf_models.py`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/hf_models.py)

New remote-friendly entrypoints:

- [`Makefile`](/Users/tatetower/Codex/memex-research/Makefile)
- [`bootstrap_remote.sh`](/Users/tatetower/Codex/memex-research/scripts/bootstrap_remote.sh)
- [`run_scicite_source_experiment.py`](/Users/tatetower/Codex/memex-research/scripts/run_scicite_source_experiment.py)
- [`report_run_inventory.py`](/Users/tatetower/Codex/memex-research/scripts/report_run_inventory.py)
- [`package_run_artifacts.py`](/Users/tatetower/Codex/memex-research/scripts/package_run_artifacts.py)
