# Research Sprint 1 Runbook

This directory contains the execution surface for the Sprint 1 Memex pilot:

- benchmark prep for:
  - Persuasive Essays / UKP -> `span_role`
  - SciCite -> `source_materiality`
- classifier training for:
  - `microsoft/deberta-v3-base`
  - `allenai/scibert_scivocab_uncased`
- residual-stream probe experiments for:
  - `meta-llama/Llama-3.1-8B`

The implementation code lives under:

- [`src/lesswrong_provenance/experimental/classifier_research`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research)

The fixed 10-post qualitative transfer set lives under:

- [`analysis/memex_transfer_seed_10/manifest.json`](/Users/tatetower/Codex/memex-research/analysis/memex_transfer_seed_10/manifest.json)

## Best Way To Run This

Use **Google Colab with GPU**.

Recommended split:

- dataset prep: CPU or any small GPU runtime
- DeBERTa / SciBERT training: `T4` is enough, `L4` is better
- Llama hidden-state probe path: `L4` or `A100` strongly preferred

Reason:

- the classifier path is standard fine-tuning and is cheap
- the probe path has a much worse memory profile because it needs hidden states

Do not start with the probe path. Run the classifiers first.

## Environment Setup

From the repo root:

```bash
pip install -e ".[dev,research]"
```

If you are using Colab, the simplest pattern is:

1. clone the repo
2. `cd /content/memex-research`
3. install `.[dev,research]`
4. mount Drive only if you want persistent output artifacts

## Expected Input Layout

### Persuasive Essays / UKP

This script expects a root directory containing `.txt` / `.ann` files either:

- under benchmark-native split directories:
  - `train/`
  - `dev/` or `validation/`
  - `test/`
- or via explicit split manifests

Example:

```text
data/pe_raw/
  train/
    essay001.txt
    essay001.ann
  test/
    essay900.txt
    essay900.ann
```

### SciCite

This script expects JSONL split files from SciCite, typically:

```text
data/scicite_raw/
  train.jsonl
  dev.jsonl
  test.jsonl
```

## Step 1: Prepare Benchmark Artifacts

### Persuasive Essays / UKP -> span role JSONL

```bash
python scripts/prepare_pe_span_benchmark.py \
  --input-root /content/data/pe_raw \
  --output-dir /content/data/pe_span_benchmark
```

Output:

- `/content/data/pe_span_benchmark/train.jsonl`
- `/content/data/pe_span_benchmark/dev.jsonl` if available
- `/content/data/pe_span_benchmark/test.jsonl`
- `/content/data/pe_span_benchmark/summary.json`

### SciCite -> source materiality JSONL

Use the full split first. Do not subsample unless you are debugging.

```bash
python scripts/prepare_scicite_material_use_benchmark.py \
  --train-input /content/data/scicite_raw/train.jsonl \
  --dev-input /content/data/scicite_raw/dev.jsonl \
  --test-input /content/data/scicite_raw/test.jsonl \
  --output-dir /content/data/scicite_material_benchmark \
  --per-label 0
```

Output:

- `/content/data/scicite_material_benchmark/train.jsonl`
- `/content/data/scicite_material_benchmark/dev.jsonl`
- `/content/data/scicite_material_benchmark/test.jsonl`
- `/content/data/scicite_material_benchmark/summary.json`

## Step 2: Train Classifier Baselines

Write runs under a single root like:

- `/content/memex-research/analysis/memex_runs/`

### PE + DeBERTa span-role baseline

```bash
python scripts/run_classifier_train.py \
  --task span_role \
  --dataset pe \
  --model-name microsoft/deberta-v3-base \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_deberta
```

### SciCite + DeBERTa source-materiality baseline

```bash
python scripts/run_classifier_train.py \
  --task source_materiality \
  --dataset scicite \
  --model-name microsoft/deberta-v3-base \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_deberta
```

### SciCite + SciBERT comparison run

```bash
python scripts/run_classifier_train.py \
  --task source_materiality \
  --dataset scicite \
  --model-name allenai/scibert_scivocab_uncased \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_scibert
```

Each classifier run writes:

- `metrics.json`
- `run_config.json`
- `predictions.jsonl`
- `model/`
- `transfer/`

## Step 3: Run Probe Experiments

Do this only after the classifier baselines are finished.

### PE + Llama probe run

```bash
python scripts/run_probe_experiment.py \
  --task span_role \
  --dataset pe \
  --model-name meta-llama/Llama-3.1-8B \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_probe_llama31_8b
```

### SciCite + Llama probe run

```bash
python scripts/run_probe_experiment.py \
  --task source_materiality \
  --dataset scicite \
  --model-name meta-llama/Llama-3.1-8B \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_probe_llama31_8b
```

Each probe run writes:

- `summary.json`
- `run_config.json`
- `layer_XX.metrics.json`
- `layer_XX.probe.joblib`
- `transfer/`

If Llama 3.1 8B does not fit in Colab memory, downshift to the smallest same-family model that works and record that change in the results bundle.

## How To Read The Outputs

Classifier path:

- `metrics.json` is the benchmark score surface
- `transfer/` is the qualitative LessWrong sanity check

Probe path:

- `summary.json` contains the best layer ranking
- compare the best-layer F1 to the matching DeBERTa benchmark F1

Sprint 1 probe gate:

- continue probe work only if the best probe layer reaches at least `90%` of the best DeBERTa F1 on the same benchmark task

## Recommended Execution Order

1. prepare PE
2. prepare SciCite
3. run PE + DeBERTa
4. run SciCite + DeBERTa
5. run SciCite + SciBERT
6. inspect transfer outputs from all three classifier runs
7. only then run the probe experiments

This is the right order because the classifier baselines are the mainline path. The probe path is a research gate, not the foundation.

## Local Validation

The local scaffold is already validated:

- `mypy`: clean
- `ruff check src tests scripts`: clean
- `pytest -q`: green

That means if a Colab run fails, it is much more likely to be:

- dataset layout
- missing auth for model download
- insufficient GPU memory
- missing research dependencies

not a broken local code path.
