# Research Sprint 1 Runbook

This directory contains the execution surface for the Sprint 1 Memex pilot:

- benchmark prep for:
  - Persuasive Essays / UKP -> `span_role`
  - SciCite -> `source_materiality`
- classifier training for:
  - `microsoft/deberta-v3-base`
  - `allenai/scibert_scivocab_uncased`
- residual-stream probe experiments for:
  - public 7B-ish open models such as:
    - `Qwen/Qwen2.5-7B-Instruct`
    - `mistralai/Mistral-7B-v0.3`

The implementation code lives under:

- [`src/memex_research/classifier_research`](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research)

The fixed 10-post qualitative transfer set lives under:

- [`analysis/memex_transfer_seed_10/manifest.json`](/Users/tatetower/Codex/memex-research/analysis/memex_transfer_seed_10/manifest.json)

Run artifact helpers live under:

- [`scripts/report_run_inventory.py`](/Users/tatetower/Codex/memex-research/scripts/report_run_inventory.py)
- [`scripts/package_run_artifacts.py`](/Users/tatetower/Codex/memex-research/scripts/package_run_artifacts.py)

## Best Way To Run This

Use a normal remote GPU shell with the repo cloned locally. Colab works, but the better workflow is:

```bash
make bootstrap
make scicite-deberta
make scicite-scibert
make scicite-deberta-transfer
make scicite-probe-qwen
```

If you do not want `make`, the single-command equivalents are:

```bash
python scripts/run_scicite_source_experiment.py --model-name microsoft/deberta-v3-base
python scripts/run_scicite_source_experiment.py --model-name allenai/scibert_scivocab_uncased
python scripts/run_scicite_source_experiment.py --model-name microsoft/deberta-v3-base --run-transfer --transfer-only --skip-quality-checks
```

Recommended split:

- classifier training first
- transfer pass second
- probe experiments only after classifier baselines look good

Do not start with the probe path.

## Environment Setup

From the repo root:

```bash
pip install -e ".[dev,research]"
```

On a fresh remote box, the intended bootstrap is:

```bash
bash scripts/bootstrap_remote.sh
```

That script installs:

- `git`
- `python3-venv`
- `tmux`
- the local editable package with `.[dev,research]`
- and upgrades `pip`, `setuptools`, and `wheel` before dependency install

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
- `diagnostics.json`
- `report.md`
- `predictions.jsonl`
- `model/`

Transfer is opt-in. Run it separately after you confirm the benchmark result is worth inspecting on LessWrong.

## Diagnostics And Packaging

Every benchmark or probe run writes:

- `run_config.json`
- `diagnostics.json`
- `report.md`

Classifier runs also write:

- `metrics.json`
- `predictions.jsonl`
- `model/`

Probe runs also write:

- `summary.json`
- `layer_XX.metrics.json`
- `layer_XX.probe.joblib`

To inventory a run tree:

```bash
python scripts/report_run_inventory.py \
  --run-root /content/memex-research/analysis/memex_runs \
  --output-dir /content/memex-research/analysis/run_inventory
```

To package only selected benchmark runs without the large `trainer/` checkpoints:

```bash
python scripts/package_run_artifacts.py \
  --run-root /content/memex-research/analysis/memex_runs \
  --output /content/memex-research/analysis/packaged_runs/scicite_runs.tar.gz \
  --run-name scicite_deberta \
  --run-name scicite_scibert
```

To export only the trained model directories for those runs:

```bash
python scripts/package_run_artifacts.py \
  --run-root /content/memex-research/analysis/memex_runs \
  --output /content/memex-research/analysis/packaged_runs/scicite_models.tar.gz \
  --run-name scicite_deberta \
  --run-name scicite_scibert \
  --exclude-transfer \
  --exclude-predictions \
  --exclude-reports
```

Packaging defaults:

- includes `metrics.json`, `run_config.json`, `diagnostics.json`, `report.md`
- includes `model/`, `predictions.jsonl`, and `transfer/` if present
- excludes `trainer/` by default to keep archives small

Only pass `--include-trainer` if you explicitly want full trainer checkpoints.

## Step 3: Run Probe Experiments

Do this only after the classifier baselines are finished.

### PE + public probe run

```bash
python scripts/run_probe_experiment.py \
  --task span_role \
  --dataset pe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_probe_qwen25_7b_instruct
```

### SciCite + public probe run

```bash
python scripts/run_probe_experiment.py \
  --task source_materiality \
  --dataset scicite \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_probe_qwen25_7b_instruct
```

Each probe run writes:

- `summary.json`
- `run_config.json`
- `diagnostics.json`
- `report.md`
- `layer_XX.metrics.json`
- `layer_XX.probe.joblib`

Transfer is opt-in:

```bash
python scripts/run_probe_experiment.py \
  --task source_materiality \
  --dataset scicite \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_probe_qwen25_7b_instruct \
  --run-transfer
```

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

1. `make bootstrap`
2. `make scicite-deberta`
3. inspect `analysis/memex_runs/scicite_deberta/metrics.json`
4. `make scicite-scibert`
5. inspect `analysis/memex_runs/scicite_scibert/metrics.json`
6. `make scicite-deberta-transfer`
7. inspect `analysis/memex_runs/scicite_deberta/transfer/`
8. only then run the probe experiments

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
