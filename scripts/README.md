# Research Sprint 1 Runbook

This directory contains the execution surface for the Sprint 1 Memex pilot:

- benchmark prep for:
  - CDCP -> `span_role`
  - Persuasive Essays / UKP -> `span_role` auxiliary baseline
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
make cdcp-prepare
make cdcp-deberta
make cdcp-probe-qwen
make cdcp-sae-qwen
make scicite-deberta
make scicite-scibert
make scicite-probe-qwen
make scicite-sae-qwen
```

If you do not want `make`, the single-command equivalents are:

```bash
python scripts/run_scicite_source_experiment.py --model-name microsoft/deberta-v3-base
python scripts/run_scicite_source_experiment.py --model-name allenai/scibert_scivocab_uncased
python scripts/run_scicite_source_experiment.py --model-name microsoft/deberta-v3-base --run-transfer --transfer-only --skip-quality-checks
```

Recommended split:

- classifier training first
- candidate-only transfer second
- heuristic transfer only after that if you want to debug candidate generation
- probe experiments only after classifier baselines look good
- pretrained SAE experiments only after the corresponding probe clears the continuation gate

Do not start with the probe path.
Do not start with SAE training or SAE reuse until the linear probe path is already credible.

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

### CDCP

This script downloads the public Hugging Face mirror and writes normalized
JSONL split files:

```bash
python scripts/prepare_cdcp_span_benchmark.py \
  --dataset-name DFKI-SLT/cdcp \
  --output-dir /content/data/cdcp_span_benchmark
```

Output:

- `/content/data/cdcp_span_benchmark/train.jsonl`
- `/content/data/cdcp_span_benchmark/dev.jsonl`
- `/content/data/cdcp_span_benchmark/test.jsonl`
- `/content/data/cdcp_span_benchmark/summary.json`

### Persuasive Essays / UKP

This remains useful as an auxiliary claim/premise baseline. It expects a
root directory containing `.txt` / `.ann` files either:

- under benchmark-native split directories:
  - `train/`
  - `dev/` or `validation/`
  - `test/`
- or via explicit split manifests

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

### CDCP + DeBERTa span-role baseline

```bash
python scripts/run_classifier_train.py \
  --task span_role \
  --dataset cdcp \
  --model-name microsoft/deberta-v3-base \
  --input-dir /content/data/cdcp_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/cdcp_deberta
```

### CDCP + Qwen probe run

```bash
python scripts/run_probe_experiment.py \
  --task span_role \
  --dataset cdcp \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/cdcp_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/cdcp_probe_qwen25_7b_instruct
```

### CDCP + Qwen pretrained-SAE run

```bash
python scripts/run_sae_experiment.py \
  --task span_role \
  --dataset cdcp \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/cdcp_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/cdcp_sae_qwen25_7b_instruct
```

### PE + DeBERTa span-role baseline

```bash
python scripts/run_classifier_train.py \
  --task span_role \
  --dataset pe \
  --model-name microsoft/deberta-v3-base \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_deberta
```

### PE + Qwen probe run

```bash
python scripts/run_probe_experiment.py \
  --task span_role \
  --dataset pe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_probe_qwen25_7b_instruct
```

### PE + Qwen pretrained-SAE run

```bash
python scripts/run_sae_experiment.py \
  --task span_role \
  --dataset pe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/pe_span_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/pe_sae_qwen25_7b_instruct
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

### SciCite + Qwen probe run

```bash
python scripts/run_probe_experiment.py \
  --task source_materiality \
  --dataset scicite \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_probe_qwen25_7b_instruct
```

### SciCite + Qwen pretrained-SAE run

```bash
python scripts/run_sae_experiment.py \
  --task source_materiality \
  --dataset scicite \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --input-dir /content/data/scicite_material_benchmark \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_sae_qwen25_7b_instruct
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

- `probe_status.json`
- `summary.json`
- `layer_XX.metrics.json`
- `layer_XX.probe.joblib`

Pretrained-SAE runs write:

- `sae_status.json`
- `summary.json`
- `sae_layer_XX.metrics.json`
- `sae_layer_XX.features.json`
- `sae_layer_XX.classifier.joblib`

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

## SAE Strategy

Current SAE work is intentionally split into two phases:

1. **Pretrained SAE reuse**
   - use published SAEs for a public model such as `Qwen/Qwen2.5-7B-Instruct`
   - determine whether SAE feature classifiers preserve enough of the linear-probe signal
2. **Custom SAE training**
   - only after the pretrained SAE path is credible
   - prioritize benchmark/task layers that probe best, especially early layers such as source-materiality layer `1`

For Qwen, the current pretrained SAE release supports only a subset of layers:

- `3, 7, 11, 15, 19, 23, 27`

So if you want to test layer `1`, that requires a later custom-SAE track rather than the current pretrained-SAE runner.

## Candidate-Only Source Transfer

When you want to measure classifier transfer without conflating it with
LessWrong candidate-generation heuristics, use the candidate-only path.

Expected input format:

- JSONL
- one candidate per line
- required fields:
  - `post_id`
  - `name`
  - `context`
- optional fields:
  - `candidate_id`
  - `title`
  - `slug`
  - `url`
  - `origin`
  - `gold_label`

Example row:

```json
{"post_id":"6hfGNLf4Hg5DXqJCF","title":"A Fable of Science and Politics","slug":"a-fable-of-science-and-politics","candidate_id":"c1","name":"History of the Wars","context":"Procopius said ...","gold_label":"SOURCE"}
```

Run the classifier-only transfer like this:

```bash
python scripts/run_source_candidate_transfer.py \
  --candidates-path /content/data/lw_source_candidates.jsonl \
  --model-dir /content/memex-research/analysis/memex_runs/scicite_scibert/model \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_scibert/transfer_candidates
```

Run the probe-only transfer like this:

```bash
python scripts/run_source_candidate_transfer.py \
  --candidates-path /content/data/lw_source_candidates.jsonl \
  --probe-dir /content/memex-research/analysis/memex_runs/scicite_probe_qwen25_7b_instruct \
  --probe-model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir /content/memex-research/analysis/memex_runs/scicite_probe_qwen25_7b_instruct/transfer_candidates
```

Make targets are also available if you set `CANDIDATE_TRANSFER_PATH`:

```bash
make scicite-scibert-candidate-transfer CANDIDATE_TRANSFER_PATH=/content/data/lw_source_candidates.jsonl
make scicite-probe-qwen-candidate-transfer CANDIDATE_TRANSFER_PATH=/content/data/lw_source_candidates.jsonl
```

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

- `probe_status.json`
- `summary.json`
- `run_config.json`
- `diagnostics.json`
- `report.md`
- `layer_XX.metrics.json`
- `layer_XX.probe.joblib`

During long probe runs, `probe_status.json` is the fastest heartbeat. It is updated while:

- loading the tokenizer/model
- caching hidden states for each split
- fitting each layer probe
- completing the run

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
- `train_status.json` is the fastest heartbeat for classifier training runs
- `train_events.jsonl` is the full stage/step/eval event stream
- `train_analytics.json` stores split analytics and trainer history
- `transfer/` is the qualitative LessWrong sanity check

Probe path:

- `summary.json` contains the best layer ranking
- compare the best-layer F1 to the matching DeBERTa benchmark F1

Sprint 1 probe gate:

- continue probe work only if the best probe layer reaches at least `90%` of the best DeBERTa F1 on the same benchmark task

During long classifier runs, `train_status.json` is updated while:

- loading benchmark data
- loading the model
- finishing encoding
- configuring the trainer
- starting training
- beginning epochs
- completing trainer steps
- logging trainer metrics
- saving checkpoints
- completing evaluation
- running held-out prediction
- saving artifacts
- completing the run

For merged benchmark prep, the merge path writes:

- `merge_status.json`
- `merge_events.jsonl`
- `summary.json`

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
