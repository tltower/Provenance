SHELL := /bin/bash

PROJECT_ROOT := $(abspath .)
DATA_ROOT ?= /workspace/data
RUN_ROOT ?= $(PROJECT_ROOT)/analysis/memex_runs
CANDIDATE_TRANSFER_PATH ?= $(PROJECT_ROOT)/analysis/memex_transfer_seed_10/source_candidates.jsonl
PE_RAW_ROOT ?= $(DATA_ROOT)/pe_raw
PE_BENCHMARK_ROOT ?= $(DATA_ROOT)/pe_span_benchmark
CDCP_DATASET_NAME ?= DFKI-SLT/cdcp
CDCP_BENCHMARK_ROOT ?= $(DATA_ROOT)/cdcp_span_benchmark
VENV_BIN := $(PROJECT_ROOT)/.venv/bin
PYTHON := $(VENV_BIN)/python

.PHONY: bootstrap quality run-inventory package-scicite-runs package-scicite-models pe-prepare pe-deberta pe-probe-qwen pe-sae-qwen cdcp-prepare cdcp-deberta cdcp-probe-qwen cdcp-sae-qwen scicite-deberta scicite-scibert scicite-deberta-transfer scicite-scibert-transfer scicite-probe-qwen scicite-sae-qwen scicite-deberta-candidate-transfer scicite-scibert-candidate-transfer scicite-probe-qwen-candidate-transfer

bootstrap:
	bash scripts/bootstrap_remote.sh

quality:
	cd $(PROJECT_ROOT) && $(PYTHON) -m pytest -q
	cd $(PROJECT_ROOT) && $(PYTHON) -m mypy src
	cd $(PROJECT_ROOT) && $(VENV_BIN)/ruff check src tests scripts

run-inventory:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/report_run_inventory.py \
		--run-root $(RUN_ROOT) \
		--output-dir $(PROJECT_ROOT)/analysis/run_inventory

package-scicite-runs:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/package_run_artifacts.py \
		--run-root $(RUN_ROOT) \
		--output $(PROJECT_ROOT)/analysis/packaged_runs/scicite_runs.tar.gz \
		--run-name scicite_deberta \
		--run-name scicite_scibert

package-scicite-models:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/package_run_artifacts.py \
		--run-root $(RUN_ROOT) \
		--output $(PROJECT_ROOT)/analysis/packaged_runs/scicite_models.tar.gz \
		--run-name scicite_deberta \
		--run-name scicite_scibert \
		--exclude-transfer \
		--exclude-predictions \
		--exclude-reports

pe-prepare:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/prepare_pe_span_benchmark.py \
		--input-root $(PE_RAW_ROOT) \
		--output-dir $(PE_BENCHMARK_ROOT)

pe-deberta:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_classifier_train.py \
		--task span_role \
		--dataset pe \
		--model-name microsoft/deberta-v3-base \
		--input-dir $(PE_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/pe_deberta

pe-probe-qwen:
	mkdir -p $(RUN_ROOT)/pe_probe_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_probe_experiment.py \
		--task span_role \
		--dataset pe \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(PE_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/pe_probe_qwen25_7b_instruct

pe-sae-qwen:
	mkdir -p $(RUN_ROOT)/pe_sae_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_sae_experiment.py \
		--task span_role \
		--dataset pe \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(PE_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/pe_sae_qwen25_7b_instruct

cdcp-prepare:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/prepare_cdcp_span_benchmark.py \
		--dataset-name $(CDCP_DATASET_NAME) \
		--output-dir $(CDCP_BENCHMARK_ROOT)

cdcp-deberta:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_classifier_train.py \
		--task span_role \
		--dataset cdcp \
		--model-name microsoft/deberta-v3-base \
		--input-dir $(CDCP_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/cdcp_deberta

cdcp-probe-qwen:
	mkdir -p $(RUN_ROOT)/cdcp_probe_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_probe_experiment.py \
		--task span_role \
		--dataset cdcp \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(CDCP_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/cdcp_probe_qwen25_7b_instruct

cdcp-sae-qwen:
	mkdir -p $(RUN_ROOT)/cdcp_sae_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_sae_experiment.py \
		--task span_role \
		--dataset cdcp \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(CDCP_BENCHMARK_ROOT) \
		--output-dir $(RUN_ROOT)/cdcp_sae_qwen25_7b_instruct

scicite-deberta:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_scicite_source_experiment.py \
		--model-name microsoft/deberta-v3-base \
		--data-root $(DATA_ROOT) \
		--output-root $(RUN_ROOT)

scicite-scibert:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_scicite_source_experiment.py \
		--model-name allenai/scibert_scivocab_uncased \
		--data-root $(DATA_ROOT) \
		--output-root $(RUN_ROOT)

scicite-deberta-transfer:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_scicite_source_experiment.py \
		--model-name microsoft/deberta-v3-base \
		--data-root $(DATA_ROOT) \
		--output-root $(RUN_ROOT) \
		--run-transfer \
		--transfer-only \
		--skip-quality-checks

scicite-scibert-transfer:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_scicite_source_experiment.py \
		--model-name allenai/scibert_scivocab_uncased \
		--data-root $(DATA_ROOT) \
		--output-root $(RUN_ROOT) \
		--run-transfer \
		--transfer-only \
		--skip-quality-checks

scicite-probe-qwen:
	mkdir -p $(RUN_ROOT)/scicite_probe_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_probe_experiment.py \
		--task source_materiality \
		--dataset scicite \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(DATA_ROOT)/scicite_material_benchmark \
		--output-dir $(RUN_ROOT)/scicite_probe_qwen25_7b_instruct

scicite-sae-qwen:
	mkdir -p $(RUN_ROOT)/scicite_sae_qwen25_7b_instruct
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_sae_experiment.py \
		--task source_materiality \
		--dataset scicite \
		--model-name Qwen/Qwen2.5-7B-Instruct \
		--input-dir $(DATA_ROOT)/scicite_material_benchmark \
		--output-dir $(RUN_ROOT)/scicite_sae_qwen25_7b_instruct

scicite-deberta-candidate-transfer:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_source_candidate_transfer.py \
		--candidates-path $(CANDIDATE_TRANSFER_PATH) \
		--model-dir $(RUN_ROOT)/scicite_deberta/model \
		--output-dir $(RUN_ROOT)/scicite_deberta/transfer_candidates

scicite-scibert-candidate-transfer:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_source_candidate_transfer.py \
		--candidates-path $(CANDIDATE_TRANSFER_PATH) \
		--model-dir $(RUN_ROOT)/scicite_scibert/model \
		--output-dir $(RUN_ROOT)/scicite_scibert/transfer_candidates

scicite-probe-qwen-candidate-transfer:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/run_source_candidate_transfer.py \
		--candidates-path $(CANDIDATE_TRANSFER_PATH) \
		--probe-dir $(RUN_ROOT)/scicite_probe_qwen25_7b_instruct \
		--probe-model-name Qwen/Qwen2.5-7B-Instruct \
		--output-dir $(RUN_ROOT)/scicite_probe_qwen25_7b_instruct/transfer_candidates
