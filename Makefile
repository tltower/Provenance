SHELL := /bin/bash

PROJECT_ROOT := $(abspath .)
DATA_ROOT ?= /workspace/data
RUN_ROOT ?= $(PROJECT_ROOT)/analysis/memex_runs
VENV_BIN := $(PROJECT_ROOT)/.venv/bin
PYTHON := $(VENV_BIN)/python

.PHONY: bootstrap quality run-inventory package-scicite-runs package-scicite-models scicite-deberta scicite-scibert scicite-deberta-transfer scicite-scibert-transfer scicite-probe-qwen

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
