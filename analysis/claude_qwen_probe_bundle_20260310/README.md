# Memex Qwen Probe Bundle (2026-03-10)

This bundle packages the first completed public-model linear-probe result for `memex-research` so Claude can review the state of the research program and advise on next steps.

This bundle is benchmark-first. The main signal here is the SciCite source-materiality benchmark, not the current LessWrong transfer adapter.

## Executive Summary

The first public-model probe run worked.

Benchmark results now available:

| Method | Model | Task | Benchmark | Macro F1 |
|---|---|---|---|---:|
| Classifier | `microsoft/deberta-v3-base` | `SOURCE / NOT_SOURCE` | SciCite | `0.8698` |
| Classifier | `allenai/scibert_scivocab_uncased` | `SOURCE / NOT_SOURCE` | SciCite | `0.8738` |
| Linear probe | `Qwen/Qwen2.5-7B-Instruct` | `SOURCE / NOT_SOURCE` | SciCite | `0.8380` |

The probe result is below the trained classifiers, but it is close enough to justify continuing the probe/SAE track:

- best classifier baseline: `0.8738`
- probe continuation threshold (`90%` of best classifier): about `0.786`
- best Qwen probe result: `0.8380`

So:

- the practical classifier track is working
- the public-model probe track is also real
- the probe track passes the continuation gate

## Probe Result

Probe run:

- model: `Qwen/Qwen2.5-7B-Instruct`
- task: `source_materiality`
- dataset: `SciCite`
- selection key: `dev_f1_macro`
- remote git head: `0e514335bab94fd105a0c3c110338500ff00d8b0`

Best layer:

- layer: `1`
- dev macro F1: `0.8405769700308644`
- test macro F1: `0.8380450933052755`

Notable shape of the result:

- early layers were strongest
- middle layers remained competitive
- deeper layers degraded gradually
- layer `0` was much worse (`0.6604` test F1), so the signal is not a trivial lexical baseline

This is strong enough to make the probe experiment worth continuing.

## Classifier Baselines

### SciCite + DeBERTa

- accuracy: `0.8699623858140785`
- macro F1: `0.8698225228350098`
- precision macro: `0.8701865326865328`
- recall macro: `0.8720739765593075`

### SciCite + SciBERT

- accuracy: `0.8742611499193982`
- macro F1: `0.8738283151380293`
- precision macro: `0.8734080303324907`
- recall macro: `0.8745420288643708`

Interpretation:

- SciBERT is the strongest benchmark baseline so far
- the gap between SciBERT and the Qwen probe is about `0.0358` macro F1
- that gap is small enough that the probe track is not a gimmick

## What This Does And Does Not Show

What it shows:

- a benchmark-trained classifier works on SciCite
- a public open-weight model contains linearly decodable source-materiality signal
- the probe path no longer depends on gated Llama access

What it does not show:

- that SAEs will work
- that transfer to LessWrong is solved
- that the transfer adapter is trustworthy yet

The current transfer adapter is still the weakest part of the repo. Earlier remote runs showed:

- `posts_with_candidates = 6 / 10`
- `total_candidates = 69`
- one post had `45` candidates while four posts had `0`

So benchmark results are still the main truth signal.

## Remote Runtime Notes

The remote Qwen probe run succeeded on:

- provider: RunPod
- GPU: `NVIDIA A100-SXM4-80GB`
- Python: `3.11.10`
- `torch`: `2.10.0`
- `transformers`: `4.57.6`
- `scikit-learn`: `1.8.0`

Operational notes:

- the run used GPU for hidden-state extraction
- it then shifted into CPU-heavy logistic-regression fitting by layer
- this mixed utilization pattern is expected for the current implementation

The repo has since been improved so the next run has better live observability:

- `probe_status.json` heartbeat
- stage prints
- explicit output-dir creation before `tee`

Those changes were pushed after this successful run, so this run itself does not include the new heartbeat file.

## Recommendation

The research tracks should stay parallel:

1. keep the classifier track
2. continue the probe track
3. delay SAE work until there is at least one more successful public-model probe result on another task

Most useful next experiments:

1. run the `span_role` classifier baseline
2. run the `span_role` public-model probe
3. compare whether argument-role signal is also linearly decodable
4. only then decide whether SAE work is worth the next pod cycle

## Questions For Claude

1. Does this Qwen probe result justify moving to SAEs now, or should the next step be another probe on a second task first?
2. Is the `0.8380` probe result close enough to the classifier baseline to count as genuinely strong, or only “promising but not yet convincing”?
3. Should the next benchmark priority be `span_role`, or should we first run a second source-materiality probe on another public model for robustness?
4. Given the earlier transfer-candidate failures, what is the cleanest way to test LessWrong transfer without conflating classifier quality and candidate generation quality?

## Included Files

- `probe_summary.json`
- `probe_report.md`
- `probe_diagnostics.json`
- `manifest.json`

