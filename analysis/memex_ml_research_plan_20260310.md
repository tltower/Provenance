# Memex ML Research Plan

Date: 2026-03-10

## Purpose

This note captures the current ML research direction for Memex.

The core thesis is:

- language models already encode a large amount of latent argument and source-use structure
- that structure should be extractable in ways that are useful for products
- the right way to test this is not one benchmark at a time, but a broad benchmark portfolio across domains and labeling schemes

The goal is to build a research program that is:

- technically impressive
- product-relevant
- honest about label mismatch and dataset heterogeneity
- strong enough to serve as a resume / website / research-engineering signal

## Current Grounded State

Relevant current project surfaces:

- [README.md](/Users/tatetower/Codex/memex-research/README.md)
- [scripts/README.md](/Users/tatetower/Codex/memex-research/scripts/README.md)
- [tasks.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/tasks.py)
- [datasets.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/datasets.py)
- [train_span.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/train_span.py)
- [train_source.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/train_source.py)
- [probes.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/probes.py)
- [sae.py](/Users/tatetower/Codex/memex-research/src/memex_research/classifier_research/sae.py)
- [memex_research_engineering_plan.md](/Users/tatetower/Codex/memex-research/analysis/memex_research_engineering_plan.md)
- [memex_pipeline_research_plan_20260309.md](/Users/tatetower/Codex/memex-research/analysis/memex_pipeline_research_plan_20260309.md)

Current repo reality:

- the codebase is benchmark-first
- `span_role` is currently framed as token/span classification
- `source_materiality` is currently framed as sequence classification
- supported benchmark datasets in the standalone repo are currently:
  - `pe`
  - `cdcp`
  - `scicite`
- supported public probe models currently include:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.1-8B`
  - `mistralai/Mistral-7B-v0.3`
- supported pretrained SAE reuse currently exists for:
  - `Qwen/Qwen2.5-7B-Instruct`

Current benchmark results already in hand:

- `SciCite + DeBERTa-v3-base`: macro F1 `0.8698`
- `SciCite + SciBERT`: macro F1 `0.8738`
- `SciCite + Qwen2.5-7B-Instruct` linear probe: macro F1 `0.8380`
- best probe layer on `SciCite`: `1`

Relevant run bundles:

- [claude_sprint1_remote_bundle_20260309/README.md](/Users/tatetower/Codex/memex-research/analysis/claude_sprint1_remote_bundle_20260309/README.md)
- [claude_qwen_probe_bundle_20260310/README.md](/Users/tatetower/Codex/memex-research/analysis/claude_qwen_probe_bundle_20260310/README.md)

## Research Thesis

The main claim of this program should be:

- there is reusable latent structure for argument roles and source use across heterogeneous datasets
- BERT-family classifiers can exploit that structure directly
- open-model residual streams and SAE features also expose that structure

This is not just a practical-classifier program and not just a mech-interp program.
It is a combined extraction program:

- discriminative classifiers as the practical path
- probes as the "is the representation already there?" path
- SAEs as the "can we expose that representation in a sparse, reusable form?" path

## Main Design Choices

## 1. Use many datasets, but do not start with giant canonicalization

The project should use a large benchmark portfolio.

However, it should **not** begin by forcing every dataset into one hand-built canonical ontology.

Reason:

- label schemes differ
- annotation units differ
- domains differ
- a giant canonicalization layer quickly becomes its own research project

So the first serious program should be:

- broad benchmark portfolio
- minimal per-dataset adapters
- shared representations
- honest per-dataset evaluation
- cross-dataset transfer where it is interpretable

## 2. Separate task families

Keep two distinct ML families:

### Argument-role family

Purpose:

- recover proposition / support / evidence style structure from text

Important scope note:

- product ontology can later differ from benchmark ontology
- contradiction is out of scope for the first role head and should be handled later as a relation or diagnostic layer

### Citation / source-use family

Purpose:

- detect whether a candidate mention is materially used as a source in context

The source-use family should span multiple fields, not just scientific citation benchmarks.

## 3. Bet on model capability, not dataset-specific hacks

The governing preference for v1 is:

- trust the model to learn broad common structure
- avoid special-case tricks for each dataset whenever possible

This means:

- broader dataset coverage
- lighter adapters
- shared model families
- fewer bespoke conversion rules

It also means accepting a coarser prediction surface in v1 when necessary.

## Task Family Plans

## Argument-role family

### V1 prediction surface

The first broad multi-dataset argument program should be sentence/chunk-first rather than exact-span-first.

Reason:

- more datasets become compatible
- fewer dataset-specific tweaks are needed
- it fits the "learn the latent structure" thesis better than a heavily engineered span pipeline

This does **not** mean exact spans are unimportant.
It means exact span extraction should be a later refinement layer rather than the first unification target.

### Dataset roadmap

Tier 1:

- `PE`
- `CDCP`
- `AAEC`
- `AbstRCT`
- `ArgMicro`
- `SciArg`

Tier 2:

- UKP / sentential argument datasets
- IBM claim / evidence style datasets
- additional essay or scientific argument corpora that normalize cleanly into sentence/chunk examples

### What should be learned

The family should learn broad roles such as:

- proposition / claim-like units
- support / premise-like units
- evidence-like units
- other / non-argumentative units

The exact product-facing labels can remain a later layer.

## Citation / source-use family

### V1 prediction surface

The common unit should be:

- `candidate mention + local context`

This works across:

- scientific citation intent datasets
- source-attribution datasets
- other cross-field source-use benchmarks

It is a better common surface than whole-document or whole-passage labeling.

### Dataset roadmap

Tier 1:

- `SciCite`
- `ACL-ARC`
- Spangher et al. source / informational-source dataset in news

Tier 2:

- more cross-field source datasets
- mention-detection or attribution auxiliaries only when they clearly help candidate generation

Out of scope for v1:

- quote-attribution as a full first-class family

Quote attribution may become useful later, but it should not muddy the first source-use family.

## Co-Primary Model Architectures

The written program should treat two architectures as co-primary, but with a clear execution order.

## Architecture A: Shared features, separate heads

This is the clean baseline.

Per family:

- one shared representation source
  - BERT-family encoder for classifier work
  - Qwen hidden states / SAE features for probe and SAE work
- one simple head per dataset

This keeps the label spaces honest while still testing the core claim:

- if shared representations work across many datasets, then the model is capturing something deeper than one benchmark's annotation scheme

This architecture should run first.

## Architecture B: Superset head plus label masking

This is the bolder shared-output experiment.

Idea:

- define a superset label space for a task family
- train one head over that space
- at inference time, mask out labels not valid for the target dataset before selecting the prediction

Important clarification:

- this is **not** "just take the second-most-likely label"
- the correct framing is constrained inference over a dataset's valid label subset

This architecture should run **after** the separate-head baseline, not before it.

## Experiment Matrix

## 1. Classifier experiments

For each dataset:

- train a BERT-family classifier baseline
- evaluate on the same dataset

Then:

- compare performance patterns across datasets within a family
- where label mapping is honest, run cross-dataset transfer

The point is not just raw benchmark strength.
The point is whether a shared family of models can learn structurally similar tasks across different domains and schemes.

## 2. Probe experiments

Use `Qwen/Qwen2.5-7B-Instruct` as the primary open-model line.

Run:

- per-dataset probes
- cross-dataset probes where the comparison is interpretable

Main question:

- is the relevant structure already linearly decodable from the residual stream?

Important outputs:

- best layer by dataset
- layer curve shape by dataset
- cross-dataset consistency of winning layers

## 3. SAE experiments

Use pretrained SAE reuse on top of Qwen as the first serious SAE program.

Run:

- per-dataset SAE-feature classifiers
- cross-dataset SAE comparisons where possible

Main question:

- do SAE features preserve the same useful signal that the probes see?

This is the most technically differentiated part of the research.

## Evaluation Philosophy

The main evaluation should be:

- cross-dataset transfer

Not:

- only one benchmark at a time
- only pooled scores
- only LessWrong transfer

Preferred evaluation ladder:

1. same-dataset benchmark strength
2. family-level comparison across datasets
3. cross-dataset transfer where the label relationship is interpretable
4. curated LessWrong transfer as a secondary reality check

What should **not** happen:

- collapsing everything into one hidden giant canonicalization layer and claiming a pooled score proves generalization

## How to Treat LessWrong Transfer

LessWrong transfer is still useful, but it should not be the main truth signal for this program.

Reason:

- candidate generation is still a major confound on the source side
- the new multi-dataset program is trying to prove cross-benchmark latent structure first

So the correct order is:

- benchmark portfolio first
- cross-dataset transfer second
- curated target-domain transfer third

## Decision Rules

The program should count as successful if it shows most of the following:

- strong same-dataset benchmark baselines across several datasets
- shared representation methods do not collapse across heterogeneous benchmarks
- probe results remain meaningfully strong across more than one dataset
- SAE-feature classifiers preserve enough signal to remain competitive and interpretable
- cross-dataset transfer is materially better than trivial or domain-specific baselines

The program should be considered weaker if:

- results only hold on one benchmark
- pooled or shared approaches collapse outside the training dataset
- SAE features look flashy but add little beyond the probe and classifier baselines

## Why This Matters

If this works, the story is strong:

- LLMs encode useful latent structure across many annotation schemes and domains
- that structure can be extracted into product-usable form
- BERT classifiers are the practical pull-out mechanism
- probes and SAEs show the structure is already present in the model

That is a strong basis for:

- future product work
- technical writing
- research-engineering hiring

## Immediate Implications For The Repo

The current standalone repo is a good starting scaffold, but not yet the full program.

What the current repo already supports well:

- benchmark adapters
- per-dataset training
- per-dataset probe experiments
- per-dataset pretrained SAE experiments

What the broader program will need later:

- many more dataset adapters
- sentence/chunk-first argument-role pipeline support
- broader source-use benchmark ingestion
- family-level comparison and transfer harnesses
- optional masked-label inference experiments

## Related Work

- Shui et al. 2024, multi-dataset citation-intent training with transfer-aware weighting:
  - [Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.974/)
- Augenstein et al. 2018, learning with disparate label spaces:
  - [NAACL 2018](https://aclanthology.org/N18-1172/)
- Li et al. 2023, compatible-label constraints across semantic role datasets:
  - [Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.1041/)
- Hemmer et al. 2023, constrained decoding over valid outputs:
  - [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.416/)
- Pappas and Henderson 2019, label semantics and generalization:
  - [TACL 2019](https://aclanthology.org/Q19-1009/)
- Paulo and Belrose 2026, SAE feature instability:
  - [ICLR 2026 OpenReview](https://openreview.net/forum?id=EjInprGpk9)

## Bottom Line

The right ML story is not:

- one benchmark
- one label scheme
- one classifier

It is:

- many heterogeneous benchmarks
- minimal dataset-specific hacks
- practical BERT classifiers
- probe and SAE analysis on the same tasks
- and cross-dataset evidence that the model is learning real latent structure
