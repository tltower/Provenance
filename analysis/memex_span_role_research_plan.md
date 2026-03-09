# Span Role / Argument Mining Research Plan

## Context

Source materiality classifier is now validated (DeBERTa F1 0.870, SciBERT
F1 0.874 on SciCite). This plan covers the parallel track: span role
classification for argument structure.

The code for this track already exists but has never been run. The span
training pipeline, PE benchmark prep, transfer evaluation, and probe
infrastructure are all written and tested. This plan is about executing
them and expanding to multiple datasets.

## Task Definition

```
span_role_classifier
  labels: CLAIM | PREMISE | EVIDENCE | OTHER
  type:   token classification (BIO-style, first-subword labeling)
  model:  DeBERTa-v3-base (primary)
```

## Known Issue: EVIDENCE Is Underspecified

Persuasive Essays only annotates `major_claim`, `claim`, and `premise`.
There is no EVIDENCE label in PE. The current code maps:

- `major_claim` → CLAIM
- `claim` → CLAIM
- `premise` → PREMISE
- unlabeled → OTHER

EVIDENCE is defined in the task spec but never assigned by the PE adapter.
This means the first trained model is effectively a 3-class classifier
(CLAIM / PREMISE / OTHER) until a dataset with evidence annotations is
added.

This is fine for v1. EVIDENCE becomes real when CDCP is integrated (its
`fact`, `testimony`, `reference` labels map to EVIDENCE).

## Phase 1: PE Baseline (Sprint 2, Day 1)

Run what already exists. No new code needed.

**Steps:**

1. Download Persuasive Essays dataset (Stab & Gurevych 2017)
   - 402 essays with BRAT annotations (.txt + .ann files)
   - standard train/test split

2. Prepare benchmark:
   ```bash
   python scripts/prepare_pe_span_benchmark.py \
     --input-root /data/pe_raw \
     --output-dir /data/pe_span_benchmark
   ```

3. Train DeBERTa:
   ```bash
   python scripts/run_classifier_train.py \
     --task span_role --dataset pe \
     --model-name microsoft/deberta-v3-base \
     --input-dir /data/pe_span_benchmark \
     --output-dir /runs/pe_deberta
   ```

4. Inspect outputs:
   - `metrics.json` → test F1 (expect ~0.70-0.80 macro F1 based on
     literature for PE with DeBERTa-class models)
   - `predictions.jsonl` → spot-check token-level predictions
   - transfer output on 10 LessWrong posts → qualitative look

**Expected baseline range:**

Literature reference points for component classification on PE:
- Stab & Gurevych (2017) original: ~0.73 macro F1
- DeBERTa-class models typically score 0.75-0.82
- If below 0.65: something is wrong with the pipeline
- If above 0.80: strong baseline, move to transfer

**Time:** 1 day (mostly waiting for training)

## Phase 2: Transfer Quality Check (Sprint 2, Day 2)

The source materiality transfer failed because candidate generation was
too weak (4/10 posts got zero candidates). Span role transfer is
different — it operates on raw text, not on pre-extracted candidates.
Every post gets predictions. The question is whether those predictions
are sensible.

**Steps:**

1. Read the 10 transfer output files from Phase 1
2. For each post, check:
   - Are claim spans actually claims (assertions the author makes)?
   - Are premise spans actually supporting reasoning?
   - Does OTHER correctly capture non-argumentative text?
   - Are span boundaries reasonable (not cutting mid-sentence)?
3. Write a short qualitative note per post (2-3 sentences)

**What good transfer looks like:**

A LessWrong post like "An Intuitive Explanation of Bayes's Theorem"
should produce:
- CLAIM spans on Eliezer's main assertions about Bayesian reasoning
- PREMISE spans on his supporting examples and derivations
- OTHER on narrative framing, asides, meta-commentary

**What bad transfer looks like:**

- Everything labeled OTHER (model learned essay-specific cues)
- Claims and premises swapped systematically
- Span boundaries at random token positions

**Decision gate:** If transfer is qualitatively plausible on 6+/10 posts,
proceed. If not, the PE→LessWrong domain gap is too large and we need
CDCP or in-domain data before continuing.

**Time:** 2-3 hours (manual reading)

## Phase 3: Add CDCP Dataset (Sprint 2, Days 3-5)

CDCP (Cornell) is the priority second dataset because:
- It has EVIDENCE-equivalent labels (fact, testimony, reference)
- It's user-generated text (closer to LessWrong than student essays)
- It has support/attack relations (useful later for relation classifier)
- It's the dataset the original research plan specified

**Steps:**

1. Download CDCP dataset
   - 731 paragraphs from an online discussion forum
   - Labels: policy, value, fact, testimony, reference

2. Write CDCP adapter in datasets.py:
   ```python
   def normalize_cdcp_span_record(record) -> SpanRoleExample:
       # policy, value → CLAIM
       # fact, testimony, reference → EVIDENCE
       # unlabeled → OTHER
   ```
   Note: CDCP has no direct PREMISE equivalent. This is a known mapping
   issue. Options:
   - Map everything non-claim to EVIDENCE (lose premise/evidence
     distinction)
   - Map `testimony` to PREMISE and `fact`/`reference` to EVIDENCE
   - Keep CDCP as a transfer stress test only, not a training source

3. Add CDCP prep script (similar to prepare_pe_span_benchmark.py)

4. Train DeBERTa on CDCP alone, evaluate on held-out split

5. Train DeBERTa on PE+CDCP combined, evaluate on both test splits

6. Compare:
   - PE-only model on CDCP test
   - CDCP-only model on PE test
   - Combined model on both
   - All three on LessWrong transfer

**Key question this answers:** Does multi-dataset training help or hurt
transfer to LessWrong?

**Time:** 3-5 days (includes writing adapter, debugging format issues)

## Phase 4: Probes (Sprint 2-3, Parallel Track)

Run alongside Phases 1-3. Same probe infrastructure used for source
materiality, applied to span roles.

**Steps:**

1. Fix model allowlist in hf_models.py to accept Qwen2.5-7B (or
   authenticate for Llama-3.1-8B access on RunPod)

2. Run span role probes:
   ```bash
   python scripts/run_probe_experiment.py \
     --task span_role --dataset pe \
     --model-name Qwen/Qwen2.5-7B \
     --input-dir /data/pe_span_benchmark \
     --output-dir /runs/pe_probe_qwen
   ```

3. Analyze layer-wise probe accuracy:
   - Which layers encode argument roles best?
   - Compare peak probe F1 to DeBERTa baseline
   - Is the 90% gate met?

4. If running source materiality probes in the same session, compare:
   - Do span roles and source materiality peak at the same layers?
   - If different layers: evidence that these are genuinely different
     representations (supports two-classifier architecture)
   - If same layer: the representations might be entangled (interesting
     for SAE work)

**Probe gate:** Best layer F1 >= 90% of DeBERTa span role F1

**Time:** 1-2 days (GPU time, mostly extraction)

## Phase 5: Expand If Transfer Works (Sprint 3)

Only if Phase 2 gate passes.

**Additional datasets to consider (pick 1-2):**

| Dataset | Labels | Domain | Why |
|---|---|---|---|
| AbstRCT | background/objective/method/result | medical abstracts | structured argumentation |
| ArgMicro | proponent/opponent/claim/premise | microtexts | clean small dataset |
| UKP Web Discourse | claim/premise | web forum posts | closest to LessWrong domain |
| IBM Debater Claims | claim/evidence | Wikipedia | large scale |

**Priority:** UKP Web Discourse if available (domain match), otherwise
AbstRCT (clean labels, different domain for robustness).

**For each new dataset:**
1. Write adapter in datasets.py
2. Define label mapping to CLAIM/PREMISE/EVIDENCE/OTHER
3. Train, evaluate, transfer to LessWrong
4. Add to combined training if it helps transfer

## Phase 6: Integration Into Memex Pipeline (Sprint 3-4)

Once span role classification transfers plausibly:

1. Export best model checkpoint
2. Write inference wrapper that takes raw text → labeled spans
3. Integrate into lesswrong-provenance pipeline:
   - Run after ingest, before extraction
   - Output: per-post span annotations in parquet
   - These become input features for source materiality classifier
     (the plan specifies "whether the mention lies inside a
     CLAIM/PREMISE/EVIDENCE span" as a source classifier feature)
4. Re-evaluate source materiality classifier with span features added
   - Does knowing "this citation appears inside a CLAIM span" improve
     SOURCE/NOT_SOURCE classification?

This is where the two classifier tracks merge.

## Deliverables

After Sprint 2:

1. PE span role baseline metrics (DeBERTa F1 on held-out test)
2. 10-post LessWrong qualitative transfer notes
3. CDCP adapter and cross-dataset comparison table
4. Probe layer-wise accuracy curves (if GPU access resolved)
5. Go/no-go on SAE for span roles

After Sprint 3:

6. Multi-dataset training results
7. Best model checkpoint for span role classification
8. Integration plan for lesswrong-provenance pipeline

## Comparison Table Template

Fill this in as runs complete:

| Model | Train Data | Test Data | F1 | Accuracy | Notes |
|---|---|---|---|---|---|
| DeBERTa-v3-base | PE | PE test | ? | ? | Phase 1 |
| DeBERTa-v3-base | CDCP | CDCP test | ? | ? | Phase 3 |
| DeBERTa-v3-base | PE+CDCP | PE test | ? | ? | Phase 3 |
| DeBERTa-v3-base | PE+CDCP | CDCP test | ? | ? | Phase 3 |
| Probe (Qwen 7B) | PE | PE test | ? | ? | Phase 4 |
| DeBERTa-v3-base | PE | LessWrong 10 | qualitative | - | Phase 2 |

## Acceptance Criteria

Sprint 2 is successful if:

- PE baseline F1 is in the expected range (0.70-0.82)
- LessWrong transfer is qualitatively plausible on 6+/10 posts
- CDCP adapter runs end-to-end
- Probe experiment runs (even if gate fails)
- All results documented in a bundle for Claude review

The argument mining track is viable if the trained span classifier
produces visibly sensible CLAIM/PREMISE labels on LessWrong blog posts.
If it doesn't, the domain gap is too large and in-domain labeling
becomes necessary before this track can continue.
