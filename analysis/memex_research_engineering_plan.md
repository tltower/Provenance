# Memex Research Plan + Engineering Plan

## Context

The memex-research project has completed its first sprint: source-materiality classifiers work (SciBERT F1=0.8738), the Qwen probe passes the 90% continuation gate (F1=0.8380), and the SAE infrastructure is built. The span-role track (argument decomposition) has full code but zero runs yet. Transfer evaluation is bottlenecked by candidate generation, not classifier quality.

The goal is to complete the research experiments in the next few days, do document analysis, then build toward a notes app that decomposes text into argument-structured chunks and connects them in a graph.

---

## PART 1: RESEARCH PLAN

### Current Scorecard

| Track | Method | Model | Dataset | Best F1 | Status |
|-------|--------|-------|---------|---------|--------|
| Source Materiality | Classifier | DeBERTa-v3-base | SciCite | 0.8698 | Done |
| Source Materiality | Classifier | SciBERT | SciCite | 0.8738 | Done |
| Source Materiality | Linear probe (L1) | Qwen2.5-7B | SciCite | 0.8380 | Done, passes 90% gate |
| Source Materiality | SAE features | Qwen2.5-7B | SciCite | — | Code ready, not run |
| Span Role | Classifier | DeBERTa-v3-base | PE | — | Code ready, not run |
| Span Role | Classifier | DeBERTa-v3-base | CDCP | — | Code ready, not run |
| Span Role | Linear probe | Qwen2.5-7B | PE/CDCP | — | Code ready, not run |
| Span Role | SAE features | Qwen2.5-7B | PE/CDCP | — | Code ready, not run |
| Transfer | Source heuristic | — | LW 10-post | 6/10 posts | Candidate gen is bottleneck |

### Probe Result Analysis

The Qwen source-materiality probe deserves close examination because it shapes the strategy:

- **Best layer: 1** — the model forms source-materiality signal almost immediately, before deeper reasoning layers
- **Layer 0 was much worse** (0.6604 test F1) — so this is not a trivial lexical baseline
- **Early layers strongest, gradual degradation in deeper layers** — consistent with source materiality being a relatively low-level linguistic feature
- **Dev/test consistency** (0.8406 dev, 0.8380 test) — no overfitting to layer selection

The open question for span roles: will argument structure also be early-layer (suggesting a low-level feature), or will it require deeper layers (suggesting it's a higher-order reasoning representation)? This has direct implications for whether a single lightweight probe can serve both tasks.

---

### Day 1 — Span Role Baselines + SAE (GPU pod)

Run these three experiments on a RunPod A100. They are independent and can run sequentially in one session.

**Experiment 1: CDCP span-role classifier (DeBERTa)**
```
make cdcp-prepare
make cdcp-deberta
```
- CDCP is the priority dataset because it supervises EVIDENCE (fact/testimony/reference → EVIDENCE), not just CLAIM/PREMISE
- CDCP label mapping in `datasets.py`: `value|policy|claim → CLAIM`, `fact|testimony|reference|evidence → EVIDENCE`, `premise|reason → PREMISE`
- Gate: macro F1 > 0.50 → proceed to probe. Literature range: 0.55-0.75 for CDCP token classification
- Add `--run-transfer` to get span predictions on the 10-post LessWrong seed
- Key question: does the evidence label actually fire on CDCP test data, or does the class imbalance suppress it?

**Experiment 2: PE span-role classifier (DeBERTa)**
```
make pe-prepare PE_RAW_ROOT=/workspace/data/pe_raw
make pe-deberta
```
- PE gives CLAIM/PREMISE/OTHER baseline (no evidence supervision)
- PE label mapping: `major_claim + claim → CLAIM`, `premise → PREMISE`, unlabeled → `OTHER`
- Gate: macro F1 > 0.55. Literature range: 0.65-0.80 for PE
- Add `--run-transfer` for LessWrong seed comparison
- PE is ~400 essays with BRAT annotations — training should take ~30-45 min on A100
- Note: PE has no dev split. The code generates a deterministic synthetic dev split from train (seed=17)

**Experiment 3: SciCite SAE (Qwen)**
```
make scicite-sae-qwen
```
- Tests pretrained SAE feature classifiers on layers 3, 7, 11, 15, 19, 23, 27
- Uses `andyrdt/saes-qwen2.5-7b-instruct` release via sae-lens
- Feature selection: top-1024 most-active features per layer, ranked by activation sum on training data
- Per-layer LogisticRegression on selected SAE features
- Gate: best SAE F1 > 0.80 → SAE path confirmed. F1 < 0.75 → deprioritize SAE
- Answers the probe bundle question: "Does this justify SAEs now?"
- Interesting comparison: probe used layer 1, but SAE layers start at 3. If layer 3 SAE beats layer 1 probe, SAE features are extracting something the raw residual stream doesn't expose linearly

### Day 2 — Probes + Transfer Fix (GPU pod)

**Experiment 4: Span-role Qwen probe** (on whichever dataset had stronger Day 1 baseline)
```
make cdcp-probe-qwen   # or make pe-probe-qwen
```
- Gate: probe F1 >= 90% of classifier F1
- This is the critical question: is argument-role signal linearly decodable like source materiality?
- The layer distribution will be informative:
  - If layer 1 wins again → argument structure is also a low-level feature → a single early-layer probe serves both tasks
  - If a mid/deep layer wins → argument structure is a higher-order representation → may need separate probes per task
  - If no layer is competitive → argument roles may not be linearly separable from Qwen's representations → classifier-only path
- Token-level probe (not sequence-level): this uses `_token_hidden_state_vectors` which extracts per-token representations and maps back to original words. More expensive than sequence probes — watch memory on the A100
- Expected runtime: longer than source materiality because token-level extraction produces N_tokens × N_layers vectors, not N_examples × N_layers

**Experiment 5: Span-role SAE** (conditional — only if both SciCite SAE and span probe pass their gates)
```
make cdcp-sae-qwen     # or make pe-sae-qwen
```
- Only justified if we have evidence that both (a) SAE features are useful for classification and (b) span-role signal exists in the probe layers
- If the span-role probe wins at a layer that has a corresponding SAE (3, 7, 11, 15, 19, 23, 27), run the SAE on that specific layer first
- If the probe wins at a layer without a pretrained SAE, the SAE experiment is less informative

**Experiment 6: Fix transfer evaluation**
- Hand-curate `source_candidates.jsonl` for the 10-post seed set — manually identify the real sources in each post
- Run candidate-only transfer: `make scicite-deberta-candidate-transfer`
- This cleanly separates classifier quality from candidate generation quality
- Parallel engineering task, not GPU-dependent
- The candidate-only transfer path uses `run_source_candidate_transfer.py` which bypasses `build_transfer_source_candidates()` and takes pre-generated JSONL

### Day 3 — Conditional Experiments + Analysis

Depending on Day 1-2 results, one of these paths:

**If both CDCP and PE baselines pass gates:**
- Compare transfer outputs side-by-side on the 10-post seed
- CDCP should produce EVIDENCE spans where PE produces OTHER — is that distinction meaningful on LessWrong text?
- Run the second dataset's probe (the one not run on Day 2) if pod time remains

**If CDCP fails but PE passes:**
- Investigate CDCP label mapping — is `fact/testimony/reference → EVIDENCE` too aggressive?
- Try a relaxed mapping: only `evidence → EVIDENCE`, keep others as PREMISE
- Re-run with the relaxed mapping
- Fall back to PE-only for the span-role track (accept no evidence label for now)

**If both fail:**
- The token-classification path may be too hard for these datasets at this scale
- Consider: sentence-level argument mining instead (UKP Sentential as pre-filter)
- Consider: larger model for classification (DeBERTa-large instead of base)

**If SciCite SAE passes (F1 > 0.80):**
- Document which SAE layer won and compare to probe layer 1
- If the winning SAE layer ≠ probe layer → different features at different depths
- If SAE beats probe → SAE features are capturing something the raw residual stream misses

**If SciCite SAE fails (F1 < 0.75):**
- Pretrained SAE features don't retain enough task signal for source materiality
- The probe path is sufficient — deprioritize SAE work
- Custom SAE training (Phase 4B in the research plan) becomes less justified

---

### Decision Gates Summary

| Gate | Condition | Pass → | Fail → |
|------|-----------|--------|--------|
| CDCP baseline | F1 > 0.50 | Run CDCP probe | Investigate label mapping, fall back to PE |
| PE baseline | F1 > 0.55 | Confirms span path works | PE label mapping is too noisy |
| Span probe | >= 90% of classifier | SAE track alive for spans | Classifier-only for span roles |
| SciCite SAE | F1 > 0.80 | SAE path confirmed | Deprioritize SAE, probes sufficient |
| Span transfer | Majority of 10 posts look structured | Classifiers are usable | Investigate domain gap |

### Analysis Questions to Answer After Experiments

Once all results are in, the document analysis should address:

1. **Layer geography**: Where does argument structure live in Qwen vs. where source materiality lives? Same layer or different? What does this say about the representations?

2. **Evidence supervision**: Does CDCP's evidence label actually improve transfer to LessWrong compared to PE's claim/premise-only? Or does the domain gap swamp the label advantage?

3. **SAE vs. probe**: If both produce results, is there a quality/interpretability tradeoff? SAE features are named and inspectable; probe directions are not. Does the SAE give you anything the probe doesn't, beyond interpretability?

4. **Cross-task consistency**: If both tasks have probe results, can you use the same layer for both? Or do you need task-specific layer selection? This matters for whether a single frozen model with two probe heads is viable.

5. **Transfer failure modes**: On the 10-post seed, what are the actual failure modes? All-OTHER predictions? Claim/premise everywhere? Boundary errors? Each failure mode implies a different fix.

6. **Probe layer curve shape**: For span roles, does the accuracy-by-layer curve look like the source materiality curve (early peak, gradual decline)? Or is it different? The shape tells you about the nature of the representation.

---

### Dataset Priority (from available argument mining datasets)

**Use now:** PE (supported), CDCP (supported)

**Next sprint — high value, ~2 hours each to add:**
- **IBM Debater Claim Detection** — 30k labeled sentences, massive boost to claim coverage. Needs: normalization function in `datasets.py`, prepare script in `scripts/`, Makefile target. The 30k scale means the classifier sees far more diverse claim formulations than PE's 400 essays.
- **UKP Sentential** — 25k sentences across 8 topics, labeled as pro/con/non-argument. Useful as a pre-filter: "is this sentence argumentative at all?" Binary classification before role assignment could improve precision on long documents.

**After baseline experiments — medium value, medium effort:**
- **AbstRCT** — medical argument mining with claims, premises, and relations. Different domain tests generalization. Clean annotations. The relations (support/attack) could be valuable for the connection discovery layer later.
- **IBM Debater Evidence Detection** — separate from claim detection. Evidence-specific supervision that CDCP's noisy mapping might miss.

**After MVP — lower priority for current timeline:**
- **Kialo** — structured pro/con debate trees. The tree structure is valuable for relation/graph work (Track C), not for span classification (Track A). Save for when you're building support/attack edges.
- **args.me** — 380k arguments from debate portals. Scale is good but the annotation quality is lower. Use only if you need more diverse training data after the primary datasets.

**Defer — different tasks:**
- **CMV (ChangeMyView)** — persuasion effectiveness (delta-awarded), not argument structure. The signal is "did this argument change someone's mind?" which is Track C territory.
- **VAST** — stance detection across topics. Different from role decomposition. Would matter if you're building "what position does this text take?" which is a downstream feature, not a current blocker.
- **Walton's Argumentation Schemes** — scheme typing (argument from authority, from example, etc.). This is synthesis/tagging (Track C), not span classification. Interesting for later edge typing — "this PREMISE supports this CLAIM via appeal to authority" — but not needed for the decomposition pipeline.

### Adding a New Dataset — Implementation Pattern

Each new dataset requires (based on the existing PE/CDCP/SciCite pattern):

1. **Normalization function** in `datasets.py` — e.g. `normalize_ibm_claim_record(row: dict) -> SpanRoleExample`
   - Map dataset-specific labels to the canonical ontology (CLAIM/PREMISE/EVIDENCE/OTHER)
   - Handle tokenization and offset alignment
   - Return a `SpanRoleExample` or `SourceMaterialityExample` TypedDict

2. **Prepare script** in `scripts/` — e.g. `prepare_ibm_claim_benchmark.py`
   - Download or load raw data
   - Apply normalization
   - Write train/dev/test JSONL splits via `write_split_jsonl()`
   - Generate `summary.json` with counts

3. **Makefile targets** — prepare + train + probe + sae + transfer targets
   - Follow the existing pattern: `ibm-prepare`, `ibm-deberta`, `ibm-probe-qwen`, `ibm-sae-qwen`

4. **Test coverage** in `tests/test_classifier_research.py`
   - Label mapping test
   - Round-trip normalization test
   - At least one edge case

5. **Add dataset name** to `tasks.py` validation if strict coupling is enforced

Estimated time per dataset: ~2 hours for normalization + prepare script + Makefile + tests. The training/probe/SAE infrastructure is shared.

---

### Experiment Execution Checklist

For each experiment, verify these outputs exist and are sane:

**Classifier runs:**
- [ ] `metrics.json` — accuracy, precision_macro, recall_macro, f1_macro
- [ ] `run_config.json` — hyperparameters, label list, model name
- [ ] `predictions.jsonl` — per-example predictions with confidence
- [ ] `model/` — saved tokenizer + weights
- [ ] `diagnostics.json` — Python version, CUDA info, git head, package versions
- [ ] `report.md` — human-readable summary
- [ ] If `--run-transfer`: `transfer/` directory with per-post predictions + `summary.json`

**Probe runs:**
- [ ] `summary.json` — best layer, selection key, full layer rankings
- [ ] `layer_NN.metrics.json` — per-layer results
- [ ] `layer_NN.probe.joblib` — serialized LogisticRegression
- [ ] `run_config.json` — task, model, max_length, labels
- [ ] `probe_status.json` — heartbeat file (was added after the first probe run)

**SAE runs:**
- [ ] `summary.json` — best layer, SAE release, feature cap, layer rankings
- [ ] `sae_layer_NN.metrics.json` — per-layer results
- [ ] `sae_layer_NN.features.json` — selected feature indices + counts
- [ ] `sae_layer_NN.classifier.joblib` — serialized LogisticRegression on SAE features
- [ ] `sae_status.json` — heartbeat file

**Transfer runs:**
- [ ] Per-post JSON files with predictions
- [ ] `summary.json` with aggregate counts (posts_with_candidates, total_candidates, label distribution)
- [ ] `report.md`

---

## PART 2: ENGINEERING PLAN (Summary)

The engineering goal is a CLI notes app (`memex-app/`) that consumes trained model artifacts from the research experiments.

### Minimum Viable Classifier Set

1. **Span-role model** — CDCP DeBERTa (preferred) or PE DeBERTa from Day 1
2. **Source-materiality model** — SciBERT on SciCite (already trained, F1=0.8738)
3. **Embedding model** — `sentence-transformers/all-MiniLM-L6-v2` (no training needed)

Probes and SAEs are research extras. The MVP uses HF classifiers directly.

### Architecture

- **Storage**: SQLite + FTS5. Tables: notes, chunks (with role + embedding blob), edges (typed: SUPPORTS/CONTRADICTS/EXTENDS/SIMILAR/CITES), source_refs
- **Decomposition**: Port inference logic from `transfer_eval.py` — span classification, label grouping, source candidate extraction + classification
- **Connection discovery**: Three stages — within-note structural edges (free), role-typed embedding retrieval (top-k per role), LLM evaluation on top-k pairs (cheap model)
- **Daydream**: O(n*k) associative trail discovery via role-typed retrieval + incremental cluster sweeps

### Critical Files to Reuse from memex-research

| Source File | What to Port |
|---|---|
| `transfer_eval.py` lines 398-441 | `_group_labeled_tokens` → span grouping |
| `transfer_eval.py` lines 560-608 | `run_span_transfer_hf_model` → token classification inference |
| `transfer_eval.py` lines 444-500 | `run_source_transfer_hf_model` → source classification inference |
| `transfer_eval.py` lines 233-257 | `build_transfer_source_candidates` → candidate extraction |
| `transfer_eval.py` lines 215-230 | `_candidate_context_window` → context windows |
| `tasks.py` | `SPAN_ROLE_LABELS`, `SOURCE_MATERIALITY_LABELS` — canonical label definitions |
| `hf_models.py` | `get_tokenizer`, model loading patterns |

### Verification

- **Research:** After each experiment, check metrics.json and summary.json against decision gates. Run transfer eval on 10-post seed and manually inspect 3-5 posts for span coherence.
- **Engineering:** After `decompose.py` is working, decompose 3 LessWrong posts from the seed set and verify spans match the research transfer eval output. After `connect.py`, add 10 notes and verify edges are typed correctly and the graph makes sense.
