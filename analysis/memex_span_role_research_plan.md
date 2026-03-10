# Span Role Research Plan

## Position

This is a **parallel benchmark track**, not the next stage of a single
pipeline.

Its only job is to answer two questions:

1. can a benchmark-trained span-role model learn useful argumentative
   structure at all?
2. do those representations transfer plausibly to LessWrong-style prose
   without any in-domain labels?

Do **not** integrate anything into a larger Memex pipeline until those
questions are answered.

## Task Definition

Target ontology:

- `CLAIM`
- `PREMISE`
- `EVIDENCE`
- `OTHER`

Reality for the first benchmark runs:

- the first supported benchmark path is effectively a
  `CLAIM | PREMISE | OTHER` baseline
- `EVIDENCE` stays in the task schema, but it is not meaningfully
  supervised until we add an evidence-bearing dataset

That is fine. The first sprint is about validating the track, not
pretending the full ontology is already solved.

## Benchmark Order

### Phase 1: PE Sanity Baseline

Purpose:

- validate the token-classification training path
- validate the transfer-eval surface on raw LessWrong text
- get a first `CLAIM | PREMISE | OTHER` baseline cheaply

Dataset:

- Persuasive Essays / UKP

Current label mapping:

- `major_claim` -> `CLAIM`
- `claim` -> `CLAIM`
- `premise` -> `PREMISE`
- unlabeled -> `OTHER`

Current repo behavior:

- deterministic synthetic `dev` split if benchmark `dev` is absent
- token-alignment validation during PE normalization

Success condition:

- benchmark metrics are in a plausible literature range
- transfer outputs on the 10-post seed set are not degenerate
  - not all `OTHER`
  - not claim/premise everywhere
  - boundaries look coherent

### Phase 2: Evidence-Bearing Dataset

Purpose:

- turn `EVIDENCE` into a real supervised label
- test whether evidence behavior transfers differently from claim/premise

Priority order:

1. `PERSUADE`
2. `CDCP`
3. `PE` remains as an auxiliary baseline/comparison dataset

Why this order:

- `PERSUADE` gives explicit evidence supervision
- `CDCP` is more web-like and supports later relation work
- `PE` is still useful, but weak on the part we care about most

Important note:

- do **not** merge new datasets casually
- each dataset gets its own benchmark run first
- only after that do we test mixed training

### Phase 3: Probe Track

This is the research add-on for span roles, not the first implementation
path.

Order:

1. train the benchmark classifier baseline
2. run the public-model linear probe
3. compare probe F1 to the best classifier F1
4. only then decide whether SAE work is justified

Probe continuation gate:

- best probe F1 must reach at least `90%` of the best benchmark
  classifier F1 on the same task

If the gate fails:

- stop the SAE path for span roles
- keep the classifier path

If the gate passes:

- the probe/SAE track stays alive as a research sidecar

### Phase 4: SAE Track

This is split into two separate experiments:

1. **pretrained SAE reuse**
2. **custom SAE training**

That split matters. The first question is whether an SAE feature basis can
retain useful span-role signal at all. The second question is whether a
task-targeted SAE improves on the public release.

#### Phase 4A: pretrained SAE reuse

Use a public SAE release for the same public probe model:

- `Qwen/Qwen2.5-7B-Instruct`

Goal:

- compare SAE-feature classifiers directly against the linear-probe
  baseline
- decide whether the SAE path is worth more compute

Success condition:

- best SAE-feature classifier is close enough to the best Qwen probe to
  remain interesting as a research direction

#### Phase 4B: custom SAE training

Only do this if the pretrained SAE path is credible.

When custom SAE training starts:

- train on the best probe layers first
- if span-role probes later show an early-layer winner, include that
  early layer explicitly
- do not assume the public SAE layers are the right ones for this task

This is the point where layer choice becomes an actual experimental
question rather than a tooling constraint.

## Transfer Evaluation

Span-role transfer is easier to interpret than source transfer because it
does **not** depend on candidate generation.

That means transfer review should be explicit and qualitative.

Use the fixed 10-post LessWrong seed set and score each post on:

1. are predicted `CLAIM` spans actual assertions?
2. are predicted `PREMISE` spans actual support or reasoning?
3. if `EVIDENCE` exists, are those spans concrete support rather than
   generic exposition?
4. are span boundaries coherent?
5. is the model obviously degenerate?

The point is not to pretend we have a formal gold set yet. The point is
to decide whether zero-shot transfer is promising enough to justify more
work.

## Decisions

### Continue the span-role track if:

- benchmark performance is solid
- at least a majority of the 10 transfer posts look plausibly structured
- the model is not obviously overfit to essay formatting

### Pivot before adding more engineering if:

- PE baseline transfers terribly
- the evidence-bearing dataset mapping is too noisy to trust
- probes are far below the classifier baseline

## Explicit Non-Goals

Not in scope for this stage:

- relation linking
- support/attack graph construction
- pipeline integration into `lesswrong-provenance`
- using span outputs as features for the source classifier
- multi-document synthesis

Those all come later, and only if this track works on its own.
