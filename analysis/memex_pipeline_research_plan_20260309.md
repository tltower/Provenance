# Memex Pipeline Research Plan

## Position

The Memex pipeline should **not** be researched as one monolithic system.
It should be researched as **four separate benchmark tracks**:

1. argument-role detection
2. citation/source materiality
3. document synthesis and tagging
4. semantic embedding / clustering

Only after those tracks are individually credible should they be composed
into the full pipeline.

This is the correct structure because the failure modes are different:

- argument-role errors are mostly span-boundary and ontology problems
- citation/source errors are mostly materiality and candidate-generation
  problems
- synthesis/tagging errors are mostly summarization and schema-discipline
  problems
- embeddings/clustering errors are mostly representation and retrieval
  problems

Trying to optimize them all at once makes it impossible to know what is
actually failing. The first goal is not scaffolding. The first goal is
to learn, on benchmarks, whether each technique works at all.

## Target Pipeline

### Pass 1: Argument Roles

Input:
- raw document text

Output:
- labeled spans
  - `CLAIM`
  - `PREMISE`
  - `EVIDENCE`
  - `OTHER`

Purpose:
- expose the local reasoning structure of the document
- support later truthiness / support-path reasoning
- provide better features for citation/source classification
- enable role-aware multi-document synthesis

### Pass 2: Citations / Sources

Input:
- raw document text
- candidate references
  - inline mentions
  - bibliography-like entries
  - external links
  - title-like source mentions
- optional features from pass 1
  - whether the candidate appears inside `CLAIM`, `PREMISE`, or
    `EVIDENCE` text

Output:
- reference candidates classified as
  - `SOURCE`
  - `NOT_SOURCE`
- later split `NOT_SOURCE` into `ALLUSION` vs `DROP` only if needed

Purpose:
- identify which references are materially used
- generate corpus-level citation/source links
- provide source-backed support structure for the graph

### Pass 3: Synthesis / Tags / Local Summary

Input:
- raw text
- argument-role spans from pass 1
- source outputs from pass 2

Output:
- document summary
- normalized tags / topics
- short reasoning synopsis
- structured fields for later corpus integration

Purpose:
- create the human-readable entry point for each document
- reduce later dependence on raw text for navigation and clustering
- support multi-document synthesis with structured local summaries

This pass is still an LLM call. It should remain downstream of the
classifier passes so it can consume better structure than the current
prompt-only provenance extractor does.

### Pass 4: Embeddings

Input:
- full text
- summary
- tags
- optionally role-separated text views
  - claim-only text
  - evidence-only text
  - source-backed passages only

Output:
- document embeddings
- optional role-conditioned embeddings

Purpose:
- clustering
- nearest-neighbor search
- retrieval for synthesis
- cross-document argument grouping

## What This Plan Is Not

This is **not** an integration plan.

It does not assume:
- argument-role outputs must already feed the source classifier
- the synthesis pass must already consume classifier outputs
- the embedding layer must already depend on the earlier passes

Those may all be true later. They are not part of the first research
phase.

The first research phase asks:
- can span-role methods work on benchmark data?
- can source-materiality methods work on benchmark data?
- can synthesis/tagging become more stable with structure-aware inputs?
- can different embedding views help clustering/retrieval on benchmark or
  curated transfer material?

## How The Pipeline Is Used

### Clustering

Use:
- embeddings
- tags

For:
- topical clustering
- meme / concept neighborhood discovery
- duplicate or near-duplicate reasoning patterns

### Corpus Graph Construction

Use:
- summaries
- citations / sources
- document ids

For:
- source links
- idea lineage
- citation neighborhoods
- provenance trails across documents

### Truthiness / Reasoning Model

Use:
- argument-role spans
- source-backed passages
- later relation links if added

For:
- checking whether a claim is supported by evidence vs pure assertion
- differentiating narrative text from reasoning text
- later support / attack graph construction

### Multi-Document Synthesis

Use:
- summaries
- tags
- claim/premise/evidence segmentation

For:
- compare claims across documents
- aggregate evidence across documents
- produce role-aware syntheses rather than plain semantic summaries

## Research Tracks

The project should proceed in **parallel tracks**, not one long linear
track and not one dependency chain.

## Track A: Argument Role Research

Goal:
- determine whether benchmark-trained span-role models work on their own
  benchmarks and whether they qualitatively transfer at all to
  LessWrong-style prose

Primary tasks:
- `CLAIM | PREMISE | EVIDENCE | OTHER`

Recommended datasets:
1. `PERSUADE`
   - best evidence supervision
2. `CDCP`
   - more web-like / forum-like argumentative writing
3. `PE`
   - useful claim/premise baseline, but weak on evidence

Order:
1. `PERSUADE` baseline
2. `CDCP` baseline
3. `PE` as auxiliary / comparison

Important note:
- do not pretend PE alone solves evidence detection

Success criterion:
- strong benchmark performance relative to literature / baseline
- transfer predictions on the fixed LessWrong set look plausibly
  structured, not degenerate

## Track B: Citation / Source Materiality Research

Goal:
- determine whether benchmark-trained source/use classifiers work on their
  own benchmarks and whether they transfer to informal blog-style
  reference behavior

Primary tasks:
- `SOURCE | NOT_SOURCE`

Known benchmark status:
- SciCite + DeBERTa baseline succeeded
- SciCite + SciBERT baseline succeeded and slightly outperformed DeBERTa
- current transfer bottleneck is candidate generation, not classifier
  quality

Recommended datasets:
1. `SciCite`
2. `ACL-ARC`
3. news/source-use datasets later if needed

Immediate next work:
- improve LessWrong candidate generation before judging transfer quality
- do not confuse classifier performance with candidate-generator failure

Success criterion:
- benchmark performance is strong and reproducible
- transfer behavior on LessWrong becomes interpretable once candidate
  generation is fixed

## Track C: Synthesis / Tagging Research

Goal:
- determine whether a structure-aware LLM synthesis pass is better than a
  monolithic extractor prompt

Input contract:
- role spans
- source outputs
- raw text

Questions:
- does structured input improve summaries?
- do tags become more stable?
- do local document synopses become better for later clustering?

This is an LLM research track, not a classifier track.
It can begin with synthetic or benchmark-style document packs even before
Tracks A and B are integrated, as long as the evaluation target is clear.

## Track D: Embedding / Clustering Research

Goal:
- determine what text view is best for retrieval and clustering

Candidate representations:
- full text embeddings
- summary embeddings
- tags-only embeddings
- claim-only embeddings
- evidence-only embeddings
- source-backed-passage embeddings

Questions:
- which representation clusters arguments best?
- which representation clusters topics best?
- does role-conditioned embedding help multi-document synthesis?

This track is mostly evaluation and retrieval design, not model training.

## Shared Evaluation Discipline

The tracks are separate, but they should share the same discipline:

- benchmark metrics first
- transfer sanity checks second
- only then any integration work

Each track should answer, independently:
- what is the benchmark?
- what is the baseline?
- what is the best current model?
- what fails on transfer?
- what remains uncertain?

## What Not To Do

- do not optimize one giant prompt that tries to do everything
- do not treat probe/SAE work as the only path
- do not merge all datasets into one ontology too early
- do not judge transfer while candidate generation is obviously broken
- do not integrate into the main provenance pipeline before each track
  has at least one credible evaluation

## Concrete Next Steps

1. Patch the probe stack to allow a public open model for experiments
   instead of the gated Llama path.
2. Continue Track B benchmark work, because it already has successful
   SciCite baselines.
3. Start Track A with a dataset that actually supervises evidence
   (`PERSUADE` first, not PE alone).
4. Define Track C benchmark tasks explicitly before running more LLM
   experiments.
5. Treat Track D as its own representation-research track, not something
   that waits on full classifier integration.

## Decision Rule

This project is successful if it ends with:
- a credible source-materiality research track
- a credible argument-role research track
- a credible synthesis/tagging research track
- and a credible embedding/clustering research track

Integration can come later.
