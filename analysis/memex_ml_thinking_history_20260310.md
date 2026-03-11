# Memex ML Thinking History

Date: 2026-03-10

## Why This Exists

This note preserves the evolution of the ML thinking around Memex so Claude or any later reader can see:

- what was considered
- what was rejected
- what changed
- and why the current direction looks the way it does

This is intentionally more historical than [memex_ml_research_plan_20260310.md](/Users/tatetower/Codex/memex-research/analysis/memex_ml_research_plan_20260310.md).

## Starting Point

The starting point in the repo was relatively clean and benchmark-local:

- one task family for `span_role`
- one task family for `source_materiality`
- one dataset at a time
- BERT-style classifiers for the practical path
- probes and SAEs as a serious but bounded side track

Relevant earlier notes:

- [memex_pipeline_research_plan_20260309.md](/Users/tatetower/Codex/memex-research/analysis/memex_pipeline_research_plan_20260309.md)
- [memex_research_engineering_plan.md](/Users/tatetower/Codex/memex-research/analysis/memex_research_engineering_plan.md)
- [memex_span_role_research_plan.md](/Users/tatetower/Codex/memex-research/analysis/memex_span_role_research_plan.md)
- [memex_sae_research_plan.md](/Users/tatetower/Codex/lesswrong-provenance/analysis/memex_sae_research_plan.md)

The original local direction was:

- get strong benchmark baselines
- test Qwen probes
- only then decide whether SAE work is worth more effort

## The Expansion Idea

The next idea was broader and more ambitious:

- use many different datasets from many different fields
- use many citation benchmarks for the citation classifiers / SAEs
- use many argument benchmarks for the argument classifiers / SAEs
- assume the underlying models already know a lot of the latent structure
- use benchmarks as different views onto that same latent structure

This was motivated by a stronger claim:

- the impressive part is not one classifier on one benchmark
- the impressive part is pulling out implicit structure that the models already know

This also fit the desired public story:

- good for product work
- good for a website / resume
- good for showing real technical taste

## First Attempt: Big Canonicalization

The first natural reaction to many datasets was:

- map everything into one canonical label space
- pool the data
- train unified models

Attractive features of this idea:

- one clean ontology
- one pooled benchmark story
- easy to say "the model learned the same thing across datasets"

Why it became unattractive:

- labels do not line up cleanly
- some datasets are span-based, some sentence-based, some relation-heavy
- coercing everything into one ontology risks hiding the hard parts
- the canonicalization layer starts becoming the real project

This is where the approach started feeling messy.

The core problem was not that a canonical layer is impossible.
The problem was that it threatened to consume the entire research budget before the actual representation question was answered.

## The "Am I Screwing Myself?" Moment

At this point the main worry became:

- am I screwing myself by using multiple datasets with different labeling?

The answer that emerged was:

- no, not if the heterogeneity is treated honestly
- yes, if all the different schemes are silently collapsed and then overclaimed

That led to a clearer distinction:

- multiple datasets are a feature, not a bug
- but only if the project is framed as a representation-learning program rather than a benchmark soup

## Shift: Portfolio Strategy Instead Of Ontology Hell

This led to a cleaner position:

- use many datasets
- do not over-canonicalize up front
- keep datasets legible as datasets
- compare representations across them

This became the portfolio strategy.

Instead of saying:

- "all these benchmarks are really the same task"

the better claim became:

- "these benchmarks are different views onto related latent structure, and we want to see whether the same model family captures that structure across domains"

That shift preserved the ambition while dropping the worst engineering burden.

## Product Ontology vs Research Ontology

There was also a temporary push toward product-first labels such as:

- supposition
- support
- evidence
- contradiction

That still may be right for the product.

But the settled view for the ML work became:

- do not force the product ontology into the research layer too early
- benchmark-safe labels should dominate the first research phase
- contradiction should be deferred to a later relation or diagnostic layer

This avoided turning the first ML phase into a full ontology-design project.

## The "Second Most Likely Label" Idea

One practical intuition that came up was:

- if the model's top label is not used by a particular dataset, maybe use the second most likely label instead

This was reaching for something real:

- one shared model might know more labels than any one dataset uses
- inference should respect the valid label set of the dataset

After looking into similar work, the idea got refined.

The better framing is:

- constrained inference
- or valid-label masking

That means:

- if a dataset only allows a subset of labels, predictions should be restricted to that subset before selection

That is much cleaner than an ad hoc "take second best" rule.

## Research On Similar Scenarios

The literature supported a more careful version of the broad-dataset idea.

Useful precedents:

- Shui et al. 2024 train across multiple citation-intent datasets and explicitly handle cross-dataset transfer and negative transfer:
  - [Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.974/)
- Augenstein et al. 2018 address disparate label spaces with shared spaces and transfer rather than pretending all labels are identical:
  - [NAACL 2018](https://aclanthology.org/N18-1172/)
- Li et al. 2023 improve learning across compatible but non-identical label sets using compatibility constraints:
  - [Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.1041/)
- Hemmer et al. 2023 show constrained decoding over valid outputs is a serious method, not a hack:
  - [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.416/)
- Pappas and Henderson 2019 show label semantics can support generalization:
  - [TACL 2019](https://aclanthology.org/Q19-1009/)
- Paulo and Belrose 2026 caution against treating SAE feature lists as uniquely correct:
  - [ICLR 2026 OpenReview](https://openreview.net/forum?id=EjInprGpk9)

What this changed:

- the broad-dataset idea stayed alive
- naive canonicalization got downgraded
- shared representations plus controlled output handling became the more credible direction

## Why SAEs Stayed Central

Another important clarification happened here.

Earlier planning sometimes treated SAEs as:

- interesting
- but secondary to the practical classifier path

The later thinking pushed harder:

- the SAE / probe side is the most technically differentiated part
- it is the strongest research-engineering signal
- it is the part most likely to be impressive on a website or resume

The settled position is not:

- "ignore classifiers and only do SAEs"

It is:

- classifiers are the practical extraction path
- probes show the representation is present
- SAEs are the most distinctive way to expose and reuse that representation

## Settled V1 Direction

The current direction is:

- broad benchmark portfolio
- minimal dataset-specific hacks
- benchmark-safe labels first
- separate task families for:
  - argument roles
  - source use
- coarser common surfaces in v1 when needed
- probes and SAEs as central, not decorative

And, crucially:

- many datasets are still the right bet
- just not through one giant early canonicalization layer

## Current Experimental Architecture Preference

Two architectures survived as serious options.

### 1. Shared features, separate heads

This is the cleaner and more defensible baseline.

Idea:

- one shared representation source per family
- one lightweight head per dataset

Why it survived:

- minimal ontology drama
- honest per-dataset evaluation
- still tests the latent-structure thesis

### 2. Superset head plus label masking

This is the bolder unified-output experiment.

Idea:

- one shared output head over a superset family label space
- dataset-valid label masking at inference

Why it survived:

- closest thing to the original "one system that understands many schemes" intuition
- lets the model express a richer family label space without pretending every dataset uses all labels

Why it is second, not first:

- it is riskier
- it is easier to overclaim
- it should be compared against the cleaner separate-head baseline

## Final Shape Of The Story

The story now looks like this:

1. The original repo proved the benchmark-first scaffold.
2. The next ambition was broad multi-dataset latent-structure extraction.
3. Giant canonicalization looked too messy.
4. The project shifted to a portfolio strategy.
5. "Second-best valid label" matured into masked valid-label inference.
6. The broad benchmark ambition stayed.
7. The SAE / probe line became more central, not less.

## Bottom Line

The most durable conclusion from this round of thinking is:

- do not shrink the ambition
- do not simplify by lying

Use many datasets.
Keep the label spaces honest.
Let the shared representations carry as much of the generalization burden as possible.
Use classifiers, probes, and SAEs together.

That is the current Memex ML story.
