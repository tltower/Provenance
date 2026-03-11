# Memex Greenfield Research Plan

Date: 2026-03-10

## Executive view

The strongest direction in this workspace is not "build a better notes app" and not "build a general agent manager."
It is:

- a source-grounded, argument-native knowledge product
- for people who think by writing
- where the system models claims, premises, evidence, and sources as first-class objects
- and where AI operates on that semantic layer rather than on raw text alone

The core bet is that the valuable object is not the page, and not the chat thread. It is the evolving relationship between:

- what a person is trying to say
- what supports it
- what it conflicts with
- where it came from

That is the throughline across:

- `lesswrong-provenance`
- `memex-research`
- `argument-graph-proto`
- `ai-native-editor`
- even the smaller `edu-argument-demo`

## What already exists in your work

## Resume context

Your resume shows three relevant strands:

1. quantitative modeling of complex systems
2. discourse analysis / NLP on policy and LessWrong corpora
3. strategic interest in how ideas move, compound, and shape action

Important distinction:

- the LessWrong NLP project on your resume is a past corpus-analysis project comparing RAND, arXiv, and LessWrong language
- it is not the same project as `lesswrong-provenance`

That distinction is good. It means the current work is not a random continuation of a class project. It is a deeper second-generation program:

- first: analyze discourse at the corpus level
- now: build systems that recover idea provenance, source use, and reasoning structure

## Workspace thesis by repo

### `lesswrong-provenance`

This is the production-style pipeline.

What it already proves:

- you can ingest large idea corpora
- extract citations, concepts, tags, and graph structure
- analyze memetics, community structure, and source behavior
- build a useful provenance graph without requiring a giant end-to-end model

Its strongest current lesson is architectural:

- staged pipelines beat monoliths
- extraction alone is not enough
- source identification is real infrastructure, not a UX flourish

### `memex-research`

This is the experimental sidecar.

What it already proves:

- you are not just building prompts; you are building benchmarked research machinery
- the ontology is concrete: `CLAIM | PREMISE | EVIDENCE | OTHER` and `SOURCE | NOT_SOURCE`
- you already have training, probe, SAE, transfer, and reporting infrastructure

Most important current result:

- source-materiality classifiers are already strong
- the Qwen probe result is promising
- transfer is bottlenecked by candidate generation

That means the next step is not "more vibe-based prompting." It is:

- better transfer data
- better source candidate generation
- benchmark discipline

### `argument-graph-proto` and `ai-native-editor`

These repos reveal the product thesis:

- the semantic layer should be persistent
- AI features should return patches, diagnostics, and graph updates
- the main UI is not chat but an editor with inspectable structure

This is the most important blue-sky insight in the workspace.

You are implicitly moving toward:

- an AI-native writing and reading environment
- where argument structure is the intermediate representation
- and research support is tied to specific claims and evidence gaps

### `edu-argument-demo`

This small project still fits the same pattern:

- ingest messy external information
- summarize and tag it
- turn raw text streams into structured understanding

That matters because it shows the idea is broader than LessWrong.

## My inference about your interests and beliefs

This is an inference from the repos and your notes.

You seem to believe:

- the world is becoming harder to understand faster than traditional tools improve
- people will keep wanting to write, reason, synthesize, and explain even in a highly automated world
- the best AI products are not generic assistants but tools that help humans think better about real objects
- reliability matters, so the best near-term products are ones where human review is natural
- generic workflow replacement and "agent control tower" products are crowded and likely compress toward the labs
- knowledge products remain durable because epistemic load grows with capability growth

That makes the Memex direction coherent.

It is downstream from frontier capability, but not commoditized by default, because it depends on:

- ontology
- taste
- interface design
- trust
- source discipline
- and a real theory of what "understanding" is

## What the literature says

The literature mostly supports the decomposition already emerging in your repos.

### 1. Argument mining

- Cabessa et al. 2025 show that fine-tuned open-weight LLMs can reach state of the art on PE, AbstRCT, and CDCP, which supports supervised task decomposition rather than prompt-only extraction.
- ARIES 2024 shows argument relation identification remains hard across datasets, which argues for delaying relation-heavy ambitions until span and source layers are stable.

Implication:

- continue treating span-role detection as the mainline benchmark track
- keep relation linking as a later layer, not the first hill to die on

### 2. Citation and source use

- Cohan et al. 2019 show structural context materially improves citation-intent classification and introduced SciCite.
- Spangher et al. 2023 show source identification in news is a distinct modeling problem and supports downstream source recommendation.
- Patel et al. 2024 show long-form multi-source attribution is still hard for LLMs and benefits from transformed supervision rather than raw prompting alone.

Implication:

- your `SOURCE | NOT_SOURCE` track is not peripheral; it is central
- candidate generation and attribution quality are product-defining

### 3. Argument maps as cognition tools

- Kialo's product and research framing are built around the claim that argument maps improve comprehension, evidence use, and critical thinking.

Implication:

- there is already market and pedagogical evidence that people value structured reasoning views
- but current products are mostly manual, educational, or debate-oriented

### 4. SAEs and probes

- Huben et al. 2024 show SAEs can recover interpretable residual-stream features.
- Paulo and Belrose 2026 show SAE features are not stable across seeds, so SAEs should be treated as useful decompositions, not ground truth reality.

Implication:

- probing remains a serious side track
- SAE work is worth doing for research leverage and interpretability
- but SAE should not yet be the product-critical dependency

## What companies in adjacent spaces are doing

The current market splits into several clusters.

### 1. AI knowledge workspaces

- Tana: AI-powered knowledge graph / outliner / voice capture / automation workspace
- Mem: AI thought partner for ideas, meetings, and research with recall and note cleanup
- Heptabase: visual knowledge base for research, whiteboards, PDF annotation, and source-grounded AI help

What they are proving:

- people want AI-native capture plus retrieval
- knowledge graphs and visual organization have real demand

What they are not really solving:

- rigorous claim-evidence-source structure
- argument diagnostics
- provenance-first writing

### 2. Research-grounded answer engines

- Elicit: systematic review workflow, screening, extraction, report generation
- Consensus: evidence-backed academic search and deep literature review
- NotebookLM: source-grounded notebook that answers over uploaded materials and produces study artifacts
- Perplexity: search plus spaces plus file-aware knowledge hubs

What they are proving:

- citation-grounded synthesis is a major wedge
- users care about source trust and speed to understanding

What they are not really solving:

- persistent author-side reasoning structure
- claim-level attachment of evidence inside writing workflows
- argument-aware editing

### 3. AI writing surfaces

- Lex: AI word processor with collaboration, checks, prompts, and knowledge bases

What it proves:

- writers will adopt AI if it lives inside the writing surface

What it does not center:

- argument graphs
- provenance graphs
- evidence gap tracking as the primary object

### 4. Structured debate tools

- Kialo: interactive argument maps with explicit pro/con structure and source support

What it proves:

- argument maps are legible and useful

What it lacks:

- deep AI
- source-grounded research automation
- natural import from long-form prose
- writer-centric authoring

## Market conclusion

The whitespace is not:

- "another notes app"
- "another literature review bot"
- "another chat over documents"

The whitespace is:

- a claim-evidence-source workspace for serious writing and research
- where the system can read prose, recover reasoning structure, attach sources, find gaps, and help revise
- with persistent structure across reading, drafting, and synthesis

That is genuinely different from the current clusters.

## Product thesis

The right first product is not broad consumer and not classic enterprise.

The best wedge is:

- prosumer / high-agency individuals first
- then small research-heavy teams

Best early user types:

- policy researchers
- independent writers / newsletter writers
- graduate students
- analysts
- think tank staff
- investigative or explanatory journalists
- rationalist / EA / forecasting adjacent power users

Why this wedge fits your beliefs:

- they already tolerate review loops
- they already care about source quality
- they already work in messy knowledge environments
- they can feel the difference between chat and structured understanding

## The core Memex research question

The real question is not "can we classify spans?"

It is:

- can we build a useful semantic intermediate representation for thinking work
- that beats plain embeddings and chat
- and is reliable enough to power a writing / reading product

That breaks into three research questions.

### RQ1. Representation

What is the minimum viable semantic layer?

Candidate:

- claim / premise / evidence spans
- source materiality
- local support/attack hints
- provenance metadata

### RQ2. Transfer

Does that representation hold outside benchmark datasets and outside LessWrong?

You need to know whether the system works on:

- LessWrong essays
- policy memos
- academic-style prose
- journalistic analysis

### RQ3. Usefulness

Does explicit structure actually improve user outcomes?

Not model metrics. Human outcomes:

- faster comprehension
- better sourced drafts
- fewer unsupported claims
- better revision quality
- more trustworthy AI assistance

## Recommended research program

## Phase 0: Narrow the thesis

Commit to one sentence:

> Memex is an argument-native, source-grounded workspace for understanding and writing complex ideas.

This matters because it rules out a lot:

- not general PKM
- not social media
- not agent orchestration
- not generic search

## Phase 1: Finish the benchmarked substrate

Goal:

- get the semantic layer to "credible but narrow"

Must-do work:

1. finish span-role baselines and transfer evaluation
2. improve source candidate generation, since that is the current bottleneck
3. create a benchmarked hybrid path:
   - deterministic candidate generation
   - classifier adjudication
   - structure-aware synthesis
4. compare prompt-only vs classifier-only vs hybrid on the same transfer bundle

Decision rule:

- if classifiers win clearly, they become the mainline substrate
- probes and SAEs remain research sidecars unless they beat the simpler stack on transfer or interpretability leverage

## Phase 2: Build the real transfer set

Right now the research is still too close to benchmark corpora and the 10-post seed.

Create `Memex-100`, a small but serious transfer set:

- 25 LessWrong posts
- 25 policy / strategy essays
- 25 academic or quasi-academic explainers
- 25 journalistic or analytic long-form pieces

Annotate:

- claim spans
- premise spans
- evidence spans
- material sources
- a light support relation layer where obvious

This is the highest-value next dataset you could make.

Why:

- it connects the research to the actual product domain
- it aligns with your resume and future hiring narrative
- it lets you evaluate cross-domain usefulness instead of benchmark cosplay

## Phase 3: Run product-shaped experiments, not just NLP experiments

The key thing missing from many ML-heavy projects is user-task evaluation.

Run task-based experiments inside `ai-native-editor`:

1. unsupported-claim detection
2. evidence attachment
3. source tracing
4. hostile-reader / misreading diagnosis
5. compare two drafts on argument quality

Measure:

- time to locate weak claims
- time to add support
- precision of evidence suggestions
- user trust
- whether people prefer the graph / diagnostics view to chat-only help

If this phase fails, the product thesis is wrong even if the benchmarks look good.

## Phase 4: Ship the narrowest credible product

The best MVP is still the "Show Me My Argument" direction:

- paste a document
- recover claim/premise/evidence/source structure
- show unsupported claims and dangling premises
- let users attach evidence and inspect provenance

Do not start with:

- full multi-agent research orchestration
- full automatic ghostwriting
- general-purpose second-brain positioning

Start with:

- inspectability
- diagnostics
- source-grounded revision support

## Phase 5: Expand to a true Memex

Only after the above works should you expand toward:

- multi-document reasoning
- source packs and reusable evidence objects
- concept timelines and provenance trails across many documents
- reusable claim libraries
- personalized world-model / position tracking

That is the actual Vannevar-Bush-style upside:

- not just notes
- a machine for navigating how claims connect to sources across time

## What to deprioritize

These are interesting, but likely bad priorities now.

### 1. Relation-heavy graph perfection

ARIES suggests this is still the hardest, least transferable part.

### 2. SAE as mainline product dependency

The 2026 reproducibility result is a warning.
Use SAEs for leverage, interpretation, and bloggable research.
Do not make them the thing the product must stand on this year.

### 3. Mass-market consumer positioning

The product is too cognitively heavy for a broad initial consumer wedge.
Prosumer is the better first market.

### 4. Enterprise internal-knowledge search

Perplexity and many others are already moving hard here, and the labs / platform companies can compress it further.

## What the finished research direction enables

If successful, this work can become at least four things.

### 1. A real product

An AI-native editor / reading environment for source-backed reasoning.

### 2. A research asset

A differentiated dataset and evaluation framework for:

- argument-role transfer
- source materiality
- claim-evidence tooling
- structure-aware writing assistance

### 3. A publishable / essayable narrative

You have the ingredients for strong essays or papers:

- why chat is the wrong UI for knowledge work
- why writing tools need a semantic layer
- why source materiality matters
- why argument structure is the right intermediate representation

### 4. A hiring / career story

This can position you for:

- research engineering roles
- AI product research roles
- knowledge systems / retrieval / evaluation roles
- policy-tech roles that care about evidence and synthesis

## How this works with your resume

The strongest resume story is not "I did some unrelated AI things."

It is:

- math and systems modeling
- policy / security reasoning
- discourse analysis and corpus NLP
- provenance graph construction
- benchmarked reasoning-structure research
- AI-native editor / product prototype

That is a coherent profile:

- someone who understands both complex domains and the tools needed to reason about them

If you keep going, the eventual resume should separate three distinct projects:

1. LessWrong / discourse NLP analysis
2. LessWrong provenance graph pipeline
3. Memex research plus AI-native editor

That is much stronger than flattening them together.

## Concrete 90-day plan

## Days 1-30

- finish span-role benchmark runs
- improve source candidate generation
- formalize the hybrid extractor + classifier stack
- define the `Memex-100` annotation schema
- annotate the first 20 documents

## Days 31-60

- complete `Memex-100`
- run transfer evaluation across all four domains
- wire the best current structure stack into `ai-native-editor`
- run 5-10 user sessions with serious writers / researchers

## Days 61-90

- choose the product wedge based on user-task evidence
- polish the "Show Me My Argument" demo
- write one serious essay / memo explaining the thesis
- package one public artifact:
  - demo
  - dataset
  - benchmark report
  - or all three

## Bottom line

The highest-upside move is to treat Memex as:

- not a note-taking app
- not a generic research bot
- not a mech-interp science fair project

But as:

- a new cognitive software layer for source-backed reasoning

The local repos already point there.
The literature mostly supports the decomposition.
The market has adjacent players, but none seem to own the exact intersection of:

- argument structure
- source grounding
- provenance
- and writer-centric interaction

That intersection is the bet.

## Sources

### Literature

- Cabessa et al., "Argument Mining with Fine-Tuned Large Language Models" (COLING 2025): https://aclanthology.org/2025.coling-main.442/
- Gemechu et al., "ARIES: A General Benchmark for Argument Relation Identification" (ArgMining 2024): https://aclanthology.org/2024.argmining-1.1/
- Cohan et al., "Structural Scaffolds for Citation Intent Classification in Scientific Publications" (NAACL 2019): https://aclanthology.org/N19-1361/
- Spangher et al., "Identifying Informational Sources in News Articles" (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.221/
- Patel et al., "Towards Improved Multi-Source Attribution for Long-Form Answer Generation" (NAACL 2024): https://aclanthology.org/2024.naacl-long.216/
- Huben et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (ICLR 2024): https://openreview.net/forum?id=F76bwRSLeK
- Paulo and Belrose, "Sparse Autoencoders Trained on the Same Data Learn Different Features" (ICLR 2026): https://openreview.net/forum?id=EjInprGpk9

### Company / product landscape

- Tana: https://tana.inc/
- Heptabase: https://heptabase.com/
- Mem: https://get.mem.ai/
- Elicit: https://elicit.com/
- Elicit systematic reviews: https://support.elicit.com/en/articles/7927169
- Consensus: https://consensus.app/
- Consensus deep search: https://consensus.app/home/blog/deep-search/
- NotebookLM: https://blog.google/innovation-and-ai/products/notebooklm-audio-overviews/
- Perplexity Spaces: https://www.perplexity.ai/help-center/en/articles/10352961-spaces
- Perplexity internal knowledge search: https://www.perplexity.ai/help-center/en/articles/10352958-what-is-internal-knowledge-search-for-enterprise
- Lex: https://lex.page/
- Kialo Edu features: https://www.kialo-edu.com/features
