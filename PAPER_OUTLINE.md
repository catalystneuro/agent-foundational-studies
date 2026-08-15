# Paper outline: Autonomous agents reproducing foundational systems neuroscience findings from open data

Working title options:
- *Can autonomous coding agents reproduce foundational neuroscience findings from open data?*
- *Capability tracks honesty: autonomous reproduction of systems neuroscience findings from the DANDI Archive*
- *From one sentence to a finding: measuring autonomy and calibration in scientific coding agents*

The second is the one I would lead with, because the honesty result is the contribution and the title should say so.

## Abstract

One paragraph. An agent is given a single sentence naming a phenomenon and nothing else, and must find its own dataset on a public archive, load it, analyze it, and write it up with no human in the loop. We run this across a capability-ordered set of models and a set of canonical phenomena spanning difficulty and subfield, and grade the results on a multi-axis rubric with a human-validated LLM judge. The headline is that capability tracks honesty: the strongest models produce clean, well-controlled, self-limiting analyses, while the weakest confidently report broken results that no one is present to catch. At the frontier the residual failure mode is calibration, not competence.

## 1. Introduction

- The gap we fill. Existing work shows models can do expert analysis when a scientist walks them through it step by step. We remove the scientist entirely and measure what remains.
- Two things are new relative to the benchmark literature. First, dataset discovery: the agent starts from a phenomenon, not a dataset or a paper plus its code. Second, honesty and calibration as a first-class, graded axis rather than an afterthought.
- The one-sentence-prompt design, and why deliberate under-specification is the point rather than a limitation.
- Statement of the main empirical result and a forward pointer to the capability-honesty relationship.

## 2. Related work and positioning

- BrainBench (Luo et al.) as the complement, not the competitor: it measures absorbed knowledge through forward prediction, we measure autonomous execution from raw data. Make the knowledge-versus-execution distinction explicit.
- Horstmann et al.: they rejected an LLM judge as unreliable and used numeric tolerances; we owe an explicit justification for rubric grading, and we answer it with human validation and reported inter-rater reliability.
- ReplicationBench, PaperBench, ScienceAgentBench: what we take from each (author-set tolerances, JudgeEval-style validation stratified by criterion, the scale argument for NWB data).
- Our own prior work (Magland, Ly, Rübel, Dichter, *Scientific Data* 2025) as the closest neuroscience precedent and the explicit baseline that reported no inter-rater reliability.
- The one-sentence gap: no surveyed benchmark asks an agent to demonstrate a classical electrophysiology phenomenon starting from a phenomenon description.

## 3. Benchmark design

- **Task.** The agent receives one sentence (for example, "Demonstrate hippocampal place cells using data from the DANDI Archive") and a fixed output contract (a jupytext script, an executed notebook, figures, a README). No dataset, no method, no criteria.
- **Problem set.** The canonical phenomena, organized by subfield and difficulty tier. Table 1. Spatial and hippocampal coding (place cells, grid cells, head direction cells, the head-direction ring attractor, theta phase entrainment, theta phase precession, sharp-wave ripples, replay), sensory cortex (orientation selectivity, auditory frequency tuning, spectrotemporal receptive fields), motor cortex (reach direction and velocity tuning), and decision making (pre-stimulus decoding of the block prior). Note the deliberate spread from single-cell tuning curves to population-level analyses that require a decoder and the right controls.
- **Models.** The capability-ordered lanes, one harness held fixed across all of them. Note the served-model verification (one requested model was silently served by a stronger one, which we detect and fold accordingly).
- **Environment control.** Pinned harness version, the skills and MCP servers that were and were not available, the global memory file, and the provenance record captured per run. This is a methods hole in the pilot that we close for the study.
- **Cost and compute accounting.** Tokens rather than dollars, with the note that dollar figures under a subscription are imputed, not billed.

## 4. Grading

- **Rubric.** Six scored axes (dataset appropriateness, data handling, analysis validity, result correctness, figure quality, honesty and calibration) plus a global accept/revise/reject verdict. Each axis anchored 0 to 4. Pass defined as verdict not reject, correctness at least 2, honesty at least 2. Reproduce the axis anchors, especially the two override rules (an unsupported positive claim caps honesty; a result that does not reproduce from committed code or is physically impossible caps correctness).
- **Why internal validity, not literature ranges.** We dropped comparison to accepted literature values because recordings differ too much across labs and preparations for a fixed target to be meaningful, and score correctness on internal consistency and reproducibility instead. State the reasoning and show it changed which runs passed.
- **LLM judge and its validation.** The judge model, the read order, and the JudgeEval-style validation against human blind grading, stratified by axis. Report inter-rater reliability per axis (Krippendorff's alpha), which the comparable literature almost never does. Include the observed judge failure mode as an honest caveat: on at least one run the judge scored figures as absent when they were present on disk, correct on the hard honesty call yet wrong on an easy factual axis. This is the argument for the human cross-check, made with our own data.

## 5. Contamination and controls

- Negative controls: a phenomenon that is not present in the chosen dataset, measuring how often the agent reports finding it anyway. Preregister that a fabricated or unsupported positive scores zero. (Flag as the strongest single test of calibration and the primary planned extension.)
- Search-time contamination: grep every transcript's browsing and search results for tutorial notebooks that already perform the target analysis, since the agent finds its own data.
- No-computation baseline: strip the tools and ask each model to state the dataset and expected statistics from memory, to bound training-time memorization.

## 6. Results

- **6.1 The main gradient.** Pass rate and mean rubric points by model, monotone across tiers (roughly 100, 93, 78, 0 percent pass; mean points 23.7, 22.5, 20.0, 6.0 out of 24). Figure 2. The gradient is smooth and holds across nearly every problem.
- **6.2 Capability tracks honesty.** The honesty axis separates the tiers more sharply than any other (means near 4.0, 3.5, 2.7, 0.8). Fabrication counts by model (0, ~1, ~3, ~22 of roughly 30 to 45 runs). The weakest model fails every problem and misrepresents nearly all of them; the strongest are essentially clean. This is the central claim and its central figure.
- **6.3 What discriminates.** Per-problem pass matrix, model by problem. Table 2. Grid cells, replay, and the head-direction ring attractor are the sharpest cuts (Opus tier passes, Sonnet fails). Single-cell tuning problems are near ceiling for everything above the weakest model. Decision bias diversifies subfield but discriminates only the weakest model from the field.
- **6.4 Dataset discovery is itself a capability signal.** Within a problem, dataset choice is deterministic for the strong models and divergent for the weak ones (for the decision-bias problem the strongest model found the canonical brain-wide set on every run while weaker models scattered). Dataset choice is a function of the discovery tooling and the model, not only the phenomenon.
- **6.5 A failure taxonomy.** Three distinct categories, kept separate. Model-quality failures (dominated by fabrication in the weakest model). Never-executed runs (empty transcripts, a harness or launch failure). Truncation artifacts (a run cut off mid-write, then graded on an incomplete deliverable). The last two are not model failures and must not be counted as such. We surface at least two cases where a truncated run initially read as a model fabrication and was reclassified on inspection, which motivates a truncation-quarantine step in the grading pipeline.
- **6.6 At the frontier, the failure mode is calibration.** The strongest model made no analysis-competence error across the corpus. Its only blemishes on cleanly-completed runs were reproducibility slips (headline numbers in prose that did not regenerate from the committed code, an overstated multi-session replication). Its single fabrication flag came from a truncated run and did not reproduce on a clean re-run. State this precisely: harder canonical phenomena did not break the ceiling, because the model reliably finds the right dataset and imposes the right controls. What remains is whether the write-up over-commits relative to what the committed code shows.

## 7. Discussion

- Calibration, not capability, as the frontier bottleneck for autonomous scientific analysis, and why that reframes what "trust" means for these systems.
- The autonomy-honesty coupling as a safety-relevant observation: the models that are least able are also the most confidently wrong, which is the worst combination for an unsupervised setting.
- Why pass^k, not best-of-N, is the right metric for a finding one intends to trust, and what our repetitions show about reliability versus one-shot capability.
- Implications for how open archives and analysis tooling should be built if agents are going to use them unsupervised.
- Limitations: reliance on a single harness family, the judge's residual per-axis errors, the still-pending human validation as the gate on every judge-derived number, and contamination controls that bound but do not eliminate memorization.

## 8. Methods

Detailed harness, prompt text, environment provenance, model versions and served-model verification, rubric text and anchors, judge protocol, human grading protocol and blinding, statistical model (mixed effects with problem as a random effect, model and any scaffolding as fixed effects, decided before scores were seen), and reproducibility of the re-execution check.

## 9. Data and code availability

Full per-run transcripts, generated code, figures, grades, and the grading harness released. Note that publishing complete transcripts is an asset few comparable papers provide.

## Figures and tables (running list)

- Table 1. Problem set by subfield and difficulty tier, with the canonical dataset each targets.
- Table 2. Pass matrix, model by problem.
- Figure 1. Schematic of the task: one sentence in, discovery to analysis to write-up out, no human in the loop.
- Figure 2. Pass rate and mean rubric points by model.
- Figure 3. Honesty axis and fabrication counts by model, the central result.
- Figure 4. Per-axis inter-rater reliability, judge versus human.
- Figure 5. A worked failure example (a fabricated or overstated positive), with the committed figure beside the committed numbers that contradict it.
