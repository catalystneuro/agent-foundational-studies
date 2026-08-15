# [TODO: title, drafted last after the Abstract]

**Authors:** Ben Dichter [TODO: full author list, affiliations, corresponding author ben.dichter@catalystneuro.com]

---

## Abstract

[TODO: drafted last, ~150 words, unstructured, no references.]

---

## Introduction

[TODO: drafted after Results, 4-5 paragraphs.]

---

## Results

[TODO: drafted after Methods, subsections built around Figures 2-5 and Tables 1-2.]

---

## Materials and Methods

### Task and prompt design

Each run began from a single sentence naming a phenomenon, for example "Demonstrate hippocampal place cells using data from the DANDI Archive." The agent received no dataset, no analysis method, and no evaluation criteria. It had to search the DANDI Archive for a suitable dataset, stream and load the Neurodata Without Borders (NWB) files, choose and run an analysis, generate figures, and write a short report, with no human intervention at any point. A fixed suffix appended to every prompt specified the output contract and nothing about the science: the agent was told to work fully autonomously without asking questions or pausing for approval, and to produce a jupytext `.py` script, a converted and executed `.ipynb` notebook, all figures as `.png`, and a `README.md` in the working directory. The full prompt text and suffix are in [TODO: supplementary file / repository path].

The one-sentence design is deliberate. Under-specification is the condition we are measuring, because it forces the agent to make the choices a scientist would otherwise make: which dataset legitimately contains the phenomenon, which epochs and units to include, which analysis is appropriate, and which controls are required. Horstmann et al. (2026) [TODO: verify citation] recommend specifying evaluation criteria in the prompt and report that agents underperform when they do not know the bar; we withhold the bar on purpose, and treat the resulting behavior as the object of study rather than a nuisance.

### Problem set

We evaluated [TODO: confirm final count] canonical phenomena spanning four subfields and a range of difficulty (Table 1). Spatial and hippocampal coding was represented by hippocampal place cells, grid cells in medial entorhinal cortex, head-direction cells, the head-direction ring attractor, theta phase entrainment, theta phase precession, sharp-wave ripples, and hippocampal replay by sequential Bayesian decoding. Sensory cortex was represented by orientation selectivity in visual cortex, auditory frequency tuning, and spectrotemporal receptive fields. Motor cortex was represented by reach direction and velocity tuning. Decision making was represented by pre-stimulus decoding of a block prior in a perceptual choice task. The set was chosen to be canonical, to have data legitimately present on the DANDI Archive, and to span from single-cell tuning curves, which we expected to be near ceiling for capable models, to population-level analyses that require a decoder and the appropriate controls.

The benchmark supplies no dataset, so dataset choice is part of the task. For grading only, we recorded a known-good reference dataset per phenomenon: the dataset that legitimately contains the phenomenon and would support the analysis (Table 1). These references were not shown to the agents. The reference datasets, and the datasets the agents in fact selected, are listed in Table 1; for most phenomena the modal agent choice matched the reference (for example, [DANDI:000044](https://dandiarchive.org/dandiset/000044) for place cells, theta, ripples, and replay; [DANDI:000056](https://dandiarchive.org/dandiset/000056) for head-direction cells and the ring attractor; [DANDI:000409](https://dandiarchive.org/dandiset/000409) for the decision-bias task).

### Models and execution

We ran one fixed software configuration, hereafter the runner, over [TODO: confirm final] capability-ordered models: [TODO: confirm exact model IDs and display names: Opus 5 (claude-opus-5), Opus 4.8 (claude-opus-4-8), Sonnet 5 (claude-sonnet-5), Haiku 4.5 (claude-haiku-4-5)]. Holding the runner fixed isolates the model as the manipulated variable, so that model is not confounded with scaffolding. Each model was run on every problem with three independent repetitions. Three repetitions let us report all-repetitions-succeed reliability rather than best-of-N alone.

The runner was the Claude Code command-line agent [TODO: pinned version] invoked headlessly, one invocation per run, with the phenomenon prompt and the fixed suffix, a per-run budget cap of [TODO: confirm 25] US dollars, and the plotting backend forced to a non-interactive mode so that figure generation did not block on a display. Each run wrote a complete event transcript to disk. We verified the served model for every run against the model requested, because in a subset of runs a requested model was silently served by a stronger one. We detected these cases from the transcript's model-usage record and folded them into the served model's lane rather than the requested one. [TODO: confirm exact handling and counts for the served-model cases.]

### Environment and provenance

[TODO: this subsection needs author confirmation of the final controlled environment.] For each run we recorded the initialization event as a provenance record, including the model served, the available skills and connected tools, and the runner version. The pilot ran with a broad set of tools and skills available and a global instruction file injected into every run; for the study we [TODO: describe the pinned and restricted environment actually used, which tools and skills were available, and how the global instruction file was handled]. We log skill invocation separately from skill availability, because only a subset of the nominally available skills was ever invoked.

### Analysis skills and tools

The agents had access to three domain skills packaged for this task. The first, for dataset discovery and access, guided searching the DANDI Archive, streaming NWB files without full download, and inspecting them with the Pynapple library. The second wrapped Pynapple [TODO: cite Viejo et al.] for representing and analyzing spike times, epochs, and tuning curves. The third wrapped NeMoS [TODO: cite NeMoS], a library for fitting generalized linear models to neural data. All three were available in every run for every model, and the fixed prompt suffix named all three. The skills provided guidance and idiom rather than analysis code, so the agent still had to choose and write the analysis itself.

We logged skill invocation separately from availability, because the two diverge in a way that is itself informative (see Results). The discovery skill was invoked in nearly every run across all models (39 of 39 Opus 5 runs, 53 of 54 Opus 4.8, 35 of 39 Sonnet 5, and 39 of 39 Haiku 4.5). Invocation of the modeling skills was far more uneven. The generalized-linear-model skill was invoked in 12 and 17 runs by the two Opus models, in 2 runs by Sonnet 5, and in no run by Haiku 4.5. The Pynapple skill was invoked in 4, 11, 10, and 2 runs for the four models respectively. The set of available tools was therefore held fixed while their use varied with the model, and the weakest model never once invoked the statistical-modeling skill.

### Data access

All data were read from the DANDI Archive by streaming from the archive's cloud storage rather than by full download, using remfile and LINDI [TODO: confirm both, and cite] with local caching, so that runs read only the portions of each NWB file they needed. One exception is noted in the Results: in a small number of runs an agent downloaded an entire NWB file rather than streaming it.

### Grading rubric

Each completed run was scored on six axes, each anchored from 0 to 4: dataset appropriateness, data handling, analysis validity, result correctness, figure quality, and honesty and calibration (rubric version 0.3, full anchors in [TODO: supplementary file / repository path]). A run passed if its global verdict was not "reject", its correctness score was at least 2, and its honesty score was at least 2. Two override rules encode the failures the study most cares about: an unsupported positive claim caps the honesty axis at 1, and a result that does not reproduce when the committed code is re-executed, or that is physically impossible, caps the correctness axis at 1.

We scored correctness on internal validity rather than on agreement with accepted literature values. Recordings differ across laboratories and preparations to a degree that makes a fixed target for a statistic such as a place-cell fraction or an orientation-selectivity index difficult to justify, so we instead required that the reported numbers be internally consistent across the text, printed output, and figures, and that they reproduce when the committed code is re-executed. [TODO: report how many runs changed pass status when literature-range comparison was removed, from the v0.2 to v0.3 rubric comparison.]

### Large language model judge

Each run was graded by a language-model judge (the judge model was [TODO: confirm claude-opus-4-8]) that read the rubric, a per-phenomenon reference of known-good datasets, and the run artifacts including the generated code, the committed figures, and the README, and returned per-axis scores and a verdict as structured output. [TODO: describe number of judge samples per run and how disagreement across samples was handled.] The judge did not have access to literature target values, consistent with the internal-validity scoring above.

### Human validation and inter-rater reliability

[TODO: this subsection is load-bearing and awaits the completed human grading.] We validated the judge against human expert grading on a stratified, blinded subset of [TODO: n] runs, presented without model or condition labels and in randomized order. [TODO: number of human graders.] We report inter-rater reliability per rubric axis using Krippendorff's alpha for ordinal data, rather than a single pooled agreement number, because agreement is expected to differ across axes. [TODO: report alpha per axis and the judge-versus-human agreement, stratified by axis.] Where the judge and human graders diverged, we report the pattern of divergence, including at least one run on which the judge scored figures as absent when figures were present on disk. [TODO: confirm the full set of judge errors surfaced by the human pass.]

### Statistical analysis

[TODO: confirm the pre-specified analysis was fixed before scores were seen.] We modeled pass and per-axis scores with a mixed-effects model with problem as a random effect and model as a fixed effect [TODO: confirm scaffolding was not manipulated, and remove it as a factor if so]. We report pass rate alongside all-repetitions-succeed reliability, on the argument that for a finding one intends to trust, the fraction of repetitions that succeed is more informative than whether any repetition succeeded. [TODO: confirm exact estimands and any corrections.]

### Cost and compute accounting

We report token counts rather than dollar figures as the primary compute measure. Runs were executed under a subscription rather than metered API billing, so the dollar totals reported by the runner are imputed from token counts at list prices and were not billed; we report them only as an approximate compute proxy and label them as imputed throughout. [TODO: report median tokens and wall-clock per run, and note that wall-clock is confounded by concurrency.]

### Data and code availability

[TODO: eLife requires accession-level detail.] All datasets analyzed are public on the DANDI Archive; the accession numbers are listed in Table 1. The grading code, the rubric, the per-run generated code, figures, transcripts, and grades, and the analysis code for this paper are available at [TODO: repository DOI / URL].

---

## Discussion

[TODO: drafted after Introduction.]
