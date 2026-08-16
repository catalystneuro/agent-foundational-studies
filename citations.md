# Citation ledger

Two-gate verified: (1) the reference exists, (2) it supports the specific claim it is cited for.
Status: VERIFIED (both gates), PARTIAL (exists, sub-claim not yet checked), TODO (not yet searched).

---

## Positioning / related work

**[brainbench] VERIFIED.** Luo, X., ... Love, B. C. et al. (2024). Large language models surpass
human experts in predicting neuroscience results. *Nature Human Behaviour*.
doi:10.1038/s41562-024-02046-9. PubMed 39604572.
- Supports: BrainBench measures forward prediction from the literature (choose the correct of two
  altered JNeuro abstracts, 5 domains), i.e. a knowledge task, the complement to our
  execution-from-data task. Also: confident LLM predictions were more often correct (calibration).

**[horstmann] VERIFIED (core); PARTIAL on sub-claims.** Horstmann, K. A., et al. (2026). A case
study of evaluating AI agents on a neuroscience data-to-discovery pipeline. arXiv:2606.07718
(submitted 2026-06-05).
- Supports: agents complete individual pipeline stages but not correct end-to-end discovery;
  datasets orders of magnitude larger than prior benchmarks; evaluated codex (GPT-5.5) and
  claude-code (Opus 4.7). Names challenges incl. absence of predefined iteration criteria and
  weak visual self-evaluation.
- [TODO sub-claim] confirm they explicitly (a) reject LLM-as-judge for numeric tolerances,
  (b) recommend specifying evaluation criteria in the prompt, (c) name inter-scientist agreement
  as a strategy not employed. Do not cite these specifics until checked against the PDF.

**[replicationbench] VERIFIED (core); PARTIAL on sub-claims.** ReplicationBench: Can AI Agents
Replicate Astrophysics Research Papers? (2025). arXiv:2510.24591 (v2 2025-11-23).
- Supports: 111 paper-scale tasks over 20 papers, author-co-developed, scored on faithfulness and
  correctness; best frontier models under 20%; rich taxonomy of failure modes.
- [TODO sub-claim] confirm the specific "best-of-N substantially exceeds average-of-6" reporting
  before citing it for the pass^k argument.

**[paperbench] VERIFIED.** Starace, G., et al. (OpenAI) (2025). PaperBench: Evaluating AI's Ability
to Replicate AI Research. arXiv:2504.01848; PMLR v267 (ICML 2025).
- Supports: hierarchical author-co-developed rubrics (8,316 gradable leaf tasks); LLM-judge used;
  best agent (Claude 3.5 Sonnet) 21.0% vs human ML PhDs 41.4% after 48h.
- [TODO sub-claim] the "JudgeEval F1 0.94 result-matching vs 0.72 code" figure needs the PDF.

**[scienceagentbench] VERIFIED.** ScienceAgentBench: Toward Rigorous Assessment of Language Agents
for Data-Driven Scientific Discovery (2025). arXiv:2410.05080; ICLR 2025.
- Supports: 102 tasks from 44 papers over 4 disciplines including Psychology & Cognitive
  Neuroscience; 9 experts; target output a self-contained Python program.

## Data standard / archive

**[nwb] VERIFIED.** Rübel, O., Tritt, A., Ly, R., Dichter, B. K., Ghosh, S., et al. (2022). The
Neurodata Without Borders ecosystem for neurophysiological data science. *eLife* 11:e78362.
- Supports: the NWB standard our data are stored in; Dichter is a co-author.

---

## Canonical findings (problem-set primaries)

**[place-cells] VERIFIED.** O'Keefe, J. & Dostrovsky, J. (1971). The hippocampus as a spatial map.
Preliminary evidence from unit activity in the freely-moving rat. *Brain Research* 34, 171-175.
PubMed 5124915.

**[grid-cells] VERIFIED.** Hafting, T., Fyhn, M., Molden, S., Moser, M.-B. & Moser, E. I. (2005).
Microstructure of a spatial map in the entorhinal cortex. *Nature* 436, 801-806.

**[orientation] VERIFIED.** Hubel, D. H. & Wiesel, T. N. (1962). Receptive fields, binocular
interaction and functional architecture in the cat's visual cortex. *J. Physiol.* 160, 106-154.
PubMed 14449617.

**[motor-tuning] VERIFIED.** Georgopoulos, A. P., Kalaska, J. F., Caminiti, R. & Massey, J. T.
(1982). On the relations between the direction of two-dimensional arm movements and cell discharge
in primate motor cortex. *J. Neurosci.* 2, 1527-1537. (Population vector: Georgopoulos, Schwartz &
Kettner 1986, *Science* 233, 1416, PubMed 3749885.)

**[ibl-prior] VERIFIED.** International Brain Laboratory et al. (2025). A brain-wide map of neural
activity during complex behaviour. *Nature*. doi:10.1038/s41586-025-09235-0. PubMed 40903598.
- Supports the decision-bias substrate: block-structured stimulus prior (20:80 / 80:20), prior
  expectations encoded throughout the brain. [note] there are paired IBL 2025 Nature papers
  (brain-wide map + prior); confirm which to cite for the pre-stimulus prior specifically.

## Still to verify (next batch)
- [TODO dandi] DANDI Archive citation (the archive the agents search).
- [TODO magland-dichter] Magland, Ly, Rübel & Dichter, Scientific Data 2025 (Dandiset Explorer),
  our closest neuroscience precedent and explicit baseline. Confirm venue/authors/no-IRR claim.
- [TODO alizadeh / social-science] "Read the Paper, Write the Code: Agentic Reproduction of
  Social-Science Results" arXiv:2604.21965 seen in search; confirm if this is the intended cite.
- [TODO canonical findings] one primary reference per phenomenon for the problem-set table:
  place cells (O'Keefe & Dostrovsky 1971), grid cells (Hafting et al. 2005), head-direction cells
  (Taube et al. 1990), HD ring attractor (Peyrache et al. 2015), theta phase precession
  (O'Keefe & Recce 1993), sharp-wave ripples / replay (Wilson & McNaughton 1994; Foster & Wilson
  2006), orientation selectivity (Hubel & Wiesel 1962), spectrotemporal receptive fields
  (deCharms/Theunissen), reach direction tuning (Georgopoulos et al. 1982), IBL decision prior
  (IBL et al.). Verify each before it enters the draft.
