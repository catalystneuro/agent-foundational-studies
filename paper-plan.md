# Paper plan

## Central claim (one sentence)
Autonomous LLM coding agents can reproduce foundational systems neuroscience findings from open data starting from a one-sentence prompt, and the degree to which they do it honestly rather than confidently reporting broken results tracks their capability, so that at the frontier the residual failure mode is calibration rather than competence.

## Target journal / article type
eLife, Research Article. Publish-review-curate model (public reviews). Length budget:
abstract ~150 words, introduction 4-5 paragraphs, total main text 6,000-8,000 words, full
Materials and Methods in the main text. Mandatory accession-level data availability. Rigor
over novelty: write the evidence, its scope, and its limits to be legible.

## Audience and framing
Neuroscience-methods first: what open neuro-data plus autonomous agents mean for the field,
with honesty and calibration as the measured axis. Human grader-validation is load-bearing
(inter-rater reliability central, Figure 4); IRR numbers are [TODO: needs author] until the
blind human grading is complete.

## Voice (from CLAUDE.md profile + PREREGISTRATION.md sample)
- Hedging: states plainly what the evidence shows, concedes limits openly; "suggests" for
  inference, "shows/find" for direct results. Non-promotional.
- Person: "we" throughout; passive acceptable in Methods, active elsewhere.
- Headings: descriptive, name the topic, journal sentence case.
- No em dashes. Specific numbers over qualitative claims. Admit surprise/ignorance where real.
- [TODO: run style_check.py on Magland/Dichter Scientific Data 2025 for sentence-length baseline.]

## What exists already (inputs to draw on)
- PREREGISTRATION.md (OSF-style prereg, in Ben's voice) — design, hypotheses, controls.
- RUBRIC.md (v0.3) — six-axis rubric, anchors, override rules.
- PAPER_OUTLINE.md — approved paragraph-level outline (Stage 1 largely done).
- grades.csv + per-run grade.json — full graded corpus, 4 model lanes x ~15 problems x 3 reps.
- grading/ harness (grade_run.sh, grade_all.sh, judge instructions, dataset reference).
- Full per-run transcripts, generated code, figures.
- Results in hand: monotone pass gradient (~100/93/78/0%); mean rubric points 23.7/22.5/20.0/6.0;
  honesty axis the sharpest tier separator; fabrication counts 0/1/3/22; grid, replay, ring
  attractor as top discriminators; decision bias as diversifier; failure taxonomy
  (model-quality vs never-executed vs truncation); Opus 5 calibration-only residual failures.

## Voice
- Primary profile: ~/.claude/CLAUDE.md voice directive (measured, expository, plain, no em dashes,
  descriptive Title Case headings, first person, honest/non-promotional).
- In-repo samples in the same voice: PREREGISTRATION.md.
- Prior published paper for baseline (to fetch): Magland, Ly, Rubel & Dichter, Scientific Data 2025
  (Dandiset Explorer). [TODO: confirm we can pull text for a style_check baseline.]

## Figure list (from outline)
- Table 1. Problem set by subfield and difficulty tier, with the canonical dataset each targets.
- Table 2. Pass matrix, model by problem.
- Fig 1. Task schematic: one sentence in, discovery-analysis-writeup out, no human in loop.
- Fig 2. Pass rate and mean rubric points by model.
- Fig 3. Honesty axis and fabrication counts by model (central result).
- Fig 4. Per-axis inter-rater reliability, judge vs human. [depends on human validation, pending]
- Fig 5. Worked failure example: committed figure beside the committed numbers that contradict it.

## Open scoping decisions
1. Target venue (below).
2. Scope of first paper re: human validation (load-bearing now, or LLM-judge-first with human IRR deferred).
