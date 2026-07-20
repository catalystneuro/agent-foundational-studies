# Agent Foundational Studies

A benchmark of how well a coding-agent harness, given only a one-line description of a classical neuroscience phenomenon, can autonomously (a) discover an appropriate dataset on the [DANDI Archive](https://dandiarchive.org/), (b) load and inspect it, and (c) execute the standard analysis end-to-end to demonstrate the phenomenon.

## What this is, and what this is not

**This is not a claim about agents doing novel research.** The analysis code for the foundational findings reproduced here — place fields, orientation tuning, frequency tuning, reach tuning, theta phase precession — has been written in textbooks, tutorials, and open-source notebooks for decades, and is almost certainly in the training data of any modern frontier model. We are not measuring scientific novelty.

What we *are* measuring is whether the agent can:

1. Pick a relevant public dataset given a one-line phenomenon description.
2. Load streaming NWB data correctly (LINDI / remfile).
3. Run the appropriate classical analysis end-to-end and produce a notebook + figures that demonstrate the phenomenon.

Roughly: *can it do the kind of analysis a competent first-year neuroscience graduate student would be asked to do as a rotation exercise?*

## The five prompts

Each prompt is a single sentence; the agent receives nothing else describing the task.

1. **Place cells** — *Demonstrate hippocampal place cells using data from the DANDI Archive.*
2. **Orientation selectivity** — *Demonstrate orientation selectivity in the visual system using data from the DANDI Archive.*
3. **Auditory frequency tuning** — *Demonstrate auditory frequency tuning using data from the DANDI Archive.*
4. **Reach tuning** — *Demonstrate reach direction and velocity tuning with data on the DANDI Archive.*
5. **Theta phase precession** — *Demonstrate theta phase entrainment and precession for hippocampal place cells using data from the DANDI Archive.*

## What's in the repo

| Directory | Harness | Model | Date | Runs |
|-----------|---------|-------|------|------|
| `foundational-studies-results/` | Cline | (as of Oct 2025) | 2025-10-26 | 1–2 per prompt (8 total, includes a 6th `reach_prep` prompt) |
| `runs-2026-04-27/` | Claude Code | `claude-opus-4-7` | 2026-04-27 | 3 per prompt (15 total) |

Each per-run subdirectory contains the agent's generated scripts, a Jupyter notebook, figures, a short `README.md`, and (for the new runs) `transcript.jsonl` with the full agent trace and `cost.json` with token/cost metadata.

## Reproducing the runs

### Claude Code (Opus 4.7) — `runs-2026-04-27/`

Prerequisites:

- **`claude` CLI** (Claude Code) — tested with v2.1.x. Install per [docs.anthropic.com/claude-code](https://docs.anthropic.com/claude-code).
- **Anthropic API access** to the `claude-opus-4-7` model.
- **Skills installed** at `~/.claude/skills/`:
  - `analyzing-dandi-datasets`
  - `using-pynapple`
  - `using-nemos`
- **Python environment** with `pynapple`, `pynwb`, `lindi`, `remfile`, `nemos`, `jupytext`, `tqdm`, `matplotlib`, `numpy`, `scipy`, `scikit-learn`. (The agent will install missing packages on its own as needed.)
- `jq` for transcript post-processing.

Run the full sweep:

```bash
cd runs-2026-04-27
./run_all.sh
```

The launcher is sequential and resume-safe — it skips any run whose `transcript.jsonl` already ends with a `result` event, so you can interrupt and re-run without re-doing completed work.

Useful environment variables:

```bash
N_REPS=3              # reps per prompt (default 3)
MODEL=claude-opus-4-7 # model alias or full ID
BUDGET_USD=25         # per-run --max-budget-usd (hard ceiling)
ONLY_TOPIC=placefields  # run just one topic
ONLY_REP=01             # run just one rep
```

Per-run command (built by `run_all.sh`):

```bash
claude -p "<prompt + autonomy directive>" \
  --model claude-opus-4-7 \
  --dangerously-skip-permissions \
  --append-system-prompt "$(cat SYSTEM_ADDENDUM.md)" \
  --output-format stream-json --verbose \
  --max-budget-usd 25 \
  > transcript.jsonl 2> stderr.log
```

`--dangerously-skip-permissions` is what makes the runs autonomous — the agent never pauses for tool-use approval. Each run is confined to its own `<topic>-<NN>/` scratch directory.

After the sweep, summarize with:

```bash
./summarize.sh    # writes summary.tsv: cost, turns, duration, ipynb present, png count
```

Cost: with the default $25 per-run cap, the upper bound for 15 runs is ~$375. Typical realized cost will be lower; `summary.tsv` reports actual.

### Cline (baseline) — `foundational-studies-results/`

These runs were produced in October 2025 with Cline. The exact session is not identically re-runnable here (it depended on Cline's MCP servers — `neurosift-tools`, `repo-search`, `plot-vision` — and the model version available at the time), but the prompts and `.clinerules` are committed for reference. The prompts in the table above are taken verbatim from the `cline_task_*.md` files in those directories.

## Output structure of each run

```
<topic>-<NN>/
├── transcript.jsonl    # streamed agent events (Claude Code runs only)
├── stderr.log          # CLI stderr (Claude Code runs only)
├── cost.json           # final result event (cost, num_turns, duration_ms)
├── *.py                # generated analysis scripts (often jupytext)
├── *.ipynb             # final consolidated notebook
├── *.png               # figures
└── README.md           # agent-written summary of the dataset and finding
```

## Citation / preprint

A preprint describing this benchmark is in preparation. The headline numbers (success rate per prompt, cost per run, comparison vs. the Cline baseline) come straight from `runs-2026-04-27/summary.tsv`.

## License

See [LICENSE](LICENSE).
