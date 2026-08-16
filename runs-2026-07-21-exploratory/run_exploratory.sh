#!/usr/bin/env bash
# Launch the three model sweeps concurrently. Each model runs its 5 prompts x 3
# reps sequentially in its own subdirectory (resume-safe via run_all.sh), and the
# three models proceed in parallel. Progress is echoed to progress.log with one
# line per model start and one final all-done line; per-run detail lands in each
# model's run_log.tsv.
set -u
# Force a non-interactive matplotlib backend so agent plt.show() calls never
# open a blocking GUI window. Inherited by every claude subprocess and its python.
export MPLBACKEND=Agg
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRESS="$HERE/progress.log"
: > "$PROGRESS"

launch() {  # dir model
  local d="$1" m="$2"
  echo "[launch] $d model=$m $(date -Iseconds)" >> "$PROGRESS"
  ( cd "$HERE/$d" && MODEL="$m" N_REPS=3 BUDGET_USD=25 ./run_all.sh ) \
    > "$HERE/$d/launcher.log" 2>&1
  echo "[model-done] $d $(date -Iseconds)" >> "$PROGRESS"
}

launch fable-5  claude-fable-5   &
launch opus-4-8 claude-opus-4-8  &
launch sonnet-5 claude-sonnet-5  &
wait
echo "[all-done] $(date -Iseconds)" >> "$PROGRESS"
