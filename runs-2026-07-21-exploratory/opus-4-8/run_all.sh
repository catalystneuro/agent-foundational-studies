#!/usr/bin/env bash
# Sequential launcher for the Claude Code (Opus 4.7) reproduction sweep.
#
# For each of the 5 prompts in prompts.tsv, runs N_REPS independent headless
# Claude sessions, each in its own scratch directory under runs-2026-04-27/.
# Resume-safe: a run whose transcript already ends with a "result" event is
# skipped on re-invocation.

set -u  # not -e: a failed run should not abort the whole sweep
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

N_REPS="${N_REPS:-3}"
MODEL="${MODEL:-claude-opus-4-7}"
BUDGET_USD="${BUDGET_USD:-25}"
ONLY_TOPIC="${ONLY_TOPIC:-}"   # set to a topic name to run just one
ONLY_REP="${ONLY_REP:-}"       # set to e.g. 01 to run just one rep

LOG="$HERE/run_log.tsv"
SYSTEM_ADDENDUM_TEXT="$(cat "$HERE/SYSTEM_ADDENDUM.md")"

if [[ ! -f "$LOG" ]]; then
  printf "timestamp\ttopic\trep\tstatus\tcost_usd\tnum_turns\tduration_ms\n" > "$LOG"
fi

# Per-prompt autonomy reminder appended to every user prompt.
AUTONOMY_SUFFIX=$'\n\nWork fully autonomously. The user will not respond again in this session. Do not ask questions, do not propose a plan for approval, and do not stop for confirmation. Use the analyzing-dandi-datasets, using-pynapple, and using-nemos skills as needed. Produce a final jupytext .py, a converted .ipynb, all figures as .png, and a short README.md in the current working directory.'

run_is_complete() {
  local transcript="$1"
  [[ -s "$transcript" ]] || return 1
  # last non-empty JSONL line should have type=="result"
  jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not))) | length > 0' "$transcript" >/dev/null 2>&1
}

run_one() {
  local topic="$1"
  local prompt="$2"
  local rep="$3"
  local rundir="$HERE/${topic}-${rep}"
  local transcript="$rundir/transcript.jsonl"

  mkdir -p "$rundir"

  if run_is_complete "$transcript"; then
    echo "[skip] $topic-$rep already has a completed transcript"
    return 0
  fi

  echo "[run ] $topic-$rep starting at $(date -Iseconds)"
  local t0
  t0=$(date +%s)

  (
    cd "$rundir"
    claude -p "${prompt}${AUTONOMY_SUFFIX}" \
      --model "$MODEL" \
      --dangerously-skip-permissions \
      --append-system-prompt "$SYSTEM_ADDENDUM_TEXT" \
      --output-format stream-json \
      --verbose \
      --max-budget-usd "$BUDGET_USD" \
      > transcript.jsonl 2> stderr.log
  )
  local rc=$?
  local t1
  t1=$(date +%s)

  # Extract the result event into cost.json (if present)
  if [[ -s "$transcript" ]]; then
    jq -s 'map(select(.type=="result")) | .[-1] // {}' "$transcript" \
      > "$rundir/cost.json" 2>/dev/null || true
  fi

  local status cost turns dur
  if run_is_complete "$transcript"; then
    status="ok"
  else
    status="incomplete"
  fi
  cost=$(jq -r '.total_cost_usd // .cost_usd // "NA"' "$rundir/cost.json" 2>/dev/null || echo "NA")
  turns=$(jq -r '.num_turns // "NA"' "$rundir/cost.json" 2>/dev/null || echo "NA")
  dur=$(jq -r '.duration_ms // "NA"' "$rundir/cost.json" 2>/dev/null || echo "NA")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Iseconds)" "$topic" "$rep" "$status" "$cost" "$turns" "$dur" >> "$LOG"

  echo "[done] $topic-$rep status=$status rc=$rc wall=$((t1-t0))s cost=$cost"
  return 0
}

while IFS=$'\t' read -r topic prompt; do
  [[ -z "$topic" ]] && continue
  [[ -n "$ONLY_TOPIC" && "$topic" != "$ONLY_TOPIC" ]] && continue
  for ((i=1; i<=N_REPS; i++)); do
    rep=$(printf "%02d" "$i")
    [[ -n "$ONLY_REP" && "$rep" != "$ONLY_REP" ]] && continue
    run_one "$topic" "$prompt" "$rep"
  done
done < "$HERE/prompts.tsv"

echo
echo "All requested runs done. Log: $LOG"
