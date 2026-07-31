#!/usr/bin/env bash
# Run the Opus 5 lane over its 10-problem set (read from opus-5/prompts.tsv),
# 3 reps each, in a pool. Resume-safe and headless via run_cell.sh, which also
# carries the wrong-model guard so an Opus 5 fallback would be flagged, not
# silently counted.
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
N="${PAR:-6}"
LANE=opus-5

complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }

work=()
while IFS=$'\t' read -r topic _; do
  [ -n "$topic" ] || continue
  for rep in 01 02 03; do
    complete "$LANE/${topic}-${rep}/transcript.jsonl" || work+=("$LANE $topic $rep")
  done
done < "$LANE/prompts.tsv"

echo "[opus5] $(date -Iseconds) incomplete=${#work[@]} concurrency=$N" >> parallel.log
printf '%s\n' "${work[@]}" | xargs -P "$N" -I{} bash -c './run_cell.sh {}'
echo "[opus5-done] $(date -Iseconds)" >> parallel.log
