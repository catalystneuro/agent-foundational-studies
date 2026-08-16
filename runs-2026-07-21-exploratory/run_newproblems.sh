#!/usr/bin/env bash
# Run the three lower models on the six new problems only, 3 reps each.
# Resume-safe and headless via run_cell.sh (wrong-model guard included).
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
N="${PAR:-6}"
LANES="opus-4-8 sonnet-5 haiku-4-5"
TOPICS="entrainment precession strf grid headdirection ripples"

complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }

work=()
for lane in $LANES; do
  for topic in $TOPICS; do
    for rep in 01 02 03; do
      complete "$lane/${topic}-${rep}/transcript.jsonl" || work+=("$lane $topic $rep")
    done
  done
done

echo "[newq] $(date -Iseconds) incomplete=${#work[@]} concurrency=$N" >> parallel.log
printf '%s\n' "${work[@]}" | xargs -P "$N" -I{} bash -c './run_cell.sh {}'
echo "[newq-done] $(date -Iseconds)" >> parallel.log
