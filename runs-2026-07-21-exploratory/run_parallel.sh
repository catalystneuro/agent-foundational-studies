#!/usr/bin/env bash
# Fan the remaining incomplete cells across a pool of PAR concurrent workers,
# mixed across all three models, instead of one sequential lane per model.
# Resume-safe: genuinely-complete cells are skipped by run_cell.sh.
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
N="${PAR:-6}"

complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }

work=()
for d in fable-5 opus-4-8 sonnet-5 haiku-4-5; do
  for topic in placefields orientation auditory reach theta; do
    for rep in 01 02 03; do
      complete "$d/${topic}-${rep}/transcript.jsonl" || work+=("$d $topic $rep")
    done
  done
done

echo "[parallel] $(date -Iseconds) incomplete=${#work[@]} concurrency=$N" >> parallel.log
# Interleave so consecutive workers hit different models (spreads load).
printf '%s\n' "${work[@]}" | sort -k3 -k2 \
  | xargs -P "$N" -I{} bash -c './run_cell.sh {}'
echo "[parallel-done] $(date -Iseconds)" >> parallel.log
