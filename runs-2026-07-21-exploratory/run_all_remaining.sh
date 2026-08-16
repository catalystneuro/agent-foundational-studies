#!/usr/bin/env bash
# One consolidated pool over every incomplete cell across all lanes, reading each
# lane's own prompts.tsv (so heterogeneous problem sets work). Kept at a modest
# concurrency to stay under the account rate limit; resume-safe and headless.
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
N="${PAR:-5}"

complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }

work=()
for lane in fable-5 opus-4-8 sonnet-5 haiku-4-5 opus-5; do
  [ -f "$lane/prompts.tsv" ] || continue
  while IFS=$'\t' read -r topic _; do
    [ -n "$topic" ] || continue
    for rep in 01 02 03; do
      complete "$lane/${topic}-${rep}/transcript.jsonl" || work+=("$lane $topic $rep")
    done
  done < "$lane/prompts.tsv"
done

echo "[allrem] $(date -Iseconds) incomplete=${#work[@]} concurrency=$N" >> parallel.log
printf '%s\n' "${work[@]}" | xargs -P "$N" -I{} bash -c './run_cell.sh {}'
echo "[allrem-done] $(date -Iseconds)" >> parallel.log
