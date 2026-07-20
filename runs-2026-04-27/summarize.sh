#!/usr/bin/env bash
# Summarize the per-run artifacts into summary.tsv.

set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT="$HERE/summary.tsv"
printf "topic\trep\tstatus\tcost_usd\tnum_turns\tduration_ms\tipynb\tpng_count\tpy_count\n" > "$OUT"

shopt -s nullglob
for dir in */; do
  dir="${dir%/}"
  [[ "$dir" == "__pycache__" ]] && continue
  [[ ! -f "$dir/transcript.jsonl" ]] && continue

  topic="${dir%-*}"
  rep="${dir##*-}"

  # status from the final result event
  if jq -e -s 'map(select(.type=="result")) | length > 0' "$dir/transcript.jsonl" >/dev/null 2>&1; then
    status="ok"
  else
    status="incomplete"
  fi

  cost="NA"; turns="NA"; dur="NA"
  if [[ -f "$dir/cost.json" ]]; then
    cost=$(jq -r '.total_cost_usd // .cost_usd // "NA"' "$dir/cost.json" 2>/dev/null)
    turns=$(jq -r '.num_turns // "NA"' "$dir/cost.json" 2>/dev/null)
    dur=$(jq -r '.duration_ms // "NA"' "$dir/cost.json" 2>/dev/null)
  fi

  ipynb=0
  [[ -n "$(find "$dir" -maxdepth 3 -name '*.ipynb' -print -quit 2>/dev/null)" ]] && ipynb=1
  png_count=$(find "$dir" -maxdepth 3 -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  py_count=$(find "$dir" -maxdepth 3 -name '*.py' 2>/dev/null | wc -l | tr -d ' ')

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$topic" "$rep" "$status" "$cost" "$turns" "$dur" "$ipynb" "$png_count" "$py_count" >> "$OUT"
done

echo "Wrote $OUT"
column -t -s $'\t' "$OUT"
