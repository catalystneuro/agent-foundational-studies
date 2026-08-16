#!/usr/bin/env bash
# Self-heal an open-weights lane against flaky OpenRouter infra: each pass, keep genuine
# runs (>15-turn successes) and re-run everything else (api_error deaths + truncations),
# until all 39 are genuine or MAXPASS passes. Then grade the genuine successes with the
# subscription judge (never through the proxy).
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg
lane="$1"; PAR="${2:-2}"; MAXPASS="${3:-8}"
eval "$(grep -E '^export OPENROUTER_API_KEY=' "$HOME/.zshrc")"; export OPENROUTER_API_KEY
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
turns() { jq -rs '(map(select(.type=="result"))|.[-1]).num_turns // 0' "$1" 2>/dev/null; }
for pass in $(seq 1 "$MAXPASS"); do
  work=$(mktemp); g=0
  while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
    for r in 01 02 03; do d="$lane/${topic}-${r}"; t="$d/transcript.jsonl"
      if complete "$t" && [ "$(turns "$t")" -gt 15 ]; then g=$((g+1))
      else rm -rf "$d"; echo "$lane $topic $r" >> "$work"; fi
    done
  done < "$lane/prompts.tsv"
  n=$(wc -l < "$work" | tr -d ' ')
  echo "[heal $lane] pass $pass: $g/39 genuine, re-running $n"
  [ "$n" -eq 0 ] && { rm -f "$work"; break; }
  xargs -P "$PAR" -I{} bash -c './run_cell.sh {}' < "$work"
  rm -f "$work"
done
echo "[heal $lane] grading genuine successes"
env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY bash "$PWD/../grading/grade_all.sh" "$PWD" claude-opus-4-8 4
echo "[heal-$lane-done]"
