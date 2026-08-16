#!/usr/bin/env bash
# Watch the final 2 opus-5 ripples cells to 30/30. Baselines seen terminal lines,
# reports new ones, exits at 30 complete or pool-dead.
cd "$(dirname "$0")" || exit 1
mkdir -p .monrf_seen
grep -nE '^\[(ok  |fail|WRONGMODEL)\].*opus-5/ripples-' parallel.log 2>/dev/null | cut -d: -f1 | while read -r ln; do : > ".monrf_seen/$ln"; done
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
scount() { local n=0; while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
  for rep in 01 02 03; do complete "opus-5/${topic}-${rep}/transcript.jsonl" && n=$((n+1)); done
  done < opus-5/prompts.tsv; echo "$n"; }
echo "[rf-monitor] finishing opus-5 ripples-02/03 -> 30/30"
while true; do
  grep -nE '^\[(ok  |fail|WRONGMODEL)\].*opus-5/ripples-' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".monrf_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"; done
  s=$(scount)
  if [ "$s" -ge 30 ]; then echo "[opus5-30] all 30 opus-5 runs complete"; break; fi
  if ! pgrep -f run_opus5.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[rf-ended] pool stopped, $s/30 done"; break; fi
  sleep 30
done
