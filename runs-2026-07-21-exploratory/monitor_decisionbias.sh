#!/usr/bin/env bash
# Watch decisionbias across all 4 lanes (12 runs). bash 3.2 safe.
cd "$(dirname "$0")" || exit 1
mkdir -p .mondb_seen
lanes="opus-5 opus-4-8 sonnet-5 haiku-4-5"
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[decisionbias] watching 12 runs (4 lanes x 3)"
while true; do
  d=0
  for lane in $lanes; do for r in 01 02 03; do
    t="$lane/decisionbias-$r/transcript.jsonl"; mk=".mondb_seen/${lane}_${r}"
    if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] $lane/decisionbias-$r"; }
    elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and ($x.subtype!="success")' "$t" >/dev/null 2>&1; then
      d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] $lane/decisionbias-$r ($(jq -rs '(map(select(.type=="result"))|.[-1]).terminal_reason // "?"' "$t" 2>/dev/null))"; }
    fi
  done; done
  if [ "$d" -ge 12 ]; then echo "[decisionbias-done] all 12 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[decisionbias-ended] $d/12 terminal"; break; fi
  sleep 30
done
