#!/usr/bin/env bash
# Watch replay on opus-4-8/sonnet-5/haiku-4-5 (9 runs). bash 3.2 safe. Report each
# ok/fail as it lands; exit at 9 terminal or pool dead.
cd "$(dirname "$0")" || exit 1
mkdir -p .monrl_seen
lanes="opus-4-8 sonnet-5 haiku-4-5"
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[replay-lanes] watching 9 lower-lane replay runs"
while true; do
  d=0
  for lane in $lanes; do for r in 01 02 03; do
    t="$lane/replay-$r/transcript.jsonl"; mk=".monrl_seen/${lane}_${r}"
    if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] $lane/replay-$r"; }
    elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and ($x.subtype!="success")' "$t" >/dev/null 2>&1; then
      d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] $lane/replay-$r ($(jq -rs '(map(select(.type=="result"))|.[-1]).terminal_reason // "?"' "$t" 2>/dev/null))"; }
    fi
  done; done
  if [ "$d" -ge 9 ]; then echo "[replay-lanes-done] all 9 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[replay-lanes-ended] $d/9 terminal"; break; fi
  sleep 30
done
