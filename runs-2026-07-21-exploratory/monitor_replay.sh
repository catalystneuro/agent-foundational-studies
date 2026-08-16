#!/usr/bin/env bash
# Watch the opus-5 replay pilot (3 reps). bash 3.2 safe. Reports each cell as it lands
# ok/fail, exits when all 3 terminal or pool dies.
cd "$(dirname "$0")" || exit 1
mkdir -p .monreplay_seen
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[replay] watching opus-5 replay pilot (3 reps)"
while true; do
  d=0
  for r in 01 02 03; do
    t="opus-5/replay-$r/transcript.jsonl"; mk=".monreplay_seen/$r"
    if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] opus-5/replay-$r"; }
    elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and ($x.subtype!="success")' "$t" >/dev/null 2>&1; then
      d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] opus-5/replay-$r ($(jq -rs '(map(select(.type=="result"))|.[-1]).terminal_reason // "?"' "$t" 2>/dev/null))"; }
    fi
  done
  if [ "$d" -ge 3 ]; then echo "[replay-done] all 3 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[replay-pool-ended] $d/3 terminal"; break; fi
  sleep 30
done
