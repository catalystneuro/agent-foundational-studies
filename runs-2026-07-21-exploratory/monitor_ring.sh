#!/usr/bin/env bash
# Watch opus-5 ringattractor pilot (3 reps). bash 3.2 safe.
cd "$(dirname "$0")" || exit 1
mkdir -p .monring_seen
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[ring] watching opus-5 ringattractor pilot (3 reps)"
while true; do
  d=0
  for r in 01 02 03; do
    t="opus-5/ringattractor-$r/transcript.jsonl"; mk=".monring_seen/$r"
    if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] opus-5/ringattractor-$r"; }
    elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and ($x.subtype!="success")' "$t" >/dev/null 2>&1; then
      d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] opus-5/ringattractor-$r ($(jq -rs '(map(select(.type=="result"))|.[-1]).terminal_reason // "?"' "$t" 2>/dev/null))"; }
    fi
  done
  if [ "$d" -ge 3 ]; then echo "[ring-done] all 3 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[ring-ended] $d/3 terminal"; break; fi
  sleep 30
done
