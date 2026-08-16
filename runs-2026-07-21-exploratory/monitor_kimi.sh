#!/usr/bin/env bash
# Watch the kimi-k3-max lane (13 problems x 3 = 39). bash 3.2 safe.
cd "$(dirname "$0")" || exit 1
mkdir -p .monkimi_seen
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[kimi] watching 39 runs (kimi-k3-max via OpenRouter/DeepInfra)"
while true; do
  d=0
  while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
    for r in 01 02 03; do
      t="kimi-k3-max/${topic}-${r}/transcript.jsonl"; mk=".monkimi_seen/${topic}_${r}"
      if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] kimi-k3-max/${topic}-${r}"; }
      elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and (($x.subtype!="success") or ($x.is_error==true))' "$t" >/dev/null 2>&1; then
        reason=$(jq -rs '(map(select(.type=="result"))|.[-1]) | .terminal_reason // .subtype' "$t" 2>/dev/null)
        d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] kimi-k3-max/${topic}-${r} ($reason)"; }
      fi
    done
  done < kimi-k3-max/prompts.tsv
  if [ "$d" -ge 39 ]; then echo "[kimi-done] all 39 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f 'claude -p' 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[kimi-ended] pool stopped, $d/39 terminal"; break; fi
  sleep 45
done
