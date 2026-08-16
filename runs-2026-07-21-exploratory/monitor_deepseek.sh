#!/usr/bin/env bash
# Watch the deepseek-v4-flash lane (13 problems x 3 = 39). bash 3.2 safe.
cd "$(dirname "$0")" || exit 1
mkdir -p .monds_seen
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[deepseek] watching 39 runs (deepseek-v4-flash via OpenRouter/DeepInfra)"
while true; do
  d=0
  while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
    for r in 01 02 03; do
      t="deepseek-v4-flash/${topic}-${r}/transcript.jsonl"; mk=".monds_seen/${topic}_${r}"
      if complete "$t"; then d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] deepseek-v4-flash/${topic}-${r}"; }
      elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and (($x.subtype!="success") or ($x.is_error==true))' "$t" >/dev/null 2>&1; then
        reason=$(jq -rs '(map(select(.type=="result"))|.[-1]) | .terminal_reason // .subtype' "$t" 2>/dev/null)
        d=$((d+1)); [ -f "$mk" ] || { : > "$mk"; echo "[fail] deepseek-v4-flash/${topic}-${r} ($reason)"; }
      fi
    done
  done < deepseek-v4-flash/prompts.tsv
  if [ "$d" -ge 39 ]; then echo "[deepseek-done] all 39 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f 'claude -p' 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[deepseek-ended] pool stopped, $d/39 terminal"; break; fi
  sleep 45
done
