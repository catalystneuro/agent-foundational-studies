#!/usr/bin/env bash
# Watch the 3 relaunched opus-4-8 harness-failure cells (bash 3.2 safe: marker files,
# no associative arrays). Report each as it lands; exit when all 3 terminal or pool dies.
cd "$(dirname "$0")" || exit 1
cells="entrainment-01 precession-03 headdirection-02"
mkdir -p .monhfix_seen
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
echo "[hfix] watching 3 relaunched opus-4-8 cells"
while true; do
  done_ct=0
  for c in $cells; do
    t="opus-4-8/$c/transcript.jsonl"; mk=".monhfix_seen/$c"
    if complete "$t"; then
      done_ct=$((done_ct+1))
      [ -f "$mk" ] || { : > "$mk"; echo "[ok  ] opus-4-8/$c"; }
    elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x | ($x!=null) and ($x.subtype!="success")' "$t" >/dev/null 2>&1; then
      done_ct=$((done_ct+1))
      [ -f "$mk" ] || { : > "$mk"; echo "[fail] opus-4-8/$c"; }
    fi
  done
  if [ "$done_ct" -ge 3 ]; then echo "[hfix-done] all 3 terminal"; break; fi
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[hfix-pool-ended] $done_ct/3 terminal"; break; fi
  sleep 30
done
