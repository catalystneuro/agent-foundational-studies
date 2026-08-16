#!/usr/bin/env bash
# Watch the opus-5 retry. Baselines existing opus-5 terminal lines so the storm's
# failures are not re-reported; then reports only new terminal lines. A cluster of
# fast [fail]s means the rate limit is still too tight even at PAR=3.
cd "$(dirname "$0")" || exit 1
mkdir -p .mon5r_seen
grep -nE '^\[(ok  |fail|WRONGMODEL)\].*opus-5/' parallel.log 2>/dev/null | cut -d: -f1 | while read -r ln; do : > ".mon5r_seen/$ln"; done
echo "[opus5-retry] watching the 20 retried opus-5 runs at PAR=3"
scount() {
  local n=0
  while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
    for rep in 01 02 03; do t="opus-5/${topic}-${rep}/transcript.jsonl"
      [ -s "$t" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$t" >/dev/null 2>&1 && n=$((n+1)); done
  done < opus-5/prompts.tsv; echo "$n"
}
while true; do
  grep -nE '^\[(ok  |fail|WRONGMODEL)\].*opus-5/' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".mon5r_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"
  done
  s=$(scount)
  if [ "$s" -ge 30 ]; then echo "[opus5-complete] 30/30 opus-5 done"; break; fi
  if ! pgrep -f run_opus5.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "[opus5-retry-ended] pool stopped, $s/30 done"; break
  fi
  sleep 30
done
