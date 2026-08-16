#!/usr/bin/env bash
# Report the lower-model new-problem runs (opus-4-8/sonnet-5/haiku-4-5 on the six
# new problems). Exit at 54 terminal or when the pool stops. Flags [WRONGMODEL].
cd "$(dirname "$0")" || exit 1
mkdir -p .monnew_seen
pat='^\[(ok  |fail|WRONGMODEL)\].*(opus-4-8|sonnet-5|haiku-4-5)/(entrainment|precession|strf|grid|headdirection|ripples)-'
echo "[newq-monitor] watching lower models on the 6 new problems (54 runs)"
while true; do
  grep -nE "$pat" parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".monnew_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"
  done
  done_ct=$(ls .monnew_seen 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done_ct" -ge 54 ]; then echo "[newq-done] all 54 terminal"; break; fi
  if ! pgrep -f run_newproblems.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "[newq-pool-ended] pool stopped, $done_ct/54 terminal"; break
  fi
  sleep 25
done
