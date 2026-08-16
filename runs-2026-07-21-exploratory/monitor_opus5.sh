#!/usr/bin/env bash
# Report only opus-5 terminal lines from parallel.log; exit when all 30 are
# terminal or the pool stops. Flags [WRONGMODEL] on a served/requested mismatch.
cd "$(dirname "$0")" || exit 1
mkdir -p .mon5_seen
echo "[opus5-monitor] watching opus-5 runs (flags [WRONGMODEL] on fallback)"
while true; do
  grep -nE '^\[(ok  |fail|WRONGMODEL)\].*opus-5/' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".mon5_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"
  done
  done_ct=$(ls .mon5_seen 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done_ct" -ge 30 ]; then echo "[opus5-done] all 30 terminal"; break; fi
  if ! pgrep -f run_opus5.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "[opus5-pool-ended] pool stopped, $done_ct/30 terminal"; break
  fi
  sleep 20
done
