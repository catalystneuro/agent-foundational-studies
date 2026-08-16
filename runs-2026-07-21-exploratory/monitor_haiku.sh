#!/usr/bin/env bash
# Report only haiku-4-5 terminal lines from parallel.log; exit when all 15 are
# terminal or the pool process is gone. Scoped to haiku so it ignores the stale
# [parallel-done] from the earlier analysis pool.
cd "$(dirname "$0")" || exit 1
mkdir -p .monh_seen
echo "[haiku-monitor] watching haiku-4-5 runs (flags [WRONGMODEL] on fallback)"
while true; do
  grep -nE '^\[(ok  |fail|WRONGMODEL)\].*haiku-4-5' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".monh_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"
  done
  done_ct=$(ls .monh_seen 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done_ct" -ge 15 ]; then echo "[haiku-done] all 15 terminal"; break; fi
  if ! pgrep -f run_parallel.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "[haiku-pool-ended] pool stopped, $done_ct/15 terminal"; break
  fi
  sleep 20
done
