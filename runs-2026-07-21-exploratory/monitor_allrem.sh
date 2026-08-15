#!/usr/bin/env bash
# Watch the consolidated pool. Report [fail]/[WRONGMODEL] immediately (actionable),
# emit a progress count every 10 new completions, and exit when the pool finishes.
cd "$(dirname "$0")" || exit 1
mkdir -p .monrem_seen
base_ok=$(grep -cE '^\[ok  \]' parallel.log 2>/dev/null || echo 0)   # ignore old oks
last_report=$base_ok
echo "[allrem-monitor] watching consolidated pool; failures reported live, oks batched"
while true; do
  grep -nE '^\[(fail|WRONGMODEL)\]' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
    mk=".monrem_seen/$ln"; [ -f "$mk" ] && continue; : > "$mk"; echo "$rest"
  done
  ok=$(grep -cE '^\[ok  \]' parallel.log 2>/dev/null || echo 0)
  if [ $((ok - last_report)) -ge 10 ]; then
    echo "[progress] $((ok - base_ok)) new completions this pool"; last_report=$ok
  fi
  if grep -q "allrem-done" parallel.log 2>/dev/null; then
    echo "[allrem-done] $((ok - base_ok)) completed this pool"; break
  fi
  if ! pgrep -f run_all_remaining.sh >/dev/null && [ "$(pgrep -f append-system-prompt 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "[allrem-pool-ended] pool stopped; $((ok - base_ok)) completed this pool"; break
  fi
  sleep 30
done
