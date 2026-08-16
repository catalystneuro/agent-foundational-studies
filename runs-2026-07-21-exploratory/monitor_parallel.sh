#!/usr/bin/env bash
# Emit each completed/failed cell from the parallel pool, then a done line.
cd "$(dirname "$0")" || exit 1
mkdir -p .mon2_seen
echo "[monitor] parallel pool, watching parallel.log"
while true; do
  if [ -f parallel.log ]; then
    grep -nE '^\[(ok  |fail|WRONGMODEL)\]' parallel.log 2>/dev/null | while IFS=: read -r ln rest; do
      marker=".mon2_seen/$ln"
      [ -f "$marker" ] && continue
      : > "$marker"
      echo "$rest"
    done
  fi
  if [ -f parallel.log ] && grep -q "parallel-done" parallel.log 2>/dev/null; then
    echo "[parallel-done] pool finished"
    break
  fi
  sleep 20
done
