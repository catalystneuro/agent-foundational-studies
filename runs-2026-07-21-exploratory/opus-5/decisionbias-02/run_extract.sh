#!/bin/bash
# Supervisor for 02_extract.py.
#
# Streaming ~25 GB of NWB from the archive occasionally leaves remfile waiting
# on a request that never returns.  02_extract.py skips sessions it has already
# written, so the cheapest fix is to notice that no bytes have arrived for a
# while and start it again.
CACHE=/tmp/remfile_cache_ibl
STALL_SECONDS=180

while [ "$(ls session_data/*.npz 2>/dev/null | wc -l | tr -d ' ')" -lt 20 ]; do
  python 02_extract.py >> extract.log 2>&1 &
  PID=$!
  last=$(du -sm $CACHE 2>/dev/null | cut -f1)
  while kill -0 $PID 2>/dev/null; do
    sleep $STALL_SECONDS
    cur=$(du -sm $CACHE 2>/dev/null | cut -f1)
    n=$(ls session_data/*.npz 2>/dev/null | wc -l | tr -d ' ')
    echo "$(date +%H:%M:%S)  $n/20 sessions, cache ${cur} MB"
    if [ "${cur:-0}" -le "${last:-0}" ]; then
      echo "no progress in ${STALL_SECONDS}s, restarting the extractor"
      kill -9 $PID 2>/dev/null
      break
    fi
    last=$cur
  done
  wait $PID 2>/dev/null
done
echo "SUPERVISOR_DONE $(ls session_data/*.npz | wc -l) sessions"
