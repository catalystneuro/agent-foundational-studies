#!/usr/bin/env bash
# Watch the final grade pass. Emit a progress count every 8 newly-written grade.json,
# and exit when grade_all.sh finishes (grades.csv rewritten) or the pool dies.
cd "$(dirname "$0")" || exit 1
base=$(find . -name grade.json | wc -l | tr -d ' ')
last=$base
echo "[gradepass] start: $base graded; target +34"
while true; do
  now=$(find . -name grade.json | wc -l | tr -d ' ')
  if [ $((now - last)) -ge 8 ]; then echo "[progress] $((now - base)) newly graded ($now total)"; last=$now; fi
  if ! pgrep -f grade_all.sh >/dev/null && [ "$(pgrep -f grade_run.sh 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[gradepass-done] $((now - base)) newly graded, $now total grade.json"; break
  fi
  sleep 20
done
