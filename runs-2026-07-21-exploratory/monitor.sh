#!/usr/bin/env bash
# Emit one line per completed run across the three model lanes, plus a final
# all-done line. No set -e, reads from files (not pipes) so dedup markers persist.
cd "$(dirname "$0")" || exit 1
mkdir -p .monitor_seen
echo "[monitor] up, watching fable-5 opus-4-8 sonnet-5"
while true; do
  for model in fable-5 opus-4-8 sonnet-5; do
    f="$model/run_log.tsv"
    [ -f "$f" ] || continue
    while IFS=$'\t' read -r ts topic rep status cost turns dur; do
      [ "$ts" = "timestamp" ] && continue
      [ -z "$topic" ] && continue
      marker=".monitor_seen/${model}_${topic}_${rep}"
      [ -f "$marker" ] && continue
      : > "$marker"
      if [ "$status" = "ok" ]; then
        printf '[ok]   %-9s %s-%s  $%s  %s turns\n' "$model" "$topic" "$rep" "${cost:-0}" "${turns:-?}"
      else
        printf '[FAIL] %-9s %s-%s  status=%s\n' "$model" "$topic" "$rep" "$status"
      fi
    done < "$f"
  done
  if [ -f progress.log ] && grep -q "all-done" progress.log 2>/dev/null; then
    echo "[all-done] sweep finished"
    break
  fi
  sleep 30
done
