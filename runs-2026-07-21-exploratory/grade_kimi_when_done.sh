#!/usr/bin/env bash
# Block until the kimi-k3-max sweep reaches all-terminal (pass or fail) or the
# pool dies, then grade the genuine successes with the subscription judge (opus-4-8).
# Grading must NOT go through the OpenRouter proxy, so clear those vars explicitly.
cd "$(dirname "$0")" || exit 1
complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }
terminal_count() {
  local n=0
  while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
    for r in 01 02 03; do t="kimi-k3-max/${topic}-${r}/transcript.jsonl"
      if complete "$t"; then n=$((n+1))
      elif [ -s "$t" ] && jq -e -s '(map(select(.type=="result"))|.[-1]) as $x|($x!=null) and (($x.subtype!="success") or ($x.is_error==true))' "$t" >/dev/null 2>&1; then n=$((n+1)); fi
    done
  done < kimi-k3-max/prompts.tsv; echo "$n"
}
while true; do
  tc=$(terminal_count)
  [ "$tc" -ge 39 ] && { echo "[grade] all 39 terminal"; break; }
  if ! pgrep -f run_cell.sh >/dev/null && [ "$(pgrep -f 'claude -p' 2>/dev/null|wc -l|tr -d ' ')" -eq 0 ]; then
    echo "[grade] pool ended at $tc/39 terminal"; break; fi
  sleep 30
done
ok=0; while IFS=$'\t' read -r topic _; do [ -n "$topic" ] || continue
  for r in 01 02 03; do complete "kimi-k3-max/${topic}-${r}/transcript.jsonl" && ok=$((ok+1)); done
done < kimi-k3-max/prompts.tsv
echo "[grade] $ok genuine successes to grade"
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY
bash "$PWD/../grading/grade_all.sh" "$PWD" claude-opus-4-8 4
echo "[grade-done] deepseek grading complete ($ok successes graded)"
