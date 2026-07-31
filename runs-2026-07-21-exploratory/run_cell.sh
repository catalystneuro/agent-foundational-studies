#!/usr/bin/env bash
# Run one (model-dir, topic, rep) cell, resume-safe and headless. Reuses the
# exact prompt, autonomy suffix, system addendum, model, and budget that the
# original run_all.sh harness used, so parallel-pool runs are identical in
# configuration to the sequential ones.
set -u
cd "$(dirname "$0")"
export MPLBACKEND=Agg

d="$1"; topic="$2"; rep="$3"
case "$d" in
  fable-5)   m=claude-fable-5 ;;
  opus-4-8)  m=claude-opus-4-8 ;;
  sonnet-5)  m=claude-sonnet-5 ;;
  haiku-4-5) m=claude-haiku-4-5 ;;
  opus-5)    m=claude-opus-5 ;;
  *) echo "[err ] unknown model dir $d" >&2; exit 2 ;;
esac

complete() {  # transcript -> 0 if a genuine success result exists
  [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1
}

rundir="$d/${topic}-${rep}"
t="$rundir/transcript.jsonl"
if complete "$t"; then echo "[skip] $d/$topic-$rep" >> parallel.log; exit 0; fi

prompt="$(awk -F'\t' -v k="$topic" '$1==k{print $2}' "$d/prompts.tsv")"
[ -n "$prompt" ] || { echo "[err ] no prompt for $topic in $d" >> parallel.log; exit 3; }
suffix="$(cat autonomy_suffix.txt)"

mkdir -p "$rundir"
echo "[run ] $d/$topic-$rep $(date -Iseconds)" >> parallel.log
( cd "$rundir" && claude -p "${prompt}${suffix}" \
    --model "$m" \
    --dangerously-skip-permissions \
    --append-system-prompt "$(cat ../SYSTEM_ADDENDUM.md)" \
    --output-format stream-json --verbose \
    --max-budget-usd 25 \
    > transcript.jsonl 2> stderr.log )

if complete "$t"; then
  cost=$(jq -s 'map(select(.type=="result"))|.[-1].total_cost_usd // 0' "$t" 2>/dev/null)
  # Guard against silent model fallback (e.g. gated models served as a different one).
  served="$(jq -rs '(map(select(.type=="result"))|.[-1].modelUsage // {}) | to_entries | map(select(.value.outputTokens>500) | .key) | join(",")' "$t" 2>/dev/null)"
  case "$served" in
    *"$m"*) echo "[ok  ] $d/$topic-$rep \$$cost $(date -Iseconds)" >> parallel.log ;;
    *) echo "[WRONGMODEL] $d/$topic-$rep requested=$m served=$served \$$cost $(date -Iseconds)" >> parallel.log ;;
  esac
else
  echo "[fail] $d/$topic-$rep $(date -Iseconds)" >> parallel.log
fi
