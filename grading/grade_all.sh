#!/usr/bin/env bash
# Grade every genuinely-complete run under a root, in a pool, then collect grades.csv.
# Usage: grade_all.sh <root> [judge_model] [PAR]
#   root examples: runs-2026-04-27   runs-2026-07-21-exploratory
set -u
GRADING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${1:?usage: grade_all.sh <root> [judge_model] [PAR]}"
judge="${2:-claude-opus-4-8}"
N="${3:-4}"
root="$(cd "$root" && pwd)"

complete() { [ -s "$1" ] && jq -e -s 'map(select(.type=="result" and .subtype=="success" and (.is_error|not)))|length>0' "$1" >/dev/null 2>&1; }

# A run dir is any directory containing transcript.jsonl. Grade only genuine successes
# that are not already graded.
# bash 3.2 (macOS default) has no mapfile; write the worklist to a temp file.
worklist="$(mktemp)"
find "$root" -name transcript.jsonl -not -path '*/cache/*' -print | while read -r t; do
  d="$(dirname "$t")"
  complete "$t" || continue
  [ -f "$d/grade.json" ] && continue
  echo "$d"
done > "$worklist"

echo "grading $(wc -l < "$worklist" | tr -d ' ') runs with judge=$judge at concurrency=$N"
xargs -P "$N" -I{} bash "$GRADING_DIR/grade_run.sh" {} "$judge" < "$worklist"
rm -f "$worklist"

# Collect into grades.csv
out="$root/grades.csv"
python3 - "$root" "$out" <<'PY'
import json, sys, pathlib, csv
root, out = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = []
for gj in root.rglob("grade.json"):
    d = gj.parent
    try: g = json.loads(gj.read_text())
    except Exception: continue
    rel = d.relative_to(root)
    # infer model + problem from path where possible
    parts = rel.parts
    model = parts[0] if len(parts) > 1 else "opus-4-7"   # exploratory: <model>/<topic-rep>; april: <topic-rep>
    topic = d.name.rsplit("-", 1)[0]
    ax = g.get("axes", {})
    rows.append({
        "run": str(rel), "model": model, "problem": topic,
        "dandiset": g.get("dandiset"),
        "dataset": ax.get("dataset"), "handling": ax.get("handling"),
        "analysis": ax.get("analysis"), "correctness": ax.get("correctness"),
        "figures": ax.get("figures"), "honesty": ax.get("honesty"),
        "verdict": g.get("verdict"), "pass": g.get("pass"),
        "fabrication": g.get("fabrication_flag"), "confidence": g.get("confidence"),
    })
rows.sort(key=lambda r: (r["model"], r["problem"], r["run"]))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} graded)")
PY
