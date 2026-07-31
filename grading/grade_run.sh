#!/usr/bin/env bash
# Grade one run directory with an LLM judge. Writes grade.json into the run dir.
# Layout-agnostic: works on runs-2026-04-27/<topic>-NN and
# runs-2026-07-21-exploratory/<model>/<topic>-NN alike.
#
# Usage: grade_run.sh <run_dir> [judge_model]
set -u
GRADING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$GRADING_DIR/.." && pwd)"

run_dir="${1:?usage: grade_run.sh <run_dir> [judge_model]}"
judge="${2:-claude-opus-4-8}"
run_dir="$(cd "$run_dir" && pwd)"

# System prompt = rubric (single source of truth) + reference ranges + grader procedure.
sys="$(cat "$REPO/RUBRIC.md" "$GRADING_DIR/dataset_reference.md" "$GRADING_DIR/grader_instructions.md")"

prompt="Grade the single run in the directory: ${run_dir}
Read the README, the analysis code, and every figure (open the .png files with Read),
then score all seven rubric axes and emit the JSON object as instructed. The judge must
not be told, and must not infer, which model produced the run."

# Judge runs read-only. No --dangerously-skip-permissions is needed for read-only tools,
# but the batch driver runs headless, so allow the tools it needs without prompts.
out="$(cd "$run_dir" && claude -p "$prompt" \
  --model "$judge" \
  --dangerously-skip-permissions \
  --append-system-prompt "$sys" \
  --output-format stream-json --verbose 2>"$run_dir/grade.stderr.log")"

# Pull the final assistant/result text and extract the last JSON object from it.
echo "$out" > "$run_dir/grade.transcript.jsonl"
python3 - "$run_dir" <<'PY'
import json, re, sys, pathlib
rd = pathlib.Path(sys.argv[1])
text = ""
for line in (rd / "grade.transcript.jsonl").read_text().splitlines():
    try: e = json.loads(line)
    except Exception: continue
    if e.get("type") == "result":
        text = e.get("result", "") or text
    elif e.get("type") == "assistant":
        for c in e.get("message", {}).get("content", []) or []:
            if c.get("type") == "text":
                text = c["text"]
# grab the last fenced or bare JSON object
blocks = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S)
obj = None
for b in reversed(blocks):
    try:
        obj = json.loads(b); break
    except Exception:
        continue
if obj is None:
    print("NO_JSON"); sys.exit(1)
(rd / "grade.json").write_text(json.dumps(obj, indent=2))
ax = obj.get("axes", {})
print(f"graded {rd.name}: verdict={obj.get('verdict')} pass={obj.get('pass')} "
      f"axes={[ax.get(k) for k in ('dataset','handling','analysis','correctness','figures','honesty')]}")
PY
