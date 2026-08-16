#!/usr/bin/env python3
"""Build a blinded grading sheet from a directory of completed runs.

Writes two files: a grading sheet with blinded run IDs in an order decorrelated
from topic, and the deblinding key. The key is gitignored so that a grader who
clones the repository does not receive it; regenerate it with the same salt when
the scores come back.

Graders are deliberately not blinded to the problem, because the rubric's
reference ranges are problem-specific. They are blinded to model, condition, and
repetition.

The salt matters. With a published salt and a known set of run names, the blind
IDs are reversible by brute force. For the pilot that is acceptable, since its
purpose is calibrating the rubric rather than producing headline numbers. For the
confirmatory sweep, have someone who is not grading choose the salt and withhold
it until scoring is complete:

    python make_grading_sheet.py --runs runs-2026-XX-XX --salt "$(openssl rand -hex 16)"
"""

import argparse
import csv
import hashlib
import pathlib

COLUMNS = [
    "blind_id", "problem",
    "ax1_dataset", "ax2_handling", "ax3_analysis",
    "ax4_correctness", "ax5_figures", "ax6_honesty", "ax7_verdict",
    "dandiset", "subject_session", "stated_any_limitation", "confidence", "notes",
]


def blind_id(salt: str, run: str) -> str:
    return "R" + hashlib.sha256(f"{salt}|{run}".encode()).hexdigest()[:6].upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs-2026-04-27", help="directory of run subdirectories")
    ap.add_argument("--salt", default="afs-pilot-v1", help="blinding salt; withhold for the real sweep")
    ap.add_argument("--out", default="grading_sheet_pilot.csv")
    ap.add_argument("--key", default=".blind_key_pilot.csv")
    args = ap.parse_args()

    root = pathlib.Path(args.runs)
    runs = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "cost.json").exists())
    if not runs:
        raise SystemExit(f"no completed runs found under {root}")

    ids = {r: blind_id(args.salt, r) for r in runs}
    # Sorting by blind ID decorrelates presentation order from topic and rep.
    order = sorted(runs, key=lambda r: ids[r])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in order:
            w.writerow([ids[r], r.split("-")[0]] + [""] * (len(COLUMNS) - 2))

    with open(args.key, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "run"])
        for r in order:
            w.writerow([ids[r], r])

    print(f"{len(runs)} runs -> {args.out}")
    print(f"key -> {args.key} (gitignored; graders must not open it)")


if __name__ == "__main__":
    main()
