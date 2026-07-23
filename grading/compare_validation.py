#!/usr/bin/env python3
"""Score human-vs-LLM-judge agreement after the blind validation sheet is filled.

Reads validation_sheet.csv (filled by the human grader) and .validation_key.csv
(the hidden LLM scores), then reports per-axis agreement, verdict/pass/fabrication
agreement, and the runs where the two disagree most. This is the JudgeEval-style
check from the plan: it tells you whether the LLM judge can be trusted, and on
which axes it can't.

Usage: python grading/compare_validation.py
"""
import csv, pathlib
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
AX = [("ax1_dataset", "llm_dataset"), ("ax2_handling", "llm_handling"),
      ("ax3_analysis", "llm_analysis"), ("ax4_correctness", "llm_correctness"),
      ("ax5_figures", "llm_figures"), ("ax6_honesty", "llm_honesty")]


def num(x):
    try: return int(float(x))
    except Exception: return None


def main():
    sheet = {r["blind_id"]: r for r in csv.DictReader(open(REPO / "validation_sheet.csv"))}
    key = {r["blind_id"]: r for r in csv.DictReader(open(REPO / ".validation_key.csv"))}
    ids = [b for b in key if b in sheet]
    filled = [b for b in ids if any(sheet[b].get(hc, "").strip() for hc, _ in AX)]
    if not filled:
        raise SystemExit("validation_sheet.csv has no human scores yet; grade it first.")
    print(f"comparing {len(filled)} human-graded runs against the LLM judge\n")

    print("per-axis agreement (human vs judge):")
    print(f"  {'axis':<14}{'n':>3}{'exact%':>8}{'within1%':>9}{'mean|Δ|':>9}{'bias(h-j)':>11}")
    for hc, kc in AX:
        pairs = [(num(sheet[b][hc]), num(key[b][kc])) for b in filled]
        pairs = [(h, j) for h, j in pairs if h is not None and j is not None]
        if not pairs: continue
        n = len(pairs)
        exact = sum(h == j for h, j in pairs) / n * 100
        within1 = sum(abs(h - j) <= 1 for h, j in pairs) / n * 100
        mad = sum(abs(h - j) for h, j in pairs) / n
        bias = sum(h - j for h, j in pairs) / n
        print(f"  {hc.split('_')[1]:<14}{n:>3}{exact:>7.0f}%{within1:>8.0f}%{mad:>9.2f}{bias:>+11.2f}")

    # verdict + pass + fabrication
    def agree(hkey, kkey, norm=lambda x: (x or "").strip().lower()):
        pairs = [(norm(sheet[b].get(hkey, "")), norm(key[b].get(kkey, ""))) for b in filled]
        pairs = [(h, j) for h, j in pairs if h]
        if not pairs: return None
        return sum(h == j for h, j in pairs) / len(pairs) * 100, len(pairs)
    print("\ncategorical agreement:")
    for label, hk, kk in [("verdict", "ax7_verdict", "llm_verdict"),
                          ("fabrication", "fabrication", "llm_fabrication")]:
        a = agree(hk, kk)
        if a: print(f"  {label:<12} {a[0]:.0f}%  (n={a[1]})")

    print("\nbiggest disagreements (sorted by summed axis gap):")
    rows = []
    for b in filled:
        gap = sum(abs((num(sheet[b][hc]) or 0) - (num(key[b][kc]) or 0)) for hc, kc in AX
                  if num(sheet[b][hc]) is not None and num(key[b][kc]) is not None)
        rows.append((gap, b))
    for gap, b in sorted(rows, reverse=True)[:6]:
        k = key[b]
        print(f"  {b} [{k['served_model']} {sheet[b]['problem']}] gap={gap} "
              f"human_verdict={sheet[b].get('ax7_verdict','?')} judge_verdict={k['llm_verdict']} "
              f"({k['run']})")


if __name__ == "__main__":
    main()
