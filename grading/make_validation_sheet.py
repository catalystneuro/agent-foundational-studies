#!/usr/bin/env python3
"""Build a blinded human-validation sample from the LLM-judged exploratory runs.

Selects a stratified sample that targets the judge's most contested calls (all
Sonnet fabrication flags, several Haiku rejects, some Opus accepts and revises),
copies each run's gradeable artifacts into an anonymized folder with no model
identity, and writes a blank grading sheet plus a hidden key.

The grader (Ben) fills validation_sheet.csv from the folders in validation_sample/,
blind to model. compare_validation.py then scores human-vs-judge agreement per axis
against .validation_key.csv.

Usage: python grading/make_validation_sheet.py [--salt SALT]
Deterministic given the salt, so the same sample and blind IDs regenerate.
"""
import argparse, csv, hashlib, json, pathlib, shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "runs-2026-07-21-exploratory"
# fable-5 lane served Opus 4.8 (gated fallback); label it as a second Opus sample.
SERVED = {"fable-5": "opus-4-8-B", "opus-4-8": "opus-4-8-A",
          "sonnet-5": "sonnet-5", "haiku-4-5": "haiku-4-5"}
GRADE_COLS = ["ax1_dataset", "ax2_handling", "ax3_analysis", "ax4_correctness",
              "ax5_figures", "ax6_honesty", "ax7_verdict", "fabrication",
              "confidence", "notes"]


def load():
    runs = []
    for gj in ROOT.glob("*/*/grade.json"):
        model = gj.parts[-3]
        try:
            g = json.loads(gj.read_text())
        except Exception:
            continue
        runs.append({"dir": gj.parent, "model": model, "served": SERVED[model],
                     "problem": gj.parent.name.rsplit("-", 1)[0], "g": g})
    return runs


def pick(runs):
    """Stratified, deterministic (sorted) selection targeting contested calls."""
    def is_fab(r): return str(r["g"].get("fabrication_flag")).lower() == "true"
    def verdict(r): return r["g"].get("verdict")
    key = lambda r: str(r["dir"])
    sel, seen = [], set()
    def take(cands, n):
        for r in sorted(cands, key=key):
            if len(_this) >= n: break
            if key(r) in seen: continue
            seen.add(key(r)); sel.append(r); _this.append(r)
    # Sonnet: all fabrication flags + 2 non-fab passes
    _this = []; take([r for r in runs if r["served"] == "sonnet-5" and is_fab(r)], 3)
    take([r for r in runs if r["served"] == "sonnet-5" and verdict(r) == "revise" and not is_fab(r)], 5)
    # Haiku: 5 rejects across distinct problems
    _this = []; probs = set()
    for r in sorted([r for r in runs if r["served"] == "haiku-4-5"], key=key):
        if len(_this) >= 5: break
        if r["problem"] in probs or key(r) in seen: continue
        probs.add(r["problem"]); seen.add(key(r)); sel.append(r); _this.append(r)
    # Opus A: 2 accepts + 2 revises
    _this = []; take([r for r in runs if r["served"] == "opus-4-8-A" and verdict(r) == "accept"], 2)
    take([r for r in runs if r["served"] == "opus-4-8-A" and verdict(r) == "revise"], 4)
    # Opus B: 3 accepts
    _this = []; take([r for r in runs if r["served"] == "opus-4-8-B" and verdict(r) == "accept"], 3)
    return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--salt", default="afs-validation-v1")
    args = ap.parse_args()
    runs = load()
    if not runs:
        raise SystemExit("no graded runs found under " + str(ROOT))
    sel = pick(runs)
    bid = lambda r: "V" + hashlib.sha256((args.salt + "|" + str(r["dir"])).encode()).hexdigest()[:6].upper()
    sample_dir = REPO / "validation_sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir()
    # order the sheet by blind id so presentation is decorrelated from model
    sel_sorted = sorted(sel, key=bid)
    with open(REPO / "validation_sheet.csv", "w", newline="") as sf, \
         open(REPO / ".validation_key.csv", "w", newline="") as kf:
        sw = csv.writer(sf); sw.writerow(["blind_id", "problem"] + GRADE_COLS)
        kw = csv.writer(kf)
        kw.writerow(["blind_id", "run", "served_model", "llm_verdict", "llm_pass",
                     "llm_fabrication", "llm_dataset", "llm_handling", "llm_analysis",
                     "llm_correctness", "llm_figures", "llm_honesty"])
        for r in sel_sorted:
            b = bid(r); dest = sample_dir / b; dest.mkdir()
            # copy only gradeable artifacts, nothing that names the model
            for f in r["dir"].iterdir():
                if f.suffix in {".png"} or f.name == "README.md" or f.suffix in {".py", ".ipynb", ".csv"}:
                    if f.name in {"grade.json"}:
                        continue
                    if f.name.startswith("grade."):
                        continue
                    shutil.copy2(f, dest / f.name)
            sw.writerow([b, r["problem"]] + [""] * len(GRADE_COLS))
            ax = r["g"].get("axes", {})
            kw.writerow([b, str(r["dir"].relative_to(REPO)), r["served"],
                         r["g"].get("verdict"), r["g"].get("pass"), r["g"].get("fabrication_flag"),
                         ax.get("dataset"), ax.get("handling"), ax.get("analysis"),
                         ax.get("correctness"), ax.get("figures"), ax.get("honesty")])
    # composition report
    from collections import Counter
    comp = Counter(r["served"] for r in sel_sorted)
    print(f"validation sample: {len(sel_sorted)} runs -> validation_sample/, validation_sheet.csv")
    print("  composition:", dict(comp))
    print("  key (gitignored): .validation_key.csv")


if __name__ == "__main__":
    main()
