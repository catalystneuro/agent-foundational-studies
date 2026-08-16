"""Run grid cell analysis across all sessions of DANDI 000582 (parallel per-session)."""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_lib import analyze_session

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)


def session_tag(row):
    base = os.path.basename(row["path"]).replace("_behavior+ecephys.nwb", "")
    return base


def work(row):
    tag = session_tag(row)
    out = os.path.join(OUTDIR, tag + ".pkl")
    if os.path.exists(out):
        return tag, "cached"
    recs = analyze_session(row)
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    return tag, f"{len(recs)} units"


if __name__ == "__main__":
    survey = json.load(open("session_survey.json"))
    rows = [r for r in survey if "error" not in r and r["n_units"] > 0 and r["has_pos"]]
    print(f"{len(rows)} sessions to process")
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(work, r): r for r in rows}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="sessions"):
            tag, msg = fut.result()
            tqdm.write(f"{tag}: {msg}")
    print("done")
