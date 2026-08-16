"""Run the auditory-nerve tuning analysis over a sample of fibres and cache the results."""
import pickle
import sys

import numpy as np
from tqdm import tqdm

import anf_analysis as anf
import dandi_io as dio

N_FIBRES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = sys.argv[2] if len(sys.argv) > 2 else "results_anf.pkl"

files = dio.list_assets("001262", "0.241205.0959")
# spread the sample evenly over the (alphabetically ordered, subject-grouped) file list
sel = files[:: max(1, len(files) // N_FIBRES)][:N_FIBRES]
print(f"{len(files)} fibre files in DANDI:001262; analysing {len(sel)}")

results, failed = [], []
for path, url in tqdm(sel, desc="fibres"):
    r = anf.analyze_fibre(url)
    if r is None:
        failed.append(path)
        continue
    r["path"] = path
    results.append(r)

print(f"{len(results)} fibres analysed, {len(failed)} files without usable tone sweeps")
grid = [r for r in results if np.isfinite(r.get("cf", np.nan))]
iso = [r for r in results if np.isfinite(r.get("iso_bf", np.nan))]
print(f"{len(grid)} fibres with a frequency x level grid and a measurable threshold curve")
print("  CF range (Hz): %.0f - %.0f" % (min(r["cf"] for r in grid), max(r["cf"] for r in grid)))
q = [r["q10"] for r in grid if np.isfinite(r["q10"])]
print("  Q10: median %.2f  n=%d" % (np.median(q), len(q)))
print(f"{len(iso)} fibres with fixed-level frequency sweeps")
sig = [r for r in iso if r["iso_p"] < 0.01]
print(f"  {len(sig)} frequency-selective at p < 0.01")
bw = [r["iso_bw_octaves"] for r in iso if np.isfinite(r["iso_bw_octaves"])]
print("  half-max bandwidth: median %.2f octaves (n=%d measurable)" % (np.median(bw), len(bw)))
print("subjects:", sorted({r["subject"] for r in results}))
print("ages (days):", sorted({r["age_days"] for r in results}))

with open(OUT, "wb") as fh:
    pickle.dump(results, fh)
print("wrote", OUT)
