"""Recompute the split-ensemble agreement statistic on cached results.

The first pass scored agreement with a circular correlation coefficient, which is
unstable when the decoded angles are near-uniform; this replaces it with the mean
resultant length of the angular difference.
"""
import warnings, glob, sys
warnings.filterwarnings("ignore")
import numpy as np
import ringanalysis as ra

files = sys.argv[1:] or sorted(glob.glob("cache_*.npy"))
for fn in files:
    r = np.load(fn, allow_pickle=True).item()
    for st in ["wake", "REM", "nREM"]:
        d = r[st]
        rng = np.random.default_rng(0)
        da, db = ra.split_half_decode(d["counts"], r["tc_hd"], r["bin"],
                                      np.random.default_rng(0))
        d["split_coh"] = ra.agreement(da, db)
        null = []
        for _ in range(20):
            cs = ra.shift_shuffle(d["counts"], rng)
            na, nb = ra.split_half_decode(cs, r["tc_hd"], r["bin"],
                                          np.random.default_rng(0))
            null.append(ra.agreement(na, nb))
        d["null_coh"] = np.array(null)
        print(f"{r['sid']} {st}: agreement={d['split_coh']:.2f} "
              f"(null {np.mean(null):.3f})", flush=True)
    np.save(fn, r, allow_pickle=True)
