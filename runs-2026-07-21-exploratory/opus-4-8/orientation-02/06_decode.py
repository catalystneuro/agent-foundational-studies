"""Population decoding of grating orientation, as a function of population size.

A linear discriminant reports which of the four orientations (direction folded
modulo 180) was on the screen, from the single-trial spike counts of N randomly
chosen units. Sweeping N matters: at N = 25 both cortex and LGd are close to
ceiling, so a single population size says little. The slope at small N is what
reveals how much orientation information an average unit carries.
"""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from analyze_helpers import responsive

R = responsive(pd.read_pickle("units.pkl"))
SIZES = [1, 2, 4, 8, 16, 32]
N_REPEATS = 15
rng = np.random.default_rng(0)

rows = []
jobs = [(ses, lbl, m) for ses in sorted(R.session.unique())
        for lbl, m in [("visual cortex", R.region == "cortex"), ("LGd", R.area == "LGd")]]
for ses, label, mask in tqdm(jobs, desc="decoding"):
    d = np.load(f"extracted/{ses}.npz", allow_pickle=True)
    rates = d["dg_rates"].astype(float)
    ori4 = d["dg_ori"] % 180
    sel = R[(R.session == ses) & mask]
    idx_map = {int(u): i for i, u in enumerate(d["unit_ids"])}
    pool = np.array([idx_map[int(u)] for u in sel.unit_id])
    for n in SIZES:
        if len(pool) < n:
            continue
        for rep in range(N_REPEATS):
            take = rng.choice(pool, n, replace=False)
            X = rates[take].T
            ok = np.isfinite(X).all(axis=1)
            X = np.sqrt(np.clip(X[ok], 0, None))   # variance-stabilising transform
            y = ori4[ok]
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            acc = np.mean([LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                           .fit(X[tr], y[tr]).score(X[te], y[te]) for tr, te in cv.split(X, y)])
            ysh = rng.permutation(y)
            sh = np.mean([LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                          .fit(X[tr], ysh[tr]).score(X[te], ysh[te])
                          for tr, te in cv.split(X, ysh)])
            rows.append(dict(session=ses, group=label, n_units=n, rep=rep,
                             accuracy=acc, shuffled=sh))

dec = pd.DataFrame(rows)
dec.to_csv("decoding_results.csv", index=False)
print(dec.groupby(["group", "n_units"])[["accuracy", "shuffled"]].mean().to_string())
