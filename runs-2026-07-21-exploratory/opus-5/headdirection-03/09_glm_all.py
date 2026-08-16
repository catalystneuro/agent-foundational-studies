"""Fit a cyclic-spline Poisson GLM of head direction for every unit, every session."""
import os
import pickle

import numpy as np
import nemos as nmo
import pynapple as nap
from tqdm import tqdm

import hd_lib

BIN = 0.1
N_BASIS = 10


def glm_session(session):
    d = hd_lib.load_session(session)
    hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]
    mid = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
    train, test = nap.IntervalSet(sq.start[0], mid), nap.IntervalSet(mid, sq.end[0])
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=N_BASIS, label="head_direction")

    data = {}
    for name, ep in [("train", train), ("test", test)]:
        counts = units.count(BIN, ep)
        feat = hd.bin_average(BIN, ep)
        ok = ~np.isnan(feat.values)
        data[name] = (np.asarray(basis.compute_features(feat.values[ok])),
                      np.asarray(counts.values)[ok], feat.values[ok])
    Xtr, ytr, _ = data["train"]
    Xte, yte, _ = data["test"]

    grid = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    Xg = np.asarray(basis.compute_features(grid))

    r2, pred_tc = [], []
    for i in range(ytr.shape[1]):
        m = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4).fit(Xtr, ytr[:, i])
        r2.append(float(m.score(Xte, yte[:, i], score_type="pseudo-r2-McFadden")))
        pred_tc.append(np.asarray(m.predict(Xg)) / BIN)
    return dict(pseudo_r2=np.array(r2), pred_tc=np.array(pred_tc), grid=grid)


if __name__ == "__main__":
    out = {}
    for s in tqdm(hd_lib.SESSIONS, desc="GLM"):
        out[s] = glm_session(s)
        print(s, "median pseudo-R2 %.3f" % np.median(out[s]["pseudo_r2"]), flush=True)
    os.makedirs("results", exist_ok=True)
    pickle.dump(out, open("results/glm.pkl", "wb"))
    print("saved results/glm.pkl")
