"""Prototype a NeMoS Poisson GLM with a cyclic-spline head-direction basis."""
import numpy as np
import nemos as nmo
import pynapple as nap
import hd_lib

S = "sub-A3701_ses-191119"
d = hd_lib.load_session(S)
hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]
BIN = 0.1

half = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
train, test = nap.IntervalSet(sq.start[0], half), nap.IntervalSet(half, sq.end[0])

basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=10, label="head_direction")

res = {}
for name, ep in [("train", train), ("test", test)]:
    counts = units.count(BIN, ep)
    feat = hd.bin_average(BIN, ep)
    ok = ~np.isnan(feat.values)
    X = basis.compute_features(feat.values[ok])
    res[name] = (np.asarray(X), np.asarray(counts.values)[ok])
    print(name, "X", X.shape, "y", res[name][1].shape)

Xtr, ytr = res["train"]
Xte, yte = res["test"]

ft = hd_lib.FastTuning(hd, sq, 60)
mvl = np.array([hd_lib.circular_mean_vector(ft.centers, ft.curve(units[c].t))[0]
                for c in units.keys()])

scores = []
for i in range(ytr.shape[1]):
    m = nmo.glm.GLM(solver_name="LBFGS",
                    regularizer="Ridge", regularizer_strength=1e-4).fit(Xtr, ytr[:, i])
    scores.append(m.score(Xte, yte[:, i], score_type="pseudo-r2-McFadden"))
scores = np.array(scores)
print("\npseudo-R2 (held-out): HD-tuned (MVL>0.3) median %.3f | others median %.3f"
      % (np.median(scores[mvl > 0.3]), np.median(scores[mvl <= 0.3])))
print("corr(MVL, pseudo-R2) = %.3f" % np.corrcoef(mvl, scores)[0, 1])
print("max %.3f min %.4f" % (scores.max(), scores.min()))
np.savez("cache/_proto_glm.npz", scores=scores, mvl=mvl)
