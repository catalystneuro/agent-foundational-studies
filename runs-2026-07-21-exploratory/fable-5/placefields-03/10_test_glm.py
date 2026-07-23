"""Does a Poisson GLM need position, or would running speed alone do?

On a linear track speed covaries with position (the rat accelerates away from
the reward site and decelerates into the next one), so a purely speed-tuned
neuron can masquerade as a place cell. Fit three GLMs per neuron and compare
their held-out likelihood.
"""

import time
import numpy as np
import nemos as nmo
from scipy.stats import poisson
import place_cell_lib as pcl

BIN = 0.1  # s
MAX_SPEED = 1.5


def per_neuron_pseudo_r2(y, mu, mu_null):
    """McFadden pseudo-R^2 per neuron, evaluated on held-out bins."""
    ll = poisson.logpmf(y, np.maximum(mu, 1e-9)).sum(axis=0)
    ll0 = poisson.logpmf(y, np.maximum(mu_null, 1e-9)).sum(axis=0)
    return 1.0 - ll / ll0


def fit_glm_models(sess, spikes, ep, bin_size=BIN, n_pos_basis=12, n_speed_basis=6,
                   reg_strength=1e-3):
    counts = spikes.count(bin_size, ep=ep)
    pos = sess["position"].interpolate(counts, ep=ep)
    spd = np.clip(sess["speed"].interpolate(counts, ep=ep), 0, MAX_SPEED)

    pos_basis = nmo.basis.BSplineEval(n_basis_funcs=n_pos_basis,
                                      bounds=(0.0, sess["extent"]), label="position")
    spd_basis = nmo.basis.BSplineEval(n_basis_funcs=n_speed_basis,
                                      bounds=(0.0, MAX_SPEED), label="speed")
    designs = {
        "speed": (spd_basis, (spd,)),
        "position": (pos_basis, (pos,)),
        "position+speed": (pos_basis + spd_basis, (pos, spd)),
    }

    y = np.asarray(counts)
    t = counts.t
    n = len(ep)
    member = np.stack([(t >= ep.start[i]) & (t <= ep.end[i]) for i in range(n)])
    idx = np.arange(n)
    folds = [(idx[0::2], idx[1::2]), (idx[1::2], idx[0::2])]

    # A unit silent in one fold gives an infinite intercept at initialization,
    # so only fit units that fire in both folds.
    m0, m1 = member[folds[0][0]].any(axis=0), member[folds[0][1]].any(axis=0)
    keep = (y[m0].sum(axis=0) >= 10) & (y[m1].sum(axis=0) >= 10)
    y = y[:, keep]

    scores = {k: np.zeros((y.shape[1], 2)) for k in designs}
    fitted = {}
    for f, (tr, te) in enumerate(folds):
        m_tr, m_te = member[tr].any(axis=0), member[te].any(axis=0)
        mu_null = np.repeat(y[m_tr].mean(axis=0)[None, :], m_te.sum(), axis=0)
        for name, (basis, inputs) in designs.items():
            X = np.asarray(basis.compute_features(*inputs))
            model = nmo.glm.PopulationGLM(regularizer="Ridge",
                                          regularizer_strength=reg_strength,
                                          solver_name="LBFGS")
            model.fit(X[m_tr], y[m_tr])
            mu = np.asarray(model.predict(X[m_te]))
            scores[name][:, f] = per_neuron_pseudo_r2(y[m_te], mu, mu_null)
            if f == 0:
                fitted[name] = (model, basis)
    return dict(scores={k: v.mean(axis=1) for k, v in scores.items()},
                fitted=fitted, counts=counts, bin_size=bin_size,
                unit_ids=np.asarray(list(spikes.keys()))[keep], keep=keep)


if __name__ == "__main__":
    t0 = time.time()
    sess = pcl.load_session(pcl.SESSIONS[0][0])
    laps = pcl.lap_epochs(sess)
    mov = pcl.moving_epochs(sess, laps)
    spikes = sess["spikes"]
    exc = spikes[np.where(spikes.get_info("cell_type") == "excitatory")[0]]

    out = fit_glm_models(sess, exc, mov["rightward"])
    for k, v in out["scores"].items():
        print(f"{k:16s} pseudo-R2: median {np.median(v):.4f}  "
              f"mean {v.mean():.4f}  max {v.max():.4f}")
    dpos = out["scores"]["position+speed"] - out["scores"]["speed"]
    print(f"\nposition improves over speed alone in {(dpos > 0).sum()}/{len(dpos)} units, "
          f"median gain {np.median(dpos):.4f}")

    # GLM position tuning on a grid, for comparison with the histogram rate map
    model, basis = out["fitted"]["position"]
    grid, _ = basis.evaluate_on_grid(200)
    Xg = np.asarray(basis.evaluate_on_grid(200)[1])
    rate = np.asarray(model.predict(Xg)) / out["bin_size"]
    print("GLM tuning grid:", rate.shape, "peak rates:",
          np.round(np.sort(rate.max(axis=0))[-5:], 1))
    print("elapsed %.1f s" % (time.time() - t0))
