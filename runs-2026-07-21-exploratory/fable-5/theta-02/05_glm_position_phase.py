"""Model-based confirmation of phase precession with NeMoS Poisson GLMs.

Three nested models are fit to 20 ms spike counts from each place field:

  A  position only          rate = exp(b0 + f(x))
  B  fixed preferred phase  rate = exp(b0 + f(x) + c*cos(phi) + s*sin(phi))
  C  precessing             rate = exp(b0 + f(x) + sum_j B_j(x)*(c_j*cos(phi) + s_j*sin(phi)))

Model C is model B with the phase-modulation vector allowed to rotate with position: its
phase term equals A(x)*cos(phi - theta_pref(x)), so theta_pref(x) is a fitted preferred phase
that may drift across the field. That is precisely phase precession, and it costs only
2*n_basis extra parameters rather than the full outer product of two bases. Cross-validation
folds are whole laps, so train and test never share a traversal.
"""

import numpy as np
import scipy.stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from tqdm import tqdm

import theta_lib as tl

BIN = 0.02
N_POS_BASIS = 8
N_FOLDS = 4

res = tl.analyze_session(tl.SESSIONS[0])
pos, theta_phase, laps = res["position"], res["theta_phase"], res["laps"]
pos_basis = nmo.basis.BSplineEval(n_basis_funcs=N_POS_BASIS, label="position")

fields = sorted([f for f in res["fields"] if f["n_spikes"] >= 100], key=lambda f: f["rho"])
print("fitting GLMs for %d place fields" % len(fields))


def field_design(f):
    """Spike counts and the three design matrices for one place field."""
    ep = res["dir_ep"][f["direction"]]
    lap_ep = laps[laps.direction == f["direction"]].intersect(ep)
    count = res["pyr"][f["unit"]].count(BIN, ep=lap_ep)
    p = pos.interpolate(count, ep=count.time_support).values
    ph = nap.Ts(t=count.t).value_from(theta_phase).values
    y = count.values.astype(float)
    lap_id = np.searchsorted(lap_ep.start, count.t, side="right") - 1
    good = np.isfinite(p) & np.isfinite(ph)
    p, ph, y, lap_id = p[good], ph[good], y[good], lap_id[good]

    B = np.asarray(pos_basis.compute_features(p))
    cos_ph, sin_ph = np.cos(ph)[:, None], np.sin(ph)[:, None]
    X = {
        "position only": B,
        "fixed preferred phase": np.hstack([B, cos_ph, sin_ph]),
        "precessing phase": np.hstack([B, B * cos_ph, B * sin_ph]),
    }
    return X, y, lap_id, p


def fit_glm(X, y):
    return nmo.glm.GLM(
        solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-3
    ).fit(X, y)


MODELS = ["position only", "fixed preferred phase", "precessing phase"]
scores = {k: [] for k in MODELS}

for f in tqdm(fields, desc="GLM fits"):
    X, y, lap_id, _ = field_design(f)
    uniq = np.unique(lap_id)
    fold_of_lap = dict(zip(uniq, np.arange(len(uniq)) % N_FOLDS))
    fold = np.array([fold_of_lap[i] for i in lap_id])
    for name in MODELS:
        ll = []
        for k in range(N_FOLDS):
            tr, te = fold != k, fold == k
            if y[te].sum() < 5:
                continue
            glm = fit_glm(X[name][tr], y[tr])
            ll.append(glm.score(X[name][te], y[te], score_type="log-likelihood"))
        scores[name].append(np.mean(ll))

scores = {k: np.array(v) for k, v in scores.items()}
base = scores["position only"]
d_fix = scores["fixed preferred phase"] - base
d_pre = scores["precessing phase"] - base
wil = scipy.stats.wilcoxon(scores["precessing phase"], scores["fixed preferred phase"])

print("\nheld-out log-likelihood per bin, relative to the position-only model:")
for name in MODELS:
    d = scores[name] - base
    print("  %-22s %+0.5f  (better in %d/%d fields)" % (name, d.mean(), (d > 0).sum(), len(d)))
print("precessing vs fixed preferred phase: Wilcoxon p = %.2g" % wil.pvalue)

# ---------------------------------------------------------------------------------
# Figure 9
# ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(d_fix, d_pre, s=22, color="tab:blue")
lim = [min(d_fix.min(), d_pre.min()) * 1.1, max(d_fix.max(), d_pre.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("fixed preferred phase (log-lik/bin gain)")
ax.set_ylabel("precessing phase (log-lik/bin gain)")
ax.set_title("Held-out likelihood gain over a\nposition-only model (one point per field)")

ax = fig.add_subplot(gs[0, 1])
ax.boxplot([scores[n] - base for n in MODELS], showfliers=False)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_ylabel("Held-out log-likelihood per bin\n(relative to position only)")
ax.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=9)
ax.set_title("Adding theta phase helps; letting the\npreferred phase precess helps more")

ax = fig.add_subplot(gs[0, 2])
ax.hist(d_pre - d_fix, bins=20, color="tab:red")
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("precessing $-$ fixed (log-lik/bin)")
ax.set_ylabel("Number of fields")
ax.set_title("Precessing model wins in %d/%d fields\n(Wilcoxon p = %.1g)"
             % ((d_pre > d_fix).sum(), len(d_pre), wil.pvalue))

# fitted preferred phase across the field for three example cells, plotted against distance
# travelled into the field so that every panel reads left to right along the run
for j, f in enumerate(fields[:3]):
    X, y, _, _ = field_design(f)
    glm = fit_glm(X["precessing phase"], y)
    w = np.asarray(glm.coef_)
    width = f["hi"] - f["lo"]
    grid_dist = np.linspace(0, width, 80)
    grid_pos = f["lo"] + grid_dist if f["direction"] == "right" else f["hi"] - grid_dist
    Bg = np.asarray(pos_basis.compute_features(grid_pos))
    c = Bg @ w[N_POS_BASIS: 2 * N_POS_BASIS]
    s = Bg @ w[2 * N_POS_BASIS:]
    pref = np.degrees(np.unwrap(np.arctan2(s, c)))
    pref -= 360 * np.floor(pref.mean() / 360)  # align the curve with the spike cloud
    depth = np.hypot(c, s)

    ax = fig.add_subplot(gs[1, j])
    sp_dist = f["x"] * width
    for shift in (0, 360):
        ax.plot(sp_dist, np.degrees(f["phase"]) + shift, ".", ms=2.5, color="0.75")
        ax.plot(grid_dist, pref + shift, "-", color="tab:red", lw=2.5,
                label="GLM preferred phase" if shift == 0 else None)
    ax.set_ylim(0, 720)
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlim(0, width)
    ax.set_xlabel("Distance into field (cm)")
    if j == 0:
        ax.set_ylabel("Theta phase (deg)")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title("unit %d (%sward): phase modulation depth %.2f"
                 % (f["unit"], f["direction"], depth.mean()), fontsize=10)

fig.suptitle("NeMoS Poisson GLMs: the preferred theta phase of a place cell shifts with "
             "position within its field (%s)" % res["session"], fontsize=13)
fig.savefig("fig09_glm_position_phase.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done")
