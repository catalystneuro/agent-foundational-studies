"""A GLM control: does theta phase predict spiking beyond speed and spike history?

A mean-resultant-length test can in principle be inflated by anything that co-varies
with the LFP: a unit that fires in bursts at ~8 Hz would show phase concentration even
without being entrained, and running speed modulates both firing rate and theta. This
script fits a Poisson GLM per unit in which speed and the unit's own spike history are
always present, and asks how much held-out log-likelihood is gained by adding a cyclic
basis over theta phase.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import analysis as an
import theta_lib as tl

SESSION = tl.SESSIONS[0]
BIN = 0.01           # s
N_FOLDS = 5
N_UNITS = 40         # units with the most spikes, to keep the fit time bounded
HIST_WINDOW = 25     # bins = 250 ms

s = an.load_session(SESSION)
ep = s["run_normal"]
units = s["units"]
df = pd.read_csv("results/unit_stats.csv")
df = df[df.session == SESSION].sort_values("n_spikes", ascending=False).head(N_UNITS)

# Shared regressors, sampled at the spike-count bins.
counts_all = units.count(BIN, ep=ep)
t_bins = counts_all.t
phase_bins = nap.Tsd(t=t_bins, d=s["phase_at"](t_bins), time_support=ep)
speed_bins = nap.Tsd(t=t_bins, d=np.interp(t_bins, s["speed"].t, tl.smooth_speed(s["speed"]).d),
                     time_support=ep)

phase_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0.0, 2 * np.pi), label="phase")
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=5, bounds=(0.0, float(speed_bins.d.max())),
                                    label="speed")
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=HIST_WINDOW, label="history")

X_phase = np.asarray(phase_basis.compute_features(phase_bins))
X_speed = np.asarray(speed_basis.compute_features(speed_bins))

# Cross-validation folds are interleaved 10 s blocks rather than contiguous fifths.
# Blocks are long compared with the spike-history window, so little information leaks
# across the split, while interleaving keeps each fold representative of the whole
# session: theta locking is not perfectly stationary, and with contiguous folds a model
# trained on one stretch is penalised for tuning that has drifted by the next.
BLOCK = int(10.0 / BIN)
fold_of = (np.arange(len(t_bins)) // BLOCK) % N_FOLDS

rows, examples = [], {}
for uid in tqdm(df.unit.astype(int), desc="GLM per unit"):
    y = np.asarray(units[int(uid)].count(BIN, ep=ep)).astype(float)
    X_hist = np.asarray(hist_basis.compute_features(nap.Tsd(t=t_bins, d=y, time_support=ep)))
    X_full = np.hstack([X_phase, X_speed, X_hist])
    X_red = np.hstack([X_speed, X_hist])
    good = np.isfinite(X_full).all(axis=1)

    d_lls, ll_full, ll_red = [], [], []
    coef_phase = None
    for k in range(N_FOLDS):
        tr, te = good & (fold_of != k), good & (fold_of == k)
        if y[te].sum() < 10:
            continue
        m_full = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                             solver_name="LBFGS").fit(X_full[tr], y[tr])
        m_red = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                            solver_name="LBFGS").fit(X_red[tr], y[tr])
        a = m_full.score(X_full[te], y[te], score_type="log-likelihood")
        b = m_red.score(X_red[te], y[te], score_type="log-likelihood")
        # scores are per time bin; express the gain per spike so units are comparable
        d_lls.append((a - b) * te.sum() / y[te].sum())
        ll_full.append(a)
        ll_red.append(b)
        if k == 0:
            coef_phase = m_full.coef_[:X_phase.shape[1]]

    if not d_lls:
        continue
    rows.append(dict(unit=int(uid), d_ll_per_spike=float(np.mean(d_lls)),
                     d_ll_sem=float(np.std(d_lls) / np.sqrt(len(d_lls))),
                     ll_full=float(np.mean(ll_full)), ll_red=float(np.mean(ll_red))))
    examples[int(uid)] = coef_phase

res = pd.DataFrame(rows).merge(df[["unit", "mrl", "shuffle_p", "cell_type", "n_spikes"]], on="unit")
res.to_csv("results/glm_results.csv", index=False)
print(res.describe())

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
grid, phase_kernels = phase_basis.evaluate_on_grid(200)
grid_deg = np.degrees(grid * 2 * np.pi / grid.max()) if grid.max() <= 1 else np.degrees(grid)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

ax = axes[0]
ax.hist(res.d_ll_per_spike, bins=25, color="crimson", alpha=0.85)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Δ held-out log-likelihood per spike\n(phase term added)")
ax.set_ylabel("units")
ax.set_title(f"{(res.d_ll_per_spike > 0).sum()}/{len(res)} units improve when\n"
             "theta phase is added to speed + spike history", fontsize=10)

ax = axes[1]
sig = res.shuffle_p < 0.05
ax.scatter(res.mrl[~sig], res.d_ll_per_spike[~sig], s=22, color="0.6", label="n.s. (shuffle)")
ax.scatter(res.mrl[sig], res.d_ll_per_spike[sig], s=22, color="crimson", label="locked")
ax.axhline(0, color="k", lw=0.8)
r = np.corrcoef(res.mrl, res.d_ll_per_spike)[0, 1]
ax.set_xlabel("mean resultant length")
ax.set_ylabel("Δ log-likelihood per spike")
ax.set_title(f"GLM gain tracks the circular statistic (r={r:.2f})", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
top = res.sort_values("d_ll_per_spike", ascending=False).head(4)
ph_normal = np.load("results/spike_phases_normal.npz")
for _, r_ in top.iterrows():
    k = examples[int(r_.unit)]
    tuning = np.exp(phase_kernels @ k)
    tuning = tuning / tuning.mean()
    line, = ax.plot(np.linspace(0, 360, len(tuning)), tuning, lw=2, label=f"u{int(r_.unit)}")
    c, h = an.phase_histogram(ph_normal[f"{SESSION}|{int(r_.unit)}"])
    ax.plot(np.degrees(c), h * len(h), lw=1, ls=":", color=line.get_color(), alpha=0.8)
ax.set_xlabel("theta phase (deg)")
ax.set_ylabel("relative rate")
ax.set_title("GLM phase tuning (solid) vs\nempirical spike-phase histogram (dotted)", fontsize=10)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("fig08_glm.png", dpi=150)
print("wrote fig08_glm.png")
