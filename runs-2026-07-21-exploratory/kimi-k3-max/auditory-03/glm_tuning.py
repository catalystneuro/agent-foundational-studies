"""GLM-based frequency tuning (nemos) on the prototype session, sub-LA3 ses-3.

Poisson GLM: spike count in 5-55 ms response window ~ B-spline basis over log2(frequency).
Compares model-based tuning curves with empirical per-frequency rates.
"""
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
import nemos as nmo
from scipy import stats

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083"
FIG = "figures"
RESP = (0.005, 0.055)
BASE = (-0.050, 0.0)

disk_cache = remfile.DiskCache('/tmp/remfile_cache_auditory03')
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
trials = nwb["trials"]
freqs = np.array(sorted(np.unique(trials["stim_frequency"])))
tone_onsets = nap.Ts(t=np.array(trials["start"]))
trial_freq = np.array(trials["stim_frequency"])
log2f = np.log2(trial_freq)

duration = float(units.time_support.end[0]) - float(units.time_support.start[0])
rates = np.array([len(units[i].t) / duration for i in range(len(units))])
keep = np.where(rates >= 0.5)[0]

# B-spline basis over log2 frequency (5 distinct freqs -> 4 basis funcs + intercept)
basis = nmo.basis.BSplineEval(n_basis_funcs=4)
X = basis.compute_features(log2f)
print("design matrix:", X.shape)

grid = np.linspace(log2f.min(), log2f.max(), 200)
Xg = basis.compute_features(grid)

win_len = RESP[1] - RESP[0]
results = []
for u in keep:
    sp = units[int(u)]
    pe = nap.compute_perievent(sp, tone_onsets, window=(BASE[0], RESP[1]))
    n_trials = len(tone_onsets)
    y = np.zeros(n_trials)
    base = np.zeros(n_trials)
    for j in range(n_trials):
        t = pe[j].t
        y[j] = np.sum((t >= RESP[0]) & (t < RESP[1]))
        base[j] = np.sum((t >= BASE[0]) & (t < BASE[1])) / (BASE[1] - BASE[0])

    model = nmo.glm.GLM(solver_name="LBFGS")
    model.fit(X, y)
    pred_grid = model.predict(Xg) / win_len  # Hz

    # null (intercept-only) model for McFadden pseudo-R2
    X0 = np.ones((n_trials, 1))
    null = nmo.glm.GLM(solver_name="LBFGS")
    null.fit(X0, y)
    ll_m = model.score(X, y)
    ll_0 = null.score(X0, y)
    pseudo_r2 = 1 - ll_m / ll_0 if ll_0 != 0 else np.nan

    # empirical response-window rate per frequency
    emp = np.array([y[trial_freq == f].mean() for f in freqs]) / win_len
    emp_sem = np.array([y[trial_freq == f].std() / np.sqrt(np.sum(trial_freq == f))
                        for f in freqs]) / win_len
    net = emp - base.mean()

    results.append(dict(unit=int(u), pred=pred_grid, emp=emp, emp_sem=emp_sem,
                        net=net, pseudo_r2=pseudo_r2,
                        bf_emp=float(freqs[np.argmax(net)]),
                        bf_glm=float(2 ** grid[np.argmax(pred_grid)]),
                        kw_p=stats.kruskal(*[ (y - 0)[trial_freq == f] for f in freqs]).pvalue))

print(f"fitted {len(results)} units; median pseudo-R2 = "
      f"{np.nanmedian([r['pseudo_r2'] for r in results]):.3f}")

# ---- Figure 5: GLM tuning curves for example units ----
sig_sorted = sorted([r for r in results if r["kw_p"] < 0.01], key=lambda r: -r["pseudo_r2"])
examples = sig_sorted[:4]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True)
for ax, r in zip(axes, examples):
    ax.errorbar(np.log2(freqs / 1000), r["emp"], yerr=r["emp_sem"], fmt="o",
                color="black", ms=4, capsize=3, label="empirical (mean ± SEM)")
    ax.plot(grid - np.log2(1000), r["pred"], color="crimson", lw=2,
            label="Poisson GLM")
    ax.set_title(f'unit {r["unit"]}  (pseudo-$R^2$={r["pseudo_r2"]:.2f})', fontsize=9)
    ax.set_xticks(np.log2(freqs / 1000))
    ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
    ax.set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("firing rate 5–55 ms (Hz)")
axes[0].legend(fontsize=8, frameon=False)
plt.suptitle("GLM-based frequency tuning curves (B-spline over log2 frequency)", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_glm_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5")

# ---- Figure 6: GLM population summary ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
pr2 = np.array([r["pseudo_r2"] for r in results])
sig_mask = np.array([r["kw_p"] < 0.01 for r in results])
ax.hist(pr2[sig_mask], bins=15, alpha=0.8, color="crimson", label="tuned (KW p<0.01)")
ax.hist(pr2[~sig_mask], bins=15, alpha=0.8, color="gray", label="not tuned")
ax.set_xlabel("McFadden pseudo-$R^2$ (frequency GLM)")
ax.set_ylabel("# units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("GLM explanatory power")

ax = axes[1]
bf_e = np.array([r["bf_emp"] for r in results])[sig_mask]
bf_g = np.array([r["bf_glm"] for r in results])[sig_mask]
ax.scatter(bf_e / 1000, bf_g / 1000, s=15, color="steelblue")
ax.plot([1.5, 40], [1.5, 40], "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([2, 4, 8, 16, 32]); ax.set_yticks([2, 4, 8, 16, 32])
ax.set_xticklabels([2, 4, 8, 16, 32]); ax.set_yticklabels([2, 4, 8, 16, 32])
ax.set_xlabel("empirical best frequency (kHz)")
ax.set_ylabel("GLM best frequency (kHz)")
ax.set_title("Best frequency: empirical vs GLM")
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_glm_population.png", dpi=150)
plt.close()
print("saved fig6")
print("DONE")
