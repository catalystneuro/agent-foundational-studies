"""Figure 8: a Poisson GLM of position tuning, fitted and scored with NeMoS.

The shuffle test in Figure 5 asks whether a cell's spikes are more spatially
concentrated than chance. This asks a stronger, model-based question: does
knowing the animal's position let a Poisson model predict the spike counts of a
cell in traversals it was never fitted on?
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from tqdm import tqdm

import pf_core as pf

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160})

SESSION = "Achilles_10252013"
GLM_BIN_SIZE = 0.04     # s
N_BASIS = 12

res = pf.analyze_session(SESSION, n_shuffles=200)
position, pyr = res["position"], res["pyr"]
TRACK_LEN = res["meta"]["track_length_cm"]
tables = {d: res["per_dir"][d]["table"] for d in pf.DIRECTIONS}
maps = {d: res["per_dir"][d]["maps"] for d in pf.DIRECTIONS}
bin_centers = res["per_dir"]["rightward"]["engine"].bin_centers

basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS, bounds=(0.0, TRACK_LEN), label="position")
grid, basis_grid = basis.evaluate_on_grid(200)


def design(epochs):
    """Binned spike counts and the position design matrix over `epochs`."""
    counts = pyr.count(GLM_BIN_SIZE, ep=epochs)
    pos = position.interpolate(counts, ep=counts.time_support)
    X = basis.compute_features(pos)
    ok = np.isfinite(np.asarray(X)).all(axis=1)
    return np.asarray(X)[ok], np.asarray(counts)[ok]


def fold_split(direction, parity):
    trials = res["trial_eps"][direction]
    idx = np.arange(len(trials))
    train = res["dir_eps"][direction].intersect(trials[idx[idx % 2 == parity]])
    test = res["dir_eps"][direction].intersect(trials[idx[idx % 2 != parity]])
    return train, test


unit_keys = np.array(list(pyr.keys()))
scores = {d: np.full((len(unit_keys), 2), np.nan) for d in pf.DIRECTIONS}
tuning_glm = {d: np.zeros((len(unit_keys), grid.size)) for d in pf.DIRECTIONS}

for direction in pf.DIRECTIONS:
    for parity in (0, 1):
        train_ep, test_ep = fold_split(direction, parity)
        Xtr, Ytr = design(train_ep)
        Xte, Yte = design(test_ep)
        for i in tqdm(range(len(unit_keys)), desc=f"GLM {direction} fold{parity}", leave=False):
            # A unit that is silent in one fold has no rate to initialise the
            # intercept from and no null model to score against; leave it NaN.
            if Ytr[:, i].sum() == 0 or Yte[:, i].sum() == 0:
                tuning_glm[direction][i] = np.nan
                continue
            model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                                solver_name="LBFGS")
            model.fit(Xtr, Ytr[:, i])
            scores[direction][i, parity] = model.score(
                Xte, Yte[:, i], score_type="pseudo-r2-McFadden")
            rate = np.exp(basis_grid @ model.coef_ + model.intercept_)
            tuning_glm[direction][i] += 0.5 * np.ravel(rate) / GLM_BIN_SIZE

cv_r2 = {}
for d in pf.DIRECTIONS:
    ok = np.isfinite(scores[d])
    n_ok = ok.sum(axis=1)
    cv_r2[d] = np.where(n_ok > 0,
                        np.where(ok, scores[d], 0.0).sum(axis=1) / np.maximum(n_ok, 1),
                        np.nan)
n_skipped = sum(int((~np.isfinite(scores[d])).any(axis=1).sum()) for d in pf.DIRECTIONS)
print(f"unit-fold combinations skipped (no spikes in a fold): {n_skipped}")
for d in pf.DIRECTIONS:
    pc = tables[d].is_place_cell.values
    print(f"{d}: cross-validated pseudo-R2  place cells {np.nanmedian(cv_r2[d][pc]):.3f}, "
          f"others {np.nanmedian(cv_r2[d][~pc]):.3f}")

# ---------------------------------------------------------------- Figure 8 ---
fig = plt.figure(figsize=(12, 6.4))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.30,
                      left=0.07, right=0.98, top=0.85, bottom=0.10)

tr = tables["rightward"]
good = tr.is_place_cell.values & (tr.peak_rate.values > 3.0)
cand = np.flatnonzero(good)
examples = list(dict.fromkeys(
    [cand[np.argmin(np.abs(tr.peak_pos_cm.values[cand] - t))] for t in (25, 75, 130)]))

for n, idx in enumerate(examples):
    ax = fig.add_subplot(gs[0, n])
    for d in pf.DIRECTIONS:
        ax.plot(bin_centers, maps[d][idx], color=pf.DIR_COLORS[d], lw=1.0, alpha=0.5,
                label=f"{d}, binned" if n == 0 else None)
        ax.plot(grid, tuning_glm[d][idx], color=pf.DIR_COLORS[d], lw=2.0, ls="-",
                label=f"{d}, GLM" if n == 0 else None)
    ax.set_xlim(0, TRACK_LEN)
    ax.set_xlabel("position (cm)")
    if n == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{'ABC'[n]}  unit {int(tr.unit.values[idx])}  "
                 f"(CV pseudo-$R^2$ = {cv_r2['rightward'][idx]:.2f})",
                 loc="left", fontweight="bold", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
d = "rightward"
pc = tables[d].is_place_cell.values
lo = min(-0.02, np.nanmin(cv_r2[d]) * 1.05)
bins = np.linspace(lo, max(0.25, np.nanmax(cv_r2[d]) * 1.05), 36)
ax.hist(cv_r2[d][~pc & np.isfinite(cv_r2[d])], bins=bins, color="0.7", label=f"not a place cell (n={(~pc).sum()})")
ax.hist(cv_r2[d][pc & np.isfinite(cv_r2[d])], bins=bins, histtype="step", lw=2.0, color=pf.DIR_COLORS[d],
        label=f"place cell (n={pc.sum()})")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("cross-validated pseudo-$R^2$"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("D  Held-out model fit", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
for d in pf.DIRECTIONS:
    pc = tables[d].is_place_cell.values
    ax.scatter(tables[d].info_bits_per_spike.values[pc], cv_r2[d][pc], s=14,
               color=pf.DIR_COLORS[d], label=f"{d}, place cell")
    ax.scatter(tables[d].info_bits_per_spike.values[~pc], cv_r2[d][~pc], s=14,
               facecolors="none", edgecolors=pf.DIR_COLORS[d], linewidths=0.7, alpha=0.6)
allinfo = np.concatenate([tables[d].info_bits_per_spike.values for d in pf.DIRECTIONS])
allr2 = np.concatenate([cv_r2[d] for d in pf.DIRECTIONS])
ok = np.isfinite(allinfo) & np.isfinite(allr2)
rho = np.corrcoef(allinfo[ok], allr2[ok])[0, 1]
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Skaggs information (bits/spike)"); ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title(f"E  Agreement of the two measures, r = {rho:.2f}", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
pos_frac = []
for d in pf.DIRECTIONS:
    pc = tables[d].is_place_cell.values
    pos_frac.append([np.nanmean(cv_r2[d][pc] > 0.01), np.nanmean(cv_r2[d][~pc] > 0.01)])
pos_frac = np.array(pos_frac)
xx = np.arange(2)
ax.bar(xx - 0.19, 100 * pos_frac[0], width=0.38, color=pf.DIR_COLORS["rightward"],
       label="rightward")
ax.bar(xx + 0.19, 100 * pos_frac[1], width=0.38, color=pf.DIR_COLORS["leftward"],
       label="leftward")
ax.set_xticks(xx); ax.set_xticklabels(["place cells", "other units"])
ax.set_ylabel("% of units with pseudo-$R^2$ > 0.01")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("F  Position improves held-out prediction", loc="left", fontweight="bold")

fig.suptitle(f"Poisson GLM of position tuning (NeMoS), {SESSION}\n"
             f"{N_BASIS} B-spline basis functions over position, "
             f"{GLM_BIN_SIZE * 1000:.0f} ms bins, fitted on half the traversals and "
             "scored on the other half", fontweight="bold", fontsize=10)
fig.savefig("fig08_glm_position_tuning.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig08_glm_position_tuning.png")
