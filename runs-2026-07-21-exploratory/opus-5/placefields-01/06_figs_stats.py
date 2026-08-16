"""Statistical characterisation of the place-cell population in one session."""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

import pf_lib as pf

SESSION = "Achilles_10252013"
COL = {"right": "#1f77b4", "left": "#d62728"}
DIRS = ("right", "left")

urls = pf.session_urls()
nwb, nwbfile, io = pf.load_session(urls[SESSION])
maze = pf.maze_epoch(nwbfile)
pos, L = pf.get_position(nwb, nwbfile)
vel, eps = pf.running_epochs(pos)
pyr = pf.select_pyramidal(nwb["units"], maze)

rng = np.random.default_rng(0)
res, nulls = {}, {}
for d in DIRS:
    res[d] = pf.direction_metrics(pyr, pos, eps[d], L, rng=rng)
    _, null = pf.shuffled_si_threshold(
        pyr, pos, eps[d], res[d]["occ"], L, n_shuffle=pf.N_SHUFFLE,
        rng=np.random.default_rng(7),
    )
    nulls[d] = null
mets = {d: res[d]["metrics"] for d in DIRS}
tcs = {d: res[d]["tc_smooth"] for d in DIRS}
bins = res["right"]["bins"]
centers = 0.5 * (bins[:-1] + bins[1:])
uidx = np.asarray(pyr.index)

# Directional selectivity: correlation between the two rate maps of each unit.
def map_corr(a, b):
    """Pearson r between two rate maps; NaN when either map is flat."""
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


dir_corr = np.array(
    [map_corr(tcs["right"].values[:, j], tcs["left"].values[:, j]) for j in range(len(uidx))]
)
pc_either = mets["right"].is_place_cell.values | mets["left"].is_place_cell.values

# Null for the directional correlation: pair each cell's rightward map with a
# different cell's leftward map, which removes any genuine same-cell coupling.
perm = rng.permutation(len(uidx))
dir_corr_null = np.array(
    [map_corr(tcs["right"].values[:, j], tcs["left"].values[:, perm[j]]) for j in range(len(uidx))]
)

fig, axs = plt.subplots(2, 4, figsize=(19, 9))
axs = axs.ravel()

# (a) spatial information against the circular-shift null
ax = axs[0]
obs = np.concatenate([mets[d].si.values for d in DIRS])
nul = np.concatenate([np.concatenate([nulls[d][u] for u in uidx]) for d in DIRS])
edges = np.linspace(0, 5, 46)
ax.hist(nul, edges, density=True, color="0.7", label=f"circular-shift null (n={len(nul)})")
ax.hist(obs, edges, density=True, histtype="step", color="k", lw=1.8,
        label=f"observed (n={len(obs)})")
pcsi = np.concatenate([mets[d].si.values[mets[d].is_place_cell.values] for d in DIRS])
ax.hist(pcsi, edges, density=True, histtype="step", color="crimson", lw=1.8,
        label=f"place cells (n={len(pcsi)})")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("probability density")
ax.set_title("(a) Spatial information vs shuffled null", fontsize=11)
ax.legend(fontsize=8)
u = stats.mannwhitneyu(obs, nul, alternative="greater")
ax.text(0.55, 0.55, f"Mann-Whitney U\np = {u.pvalue:.1e}", transform=ax.transAxes, fontsize=8)

# (b) observed SI against each unit's own 95th-percentile threshold
ax = axs[1]
handles = []
for d in DIRS:
    m = mets[d]
    ax.scatter(m.si_thresh[~m.is_place_cell], m.si[~m.is_place_cell], s=16,
               c="0.75", edgecolors="none")
    ax.scatter(m.si_thresh[m.is_place_cell], m.si[m.is_place_cell], s=18,
               c=COL[d], edgecolors="none")
    handles.append(plt.Line2D([], [], ls="", marker="o", color=COL[d],
                              label=f"{d}: {int(m.is_place_cell.sum())} place cells"))
handles.append(plt.Line2D([], [], ls="", marker="o", color="0.75", label="not classified"))
lim = [0, max(obs.max(), 5) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(0, 3)
ax.set_ylim(0, lim[1])
ax.set_xlabel("95th percentile of own shuffle null (bits/spike)")
ax.set_ylabel("observed SI (bits/spike)")
ax.set_title("(b) Per-unit significance test", fontsize=11)
ax.legend(handles=handles, fontsize=8, loc="upper left")

# (c) field width
ax = axs[2]
w = np.concatenate([mets[d].width.values[mets[d].is_place_cell.values] for d in DIRS]) * 100
ax.hist(w, np.arange(0, 105, 5), color="#4c72b0", edgecolor="w")
ax.axvline(np.median(w), color="k", ls="--")
ax.set_xlabel("place-field width (cm, >=50% of peak)")
ax.set_ylabel("count")
ax.set_title(f"(c) Field width, median {np.median(w):.0f} cm", fontsize=11)

# (d) peak in-field rate
ax = axs[3]
pr = np.concatenate([mets[d].peak_rate.values[mets[d].is_place_cell.values] for d in DIRS])
ax.hist(pr, np.arange(0, 55, 2.5), color="#dd8452", edgecolor="w")
ax.axvline(np.median(pr), color="k", ls="--")
ax.set_xlabel("in-field peak rate (Hz)")
ax.set_ylabel("count")
ax.set_title(f"(d) Peak rate, median {np.median(pr):.1f} Hz", fontsize=11)

# (e) sparsity
ax = axs[4]
sp_pc = np.concatenate([mets[d].sparsity.values[mets[d].is_place_cell.values] for d in DIRS])
sp_no = np.concatenate([mets[d].sparsity.values[~mets[d].is_place_cell.values] for d in DIRS])
e = np.linspace(0, 1, 26)
ax.hist(sp_no, e, color="0.75", label="not classified", density=True)
ax.hist(sp_pc, e, histtype="step", color="crimson", lw=1.8, label="place cells", density=True)
ax.set_xlabel(r"sparsity  $\langle\lambda\rangle^2/\langle\lambda^2\rangle$")
ax.set_ylabel("probability density")
ax.set_title(f"(e) Sparsity, place-cell median {np.median(sp_pc):.2f}", fontsize=11)
ax.legend(fontsize=8)

# (f) split-half reliability
ax = axs[5]
rl_pc = np.concatenate([mets[d].reliability.values[mets[d].is_place_cell.values] for d in DIRS])
rl_no = np.concatenate([mets[d].reliability.values[~mets[d].is_place_cell.values] for d in DIRS])
e = np.linspace(-0.6, 1, 33)
ax.hist(rl_no, e, color="0.75", label="not classified", density=True)
ax.hist(rl_pc, e, histtype="step", color="crimson", lw=1.8, label="place cells", density=True)
ax.axvline(pf.MIN_RELIABILITY, color="g", ls="--", lw=1)
ax.set_xlabel("odd vs even traversal map correlation")
ax.set_ylabel("probability density")
ax.set_title(f"(f) Within-session reliability\nplace-cell median r = {np.median(rl_pc):.2f}",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left")

# (g) where the fields sit
ax = axs[6]
for d in DIRS:
    pp = mets[d].peak_pos.values[mets[d].is_place_cell.values]
    ax.hist(pp, np.linspace(0, L, 17), histtype="step", lw=1.8, color=COL[d], label=d)
ax.set_xlabel("place-field peak position (m)")
ax.set_ylabel("count")
ax.set_title("(g) Fields cover the whole track", fontsize=11)
ax.legend(fontsize=8)

# (h) directionality: same cell in the two directions vs its own split-half.
# The reliability criterion puts a floor of 0.4 on the within-direction value, so
# this panel uses a place-cell mask that omits that criterion.
ax = axs[7]
e = np.linspace(-1, 1, 33)
norel = mets["right"].is_place_cell_norel.values | mets["left"].is_place_cell_norel.values
dc = dir_corr[norel]
dc = dc[np.isfinite(dc)]
rel_within = np.concatenate(
    [mets[d].reliability.values[mets[d].is_place_cell_norel.values] for d in DIRS]
)
rel_within = rel_within[np.isfinite(rel_within)]
ax.hist(dc, e, color="0.75", density=True,
        label=f"opposite directions (median {np.median(dc):.2f})")
ax.hist(rel_within, e, histtype="step", color="k", lw=1.8, density=True,
        label=f"same direction, odd vs even (median {np.median(rel_within):.2f})")
ax.axvline(np.median(dc), color="0.4", ls="--")
ax.axvline(np.median(rel_within), color="k", ls="--")
ax.set_xlabel("rate-map correlation")
ax.set_ylabel("probability density")
u2 = stats.mannwhitneyu(rel_within, dc, alternative="greater")
ax.set_title("(h) Fields are direction-specific:\nmaps repeat within, not across, direction "
             f"(p = {u2.pvalue:.1e})", fontsize=11)
ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle(f"{SESSION}: place-cell statistics ({len(pyr)} CA1 pyramidal cells, "
             f"{int(pc_either.sum())} place cells)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig04_statistics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig04 done")

summary = {
    "n_pyr": len(pyr),
    "n_pc_right": int(mets["right"].is_place_cell.sum()),
    "n_pc_left": int(mets["left"].is_place_cell.sum()),
    "n_pc_either": int(pc_either.sum()),
    "median_si_pc": float(np.median(pcsi)),
    "median_si_all": float(np.median(obs)),
    "median_si_null": float(np.median(nul)),
    "si_mwu_p": float(u.pvalue),
    "median_width_cm": float(np.median(w)),
    "median_peak_rate": float(np.median(pr)),
    "median_sparsity": float(np.median(sp_pc)),
    "median_reliability": float(np.median(rl_pc)),
    "median_dir_corr": float(np.median(dc)),
    "n_dir_corr": int(len(dc)),
    "dir_corr_vs_within_p": float(u2.pvalue),
    "median_within_dir_reliability": float(np.median(rel_within)),
}
for k, v in summary.items():
    print(f"  {k}: {v}")
with open("stats_summary.pkl", "wb") as fh:
    pickle.dump({"summary": summary, "dir_corr": dir_corr, "nulls": nulls}, fh)
