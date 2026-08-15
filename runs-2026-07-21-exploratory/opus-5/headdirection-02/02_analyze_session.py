"""Single-session head-direction analysis: classification, population structure,
cross-environment stability and Bayesian decoding."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import hd_analysis as ha
import hd_io

SESSION = "A3707"
NB_BINS = 120
P_THRESH = 0.01
STAB_THRESH = 0.5

assets = hd_io.list_assets("000939")
path, aid, size = [a for a in assets if SESSION in a[0]][0]
nwb, nwbfile, io = hd_io.open_session(aid, dandiset="000939")
eps = ha.get_epochs(nwbfile)
units = nwb["units"]
hd_all = nwb["head-direction"]

ep = eps["wake_square"]
hd = ha.clean_head_direction(hd_all, ep)
tc, stats, (tc1, tc2) = ha.session_stats(units, hd, ep, nb_bins=NB_BINS, n_draws=1000)
stats["author_hd"] = units.metadata["is_head_direction"].values.astype(bool)
stats["is_fs"] = units.metadata["is_fast_spiking"].values.astype(bool)
stats["is_exc"] = units.metadata["is_excitatory"].values.astype(bool)
stats["hd_cell"] = (stats.p_shift < P_THRESH) & (stats.stability > STAB_THRESH)
stats.to_csv("stats_example_session.csv")

n_hd = int(stats.hd_cell.sum())
print(f"{len(stats)} units, {n_hd} classified as head-direction cells "
      f"({100*n_hd/len(stats):.0f}%); authors flagged {int(stats.author_hd.sum())}")
print(pd.crosstab(stats.hd_cell, stats.author_hd))

# ------------------------------------------------------------------ figure 2
mvl_obs, mvl_null = ha.shift_null_mvl(units, hd, ep, nb_bins=NB_BINS, n_draws=1000)
example = stats.mvl.idxmax()
ex_i = list(stats.index).index(example)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

ax = axes[0, 0]
ax.hist(mvl_null[:, ex_i], bins=40, color="0.7", label="circular-shift null")
ax.axvline(mvl_obs[ex_i], color="crimson", lw=2, label="observed")
ax.set(xlabel="mean vector length", ylabel="shifts",
       title=f"Shift test, unit {example}\n(p < {1/len(mvl_null):.3f})")
ax.legend(fontsize=8)

ax = axes[0, 1]
bins = np.linspace(0, 1, 41)
ax.hist(stats.mvl, bins=bins, color="#1b6ca8", alpha=0.85, label="observed cells")
ax.hist(mvl_null.ravel(), bins=bins, weights=np.full(mvl_null.size, len(stats)/mvl_null.size),
        color="0.6", alpha=0.6, label="null (all shifts pooled)")
ax.set(xlabel="mean vector length", ylabel="units",
       title="Directional tuning strength vs chance")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.scatter(stats.mvl[~stats.hd_cell], stats.stability[~stats.hd_cell], s=18,
           color="0.6", label="not classified")
ax.scatter(stats.mvl[stats.hd_cell], stats.stability[stats.hd_cell], s=18,
           color="crimson", label="HD cell")
ax.axhline(STAB_THRESH, ls="--", color="k", lw=0.8)
ax.set(xlabel="mean vector length", ylabel="split-half tuning correlation",
       title="Classification criteria")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1, 0]
ax.scatter(stats.mvl, stats.hd_info, s=18,
           c=np.where(stats.hd_cell, "crimson", "0.6"))
ax.set(xlabel="mean vector length", ylabel="directional information (bits/spike)",
       title="Tuning strength and information")

ax = axes[1, 1]
kind = np.where(stats.is_fs, "fast-spiking", np.where(stats.is_exc, "excitatory", "unclassified"))
for i, lbl in enumerate(["excitatory", "fast-spiking", "unclassified"]):
    m = kind == lbl
    ax.scatter(np.random.default_rng(i).normal(i, 0.07, m.sum()), stats.mvl[m],
               s=16, c=np.where(stats.hd_cell[m], "crimson", "0.6"))
ax.set(xticks=[0, 1, 2], xticklabels=["excitatory", "fast-spiking", "unclassified"],
       ylabel="mean vector length",
       title="Tuning by cell type (red = HD cell here)")

ax = axes[1, 2]
exc = stats[stats.is_exc]
cm = pd.crosstab(exc.hd_cell, exc.author_hd).reindex(
    index=[False, True], columns=[False, True]).fillna(0).values
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="k" if cm[i, j] < cm.max() / 2 else "w", fontsize=13)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"],
       yticklabels=["no", "yes"], xlabel="author label: HD cell",
       ylabel="this analysis: HD cell",
       title="Excitatory cells only\n"
             f"agreement {100*(cm[0,0]+cm[1,1])/cm.sum():.0f}%")

fig.suptitle(f"Identifying head-direction cells  ({path.split('/')[-1]})", y=1.0)
fig.tight_layout()
fig.savefig("fig02_hd_cell_classification.png", dpi=150, bbox_inches="tight")
print("wrote fig02")

# ------------------------------------------------------------------ figure 3
hd_ids = stats.index[stats.hd_cell]
pref = stats.pref_dir[hd_ids].values
order = hd_ids[np.argsort(pref)]
norm = tc[order] / tc[order].max(0)

fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.2], wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(norm.values.T, aspect="auto", origin="lower", cmap="magma",
               extent=[0, 360, 0, len(order)])
ax.set(xlabel="head direction (deg)", ylabel="HD cell (sorted by preferred direction)",
       title=f"Tuning curves of all {len(order)} HD cells")
fig.colorbar(im, ax=ax, label="normalised rate")

ax = fig.add_subplot(gs[0, 1], projection="polar")
ax.hist(pref, bins=np.linspace(0, 2 * np.pi, 25), color="#1b6ca8")
ax.set_title("Preferred directions", pad=20, fontsize=10)
ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 2])
tch = tc[hd_ids].values
sd = tch.std(0)
keep = sd > 0
tch = (tch[:, keep] - tch[:, keep].mean(0)) / sd[keep]
corr = (tch.T @ tch) / tch.shape[0]
pref_k = pref[keep]
d = np.abs(ha.circ_diff(pref_k[:, None], pref_k[None, :]))
iu = np.triu_indices(len(pref_k), 1)
bins = np.linspace(0, np.pi, 19)
bi = np.digitize(d[iu], bins) - 1
m = np.array([corr[iu][bi == k].mean() for k in range(len(bins) - 1)])
s = np.array([corr[iu][bi == k].std() / max(np.sqrt((bi == k).sum()), 1)
              for k in range(len(bins) - 1)])
c = np.degrees((bins[:-1] + bins[1:]) / 2)
ax.scatter(np.degrees(d[iu]), corr[iu], s=3, color="0.8")
ax.errorbar(c, m, yerr=s, color="crimson", lw=2)
ax.axhline(0, ls="--", color="k", lw=0.8)
ax.set(xlabel="difference in preferred direction (deg)",
       ylabel="tuning-curve correlation", title="Pairwise tuning similarity")

fig.suptitle("Population structure of the head-direction code", y=1.03)
fig.savefig("fig03_population_structure.png", dpi=150, bbox_inches="tight")
print("wrote fig03")

# ------------------------------------------------------------------ figure 4
# same cells, different environment (square vs triangle arena)
ep2 = eps["wake_triangle"]
hd2 = ha.clean_head_direction(hd_all, ep2)
tc_t = ha.tuning_curves(units, hd2, ep2, nb_bins=NB_BINS)
mvl_t, pref_t = ha._mvl_and_pref(tc_t)
mvl_t = pd.Series(mvl_t, index=tc_t.columns)
pref_t = pd.Series(pref_t, index=tc_t.columns)

# the preferred direction is only meaningful for cells that are tuned in both
# environments, so compare that subset
both_ids = [i for i in hd_ids if mvl_t[i] > 0.3 and stats.mvl[i] > 0.3]
print(f"{len(both_ids)} of {len(hd_ids)} HD cells are tuned (MVL > 0.3) in both arenas")
delta = ha.circ_diff(pref_t[both_ids].values, stats.pref_dir[both_ids].values)
R, offset = ha.rotation_coherence(stats.pref_dir[both_ids].values,
                                  pref_t[both_ids].values)
resid = np.degrees(np.abs(ha.circ_diff(delta, offset)))
print(f"square vs triangle: common rotation {np.degrees(offset):.0f} deg, "
      f"coherence R = {R:.2f}, median |residual shift| = {np.median(resid):.1f} deg")

fig = plt.figure(figsize=(14, 4.4))
gs = fig.add_gridspec(1, 4, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(np.degrees(stats.pref_dir[both_ids]), np.degrees(pref_t[both_ids]), s=16,
           color="#1b6ca8")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set(xlabel="preferred direction, square (deg)",
       ylabel="preferred direction, triangle (deg)",
       title="Preferred directions rotate together\n"
             f"coherence R = {R:.2f} (n = {len(both_ids)})")

ax = fig.add_subplot(gs[0, 1])
ax.hist(np.degrees(ha.circ_diff(delta, offset)), bins=np.arange(-180, 185, 10),
        color="0.5")
ax.axvline(0, color="crimson", lw=1)
ax.set(xlabel="shift between environments (deg)", ylabel="HD cells",
       title="Shift after removing the\ncommon rotation "
             f"({np.degrees(offset):.0f} deg)")

ax = fig.add_subplot(gs[0, 2])
ax.scatter(stats.mvl[hd_ids], mvl_t[hd_ids], s=16, color="#1b6ca8")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="MVL, square", ylabel="MVL, triangle", title="Tuning strength")

ax = fig.add_subplot(gs[0, 3], projection="polar")
ex = (stats.mvl[both_ids] + mvl_t[both_ids]).idxmax()
for curve, lbl, c in [(tc, "square", "#1b6ca8"), (tc_t, "triangle", "#c0392b")]:
    th = np.append(curve.index.values, curve.index.values[0])
    r = np.append(curve[ex].values, curve[ex].values[0])
    ax.plot(th, r, color=c, label=lbl)
ax.set_title(f"unit {ex}", pad=20, fontsize=10)
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.25, 1.15))
ax.tick_params(labelsize=7)

fig.suptitle("The same cells keep their tuning in a different arena", y=1.05)
fig.savefig("fig04_cross_environment.png", dpi=150, bbox_inches="tight")
print("wrote fig04")

# ------------------------------------------------------------------ figure 5
# Bayesian decoding: tuning curves from the first half, decode the second half
mid = ep.start[0] + ep.tot_length() / 2
ep_train = nap.IntervalSet(start=ep.start[0], end=mid)
ep_test = nap.IntervalSet(start=mid, end=ep.end[-1])
hd_train = hd.restrict(ep_train)
hd_test = hd.restrict(ep_test)

_, stats_tr, _ = ha.session_stats(units, hd_train, ep_train, nb_bins=60, n_draws=400)
train_hd_ids = stats_tr.index[(stats_tr.p_shift < P_THRESH) & (stats_tr.stability > 0)]
other_ids = [i for i in stats_tr.index if i not in set(train_hd_ids)]
print(f"decoding with {len(train_hd_ids)} HD cells selected on training half")

BIN = 0.2


def decode(unit_ids):
    tcx = nap.compute_tuning_curves(units[list(unit_ids)], hd_train, bins=60,
                                    range=[(0, 2 * np.pi)], epochs=ep_train)
    dec, post = nap.decode_bayes(tcx, units[list(unit_ids)], ep_test, BIN)
    true = hd_test.bin_average(BIN, ep_test)
    ok = ~np.isnan(true.values)
    err = np.abs(ha.circ_diff(dec.values[ok], true.values[ok]))
    return dec, post, true, ok, err


dec, post, true, ok, err = decode(train_hd_ids)
_, _, _, ok_o, err_o = decode(other_ids)
rng = np.random.default_rng(0)
err_shuf = np.abs(ha.circ_diff(dec.values[ok], rng.permutation(true.values[ok])))
print(f"median decoding error: HD cells {np.degrees(np.median(err)):.1f} deg, "
      f"other cells {np.degrees(np.median(err_o)):.1f} deg, "
      f"shuffle {np.degrees(np.median(err_shuf)):.1f} deg")

sizes = [1, 2, 4, 8, 16, 32, min(64, len(train_hd_ids)), len(train_hd_ids)]
sizes = sorted(set([s for s in sizes if s <= len(train_hd_ids)]))
curve = []
for n in sizes:
    reps = []
    for r in range(8 if n < len(train_hd_ids) else 1):
        sel = np.random.default_rng(r).choice(train_hd_ids, n, replace=False)
        reps.append(np.degrees(np.median(decode(sel)[4])))
    curve.append((np.mean(reps), np.std(reps)))
curve = np.array(curve)

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32, height_ratios=[1, 1])

ax = fig.add_subplot(gs[0, :])
w = nap.IntervalSet(start=ep_test.start[0] + 200, end=ep_test.start[0] + 320)
pw = post.restrict(w)
ax.imshow(pw.values.T, aspect="auto", origin="lower", cmap="Greys",
          extent=[0, w.tot_length(), 0, 360])
t = hd_test.restrict(w)
ax.plot(t.index.values - w.start[0], np.degrees(t.values), ".", ms=2,
        color="crimson", label="true head direction")
ax.set(xlabel="time (s)", ylabel="head direction (deg)",
       title="Bayesian decoding from the HD-cell population (posterior in grey)")
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
bins = np.arange(0, 181, 5)
for e, lbl, c in [(err, "HD cells", "crimson"), (err_o, "other cells", "#1b6ca8"),
                  (err_shuf, "shuffled time", "0.6")]:
    ax.hist(np.degrees(e), bins=bins, histtype="step", lw=2, density=True,
            color=c, label=f"{lbl} (median {np.degrees(np.median(e)):.0f}$\\degree$)")
ax.set(xlabel="absolute decoding error (deg)", ylabel="density",
       title=f"Decoding error, {BIN*1000:.0f} ms bins")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.errorbar(sizes, curve[:, 0], yerr=curve[:, 1], marker="o", color="crimson")
ax.axhline(90, ls="--", color="0.5", lw=1)
ax.text(sizes[-1], 92, "chance", ha="right", fontsize=8, color="0.4")
ax.set(xscale="log", xlabel="number of HD cells", ylabel="median error (deg)",
       title="Decoding accuracy vs population size")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(np.degrees(true.values[ok]), np.degrees(dec.values[ok]), s=2, alpha=0.3,
           color="k")
ax.plot([0, 360], [0, 360], "--", color="crimson", lw=1)
ax.set(xlabel="true head direction (deg)", ylabel="decoded (deg)",
       title="Decoded vs true")

fig.suptitle("Head direction can be read out from the population", y=0.97)
fig.savefig("fig05_decoding.png", dpi=150, bbox_inches="tight")
print("wrote fig05")
