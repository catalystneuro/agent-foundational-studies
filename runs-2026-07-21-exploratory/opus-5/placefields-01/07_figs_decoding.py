"""Bayesian reconstruction of the animal's position from CA1 population spiking.

The place-field code is only useful if it can be read out. Rate maps are built
on one half of the traversals and used to decode position (and running
direction) on the held-out half, so nothing is fit and tested on the same data.
"""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import pf_lib as pf

SESSION = "Achilles_10252013"
BIN_SIZE = 0.25  # s, decoding time bin
BIN_SIZES = [0.05, 0.1, 0.25, 0.5, 1.0]

urls = pf.session_urls()
nwb, nwbfile, io = pf.load_session(urls[SESSION])
maze = pf.maze_epoch(nwbfile)
pos, L = pf.get_position(nwb, nwbfile)
vel, eps = pf.running_epochs(pos)
pyr = pf.select_pyramidal(nwb["units"], maze)
run = eps["run"]
fs = 1.0 / np.median(np.diff(pos.t))

# Restrict the decoder to units with a significant place field in at least one
# direction, which is the population the phenomenon is about.
rng = np.random.default_rng(0)
mets = {d: pf.direction_metrics(pyr, pos, eps[d], L, rng=rng)["metrics"] for d in ("right", "left")}
pc_mask = mets["right"].is_place_cell.values | mets["left"].is_place_cell.values
pc = pyr[list(np.asarray(pyr.index)[pc_mask])]
print(f"decoding from {len(pc)} place cells of {len(pyr)} pyramidal cells")


def joint_feature(ep):
    """(position, direction code) as a 2-column TsdFrame over `ep`."""
    p = pos.restrict(ep)
    v = np.interp(np.asarray(p.t), np.asarray(vel.t), vel.values)
    return nap.TsdFrame(
        t=p.t, d=np.c_[p.values, (v > 0).astype(float)], columns=["pos", "dir"], time_support=ep
    )


# Two-fold cross-validation over traversals. The animal alternates direction on
# successive runs, so the odd/even split has to be taken separately within each
# direction; splitting the pooled run epochs would put all rightward runs in one
# fold and all leftward runs in the other.
def half(sl):
    a = nap.IntervalSet(start=eps["right"].start[sl], end=eps["right"].end[sl])
    b = nap.IntervalSet(start=eps["left"].start[sl], end=eps["left"].end[sl])
    return a.union(b)


fold_a, fold_b = half(slice(None, None, 2)), half(slice(1, None, 2))
folds = [(fold_a, fold_b), (fold_b, fold_a)]


def full_bin_mask(t, ep, bin_size):
    """Drop the trailing partial bin of every interval, which is spike-starved."""
    ends = np.asarray(ep.end)
    idx = np.clip(np.searchsorted(ends, t, side="left"), 0, len(ends) - 1)
    return t + bin_size / 2 <= ends[idx] + 1e-9


def run_cv(bin_size, directional=True, group=None):
    """Cross-validated decoding. Returns pooled arrays of true/decoded values."""
    group = group if group is not None else pc
    out = {"t": [], "true_pos": [], "dec_pos": [], "true_dir": [], "dec_dir": [], "P": [], "bins": None}
    for train_ep, test_ep in folds:
        feat_tr = joint_feature(train_ep)
        if directional:
            tc = nap.compute_tuning_curves(
                group, feat_tr, bins=[pf.n_bins(L), 2], range=[(0, L), (-0.5, 1.5)],
                epochs=train_ep, fs=fs,
            )
        else:
            pos_only = nap.Tsd(t=feat_tr.t, d=feat_tr.values[:, 0], time_support=train_ep)
            tc = nap.compute_tuning_curves(
                group, pos_only, bins=pf.n_bins(L), range=[(0, L)],
                epochs=train_ep, fs=fs,
            )
        dec, P = nap.decode_bayes(tc, group, test_ep, bin_size=bin_size)
        feat_te = joint_feature(test_ep)
        tt = np.asarray(dec.t)
        keep_full = full_bin_mask(tt, test_ep, bin_size)
        tt = tt[keep_full]
        dec, P = dec[keep_full], np.asarray(P)[keep_full]
        out["t"].append(tt)
        out["true_pos"].append(np.interp(tt, np.asarray(feat_te.t), feat_te.values[:, 0]))
        out["true_dir"].append(np.interp(tt, np.asarray(feat_te.t), feat_te.values[:, 1]))
        if directional:
            out["dec_pos"].append(np.asarray(dec.values)[:, 0])
            out["dec_dir"].append(np.asarray(dec.values)[:, 1])
            out["P"].append(np.asarray(P).sum(axis=2))
        else:
            out["dec_pos"].append(np.asarray(dec).ravel())
            out["dec_dir"].append(np.full(len(tt), np.nan))
            out["P"].append(np.asarray(P))
        out["bins"] = np.asarray(tc.coords["pos"] if directional else tc.coords[tc.dims[1]])
    o = {k: (np.concatenate(v) if k in ("t", "true_pos", "dec_pos", "true_dir", "dec_dir")
             else v) for k, v in out.items() if k != "bins"}
    o["bins"] = out["bins"]
    o["P_folds"] = out["P"]
    order = np.argsort(o["t"])
    for k in ("t", "true_pos", "dec_pos", "true_dir", "dec_dir"):
        o[k] = o[k][order]
    o["err"] = np.abs(o["dec_pos"] - o["true_pos"])
    return o


res_dir = run_cv(BIN_SIZE, directional=True)
res_nodir = run_cv(BIN_SIZE, directional=False)

# Chance level: pair decoded values with positions drawn from the occupancy
# distribution instead of the simultaneous position.
chance = np.abs(
    rng.permutation(res_dir["dec_pos"]) - res_dir["true_pos"]
)

print(f"median error, directional maps    : {np.nanmedian(res_dir['err']) * 100:.1f} cm")
print(f"median error, non-directional maps: {np.nanmedian(res_nodir['err']) * 100:.1f} cm")
print(f"median error, chance              : {np.nanmedian(chance) * 100:.1f} cm")
dir_ok = np.round(res_dir["dec_dir"]) == np.round(res_dir["true_dir"])
print(f"running direction decoded correctly: {100 * np.nanmean(dir_ok):.1f}% of bins")

# Error as a function of the decoding time bin.
by_bin = {}
for b in BIN_SIZES:
    r = run_cv(b, directional=True)
    by_bin[b] = np.nanmedian(r["err"])
    print(f"  bin {b * 1000:4.0f} ms -> median error {by_bin[b] * 100:5.1f} cm")

# Error as a function of the number of cells used.
by_n = {}
allpc = list(np.asarray(pc.index))
for n in [2, 5, 10, 20, 40, len(allpc)]:
    errs = []
    for rep in range(3):
        sub = list(rng.choice(allpc, size=min(n, len(allpc)), replace=False))
        errs.append(np.nanmedian(run_cv(BIN_SIZE, True, group=pc[sub])["err"]))
    by_n[n] = (np.mean(errs), np.std(errs))
    print(f"  {n:3d} cells -> median error {by_n[n][0] * 100:5.1f} cm")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.3, 1.1], hspace=0.5, wspace=0.3)

# Posterior for a run of held-out traversals. Consecutive decoding bins are
# separated by the (undecoded) reward-area dwell time, so each traversal gets
# its own panel rather than being smeared onto one time axis.
train_ep, test_ep = folds[0]
feat_tr = joint_feature(train_ep)
tc = nap.compute_tuning_curves(
    pc, feat_tr, bins=[pf.n_bins(L), 2], range=[(0, L), (-0.5, 1.5)], epochs=train_ep, fs=fs
)
dec, P = nap.decode_bayes(tc, pc, test_ep, bin_size=BIN_SIZE)
t = np.asarray(dec.t)
kf = full_bin_mask(t, test_ep, BIN_SIZE)
Pm = np.asarray(P).sum(axis=2)[kf]  # marginalize over the direction dimension
dpos = np.asarray(dec.values)[kf, 0]
t = t[kf]

sgs = gs[0, :].subgridspec(1, 8, wspace=0.12)
for k in range(8):
    e0, e1 = float(test_ep.start[4 + k]), float(test_ep.end[4 + k])
    ax = fig.add_subplot(sgs[0, k])
    m = (t >= e0) & (t <= e1)
    edges_t = np.concatenate([t[m] - BIN_SIZE / 2, [t[m][-1] + BIN_SIZE / 2]]) - e0
    ax.pcolormesh(edges_t, np.linspace(0, L, pf.n_bins(L) + 1), Pm[m].T,
                  cmap="Greys", vmin=0, vmax=np.percentile(Pm[m], 99))
    pt = pos.restrict(nap.IntervalSet(start=e0, end=e1))
    ax.plot(np.asarray(pt.t) - e0, pt.values, color="#1f77b4", lw=2.2,
            label="true" if k == 0 else None)
    ax.plot(t[m] - e0, dpos[m], ".", color="crimson", ms=7,
            label="decoded" if k == 0 else None)
    ax.set_ylim(0, L)
    ax.set_xlim(0, e1 - e0)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    if k:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("track position (m)")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title(f"run {k + 1}", fontsize=9)
fig.text(0.5, 0.99,
         f"Bayesian decoding from {len(pc)} CA1 place cells, {BIN_SIZE * 1000:.0f} ms bins, "
         "eight consecutive held-out traversals (grey = posterior)",
         ha="center", fontsize=12)

# confusion matrix
ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, L, 33)
H, _, _ = np.histogram2d(res_dir["true_pos"], res_dir["dec_pos"], [edges, edges])
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
im = ax.imshow(H.T, origin="lower", extent=[0, L, 0, L], cmap="viridis", aspect="equal")
ax.plot([0, L], [0, L], "w--", lw=1)
ax.set_xlabel("true position (m)")
ax.set_ylabel("decoded position (m)")
ax.set_title("Confusion matrix\n(rows normalized)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

# error distributions
ax = fig.add_subplot(gs[1, 1])
e = np.linspace(0, L, 41)
ax.hist(chance * 100, e * 100, color="0.8", density=True, label=f"chance ({np.nanmedian(chance)*100:.0f} cm)")
ax.hist(res_nodir["err"] * 100, e * 100, histtype="step", color="k", lw=1.8, density=True,
        label=f"position-only maps ({np.nanmedian(res_nodir['err'])*100:.1f} cm)")
ax.hist(res_dir["err"] * 100, e * 100, histtype="step", color="crimson", lw=1.8, density=True,
        label=f"position x direction maps ({np.nanmedian(res_dir['err'])*100:.1f} cm)")
ax.set_xlabel("absolute decoding error (cm)")
ax.set_ylabel("probability density")
ax.set_title("Decoding error, cross-validated", fontsize=11)
ax.legend(fontsize=8)

# direction decoding
ax = fig.add_subplot(gs[1, 2])
acc = 100 * np.nanmean(dir_ok)
ax.bar([0, 1], [acc, 50], color=["crimson", "0.8"], width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["decoded", "chance"])
ax.set_ylabel("running direction correct (%)")
ax.set_ylim(0, 105)
ax.text(0, acc + 2, f"{acc:.1f}%", ha="center")
ax.set_title("Running direction is decodable\nfrom the same population", fontsize=11)

# error vs time bin
ax = fig.add_subplot(gs[2, 0])
ax.plot([b * 1000 for b in BIN_SIZES], [by_bin[b] * 100 for b in BIN_SIZES], "o-", color="crimson")
ax.set_xscale("log")
ax.set_xlabel("decoding time bin (ms)")
ax.set_ylabel("median error (cm)")
ax.set_title("Error vs time bin", fontsize=11)

# error vs number of cells
ax = fig.add_subplot(gs[2, 1])
ns = sorted(by_n)
ax.errorbar(ns, [by_n[n][0] * 100 for n in ns], yerr=[by_n[n][1] * 100 for n in ns],
            fmt="o-", color="#1f77b4", capsize=3)
ax.axhline(np.nanmedian(chance) * 100, color="0.6", ls="--", label="chance")
ax.set_xscale("log")
ax.set_xlabel("number of place cells in the decoder")
ax.set_ylabel("median error (cm)")
ax.set_title("Error vs population size\n(mean +/- SD over 3 random subsets)", fontsize=11)
ax.legend(fontsize=8)

# error as a function of true position
ax = fig.add_subplot(gs[2, 2])
bc = 0.5 * (edges[:-1] + edges[1:])
idx = np.digitize(res_dir["true_pos"], edges) - 1
med = [np.nanmedian(res_dir["err"][idx == k]) * 100 if np.sum(idx == k) > 5 else np.nan
       for k in range(len(bc))]
ax.plot(bc, med, "o-", color="crimson", ms=4)
ax.set_xlabel("true position (m)")
ax.set_ylabel("median error (cm)")
ax.set_ylim(0, None)
ax.set_title("Error vs position on the track", fontsize=11)

fig.text(0.5, 1.03, f"{SESSION}: position is recoverable from the CA1 place-cell population",
         ha="center", fontsize=14)
fig.savefig("fig05_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig05 done")

with open("decoding.pkl", "wb") as fh:
    pickle.dump(
        {
            "median_err_dir": float(np.nanmedian(res_dir["err"])),
            "median_err_nodir": float(np.nanmedian(res_nodir["err"])),
            "median_err_chance": float(np.nanmedian(chance)),
            "dir_accuracy": float(np.nanmean(dir_ok)),
            "by_bin": by_bin,
            "by_n": by_n,
            "n_place_cells": len(pc),
        },
        fh,
    )
