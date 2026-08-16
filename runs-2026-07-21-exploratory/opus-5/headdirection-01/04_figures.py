"""Figures 2-6 from the cached single-session results."""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import hd_lib


def wrap_nan(y_deg):
    """Insert NaNs at 0/360 wraps so plotted traces do not draw vertical jumps."""
    y = np.asarray(y_deg, dtype=float).copy()
    jump = np.abs(np.diff(y)) > 180
    y[1:][jump] = np.nan
    return y


with open("single_session_results.pkl", "rb") as f:
    R = pickle.load(f)

stats, tc = R["stats"], R["tc"]
hd_cells = R["hd_cells"]
name = R["session"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.dpi": 150})

# --------------------------------------------------------------------------- #
# Figure 2: example tuning curves
# --------------------------------------------------------------------------- #
sel = (stats.loc[hd_cells].sort_values("pref_dir").index.values)
examples = sel[np.linspace(0, len(sel) - 1, 8).astype(int)]
non_hd = stats[~stats["is_hd"]].sort_values("mvl").index.values[:4]

fig, axes = plt.subplots(3, 4, figsize=(13, 10.5),
                         subplot_kw={"projection": "polar"})
ang = np.asarray(tc.index)
ang_c = np.append(ang, ang[0])
for ax, u in zip(axes.ravel()[:8], examples):
    r = np.append(tc[u].values, tc[u].values[0])
    ra = np.append(R["tc_a"][u].values, R["tc_a"][u].values[0])
    rb = np.append(R["tc_b"][u].values, R["tc_b"][u].values[0])
    ax.plot(ang_c, ra, color="#4C72B0", lw=1, alpha=0.7)
    ax.plot(ang_c, rb, color="#DD8452", lw=1, alpha=0.7)
    ax.fill(ang_c, r, color="0.3", alpha=0.35)
    ax.plot(ang_c, r, color="k", lw=1.5)
    ax.set_title("unit %d\nMVL %.2f, %.1f bits/spk, peak %.0f Hz"
                 % (u, stats.loc[u, "mvl"], stats.loc[u, "hd_info"],
                    stats.loc[u, "peak_rate"]), pad=26, fontsize=9)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_yticklabels([])
for ax, u in zip(axes.ravel()[8:], non_hd):
    r = np.append(tc[u].values, tc[u].values[0])
    ax.fill(ang_c, r, color="#C44E52", alpha=0.35)
    ax.plot(ang_c, r, color="#C44E52", lw=1.5)
    ax.set_title("unit %d (not HD)\nMVL %.2f, %.2f bits/spk, peak %.0f Hz"
                 % (u, stats.loc[u, "mvl"], stats.loc[u, "hd_info"],
                    stats.loc[u, "peak_rate"]), pad=26, fontsize=9)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_yticklabels([])
fig.suptitle(f"{name}: head-direction tuning curves (postsubiculum)\n"
             "top two rows: eight HD cells spanning the compass "
             "(blue / orange = interleaved 60 s halves); bottom row: four "
             "non-directional units", y=0.99)
fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.86)
fig.savefig("fig02_tuning_curves.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 3: population statistics and significance
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
mvl_null, info_null = R["mvl_null"], R["info_null"]

ax = axes[0, 0]
ax.hist(mvl_null.ravel(), bins=60, range=(0, 1), color="0.7",
        density=True, label="time-shift null (all units)")
ax.hist(stats["mvl"], bins=30, range=(0, 1), color="#4C72B0", alpha=0.75,
        density=True, label="observed")
ax.axvline(R["mvl_threshold"], color="k", ls="--", lw=1)
ax.text(R["mvl_threshold"] + 0.02, ax.get_ylim()[1] * 0.9, "MVL = 0.3", fontsize=8)
ax.set_xlabel("mean vector length")
ax.set_ylabel("density")
ax.set_title("Directional modulation vs null")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.scatter(stats.loc[~stats["is_hd"], "mvl"], stats.loc[~stats["is_hd"], "hd_info"],
           s=22, color="0.6", label="not classified HD")
ax.scatter(stats.loc[stats["is_hd"], "mvl"], stats.loc[stats["is_hd"], "hd_info"],
           s=22, color="#4C72B0", label="HD cell")
ax.axvline(R["mvl_threshold"], color="k", ls="--", lw=1)
ax.set_xlabel("mean vector length")
ax.set_ylabel("directional information (bits/spike)")
ax.set_title("Two measures of tuning agree")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.scatter(stats["mean_rate"], stats["mvl"],
           c=np.where(stats["is_fs"], "#C44E52", "#4C72B0"), s=22)
ax.set_xscale("log")
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("mean vector length")
ax.set_title("HD tuning is carried by slow-firing cells\n"
             "(red = fast-spiking interneurons)")

ax = axes[1, 0]
z = stats["mvl_z"].values
ax.hist(z[stats["is_hd"]], bins=np.arange(0, max(z) + 5, 5), color="#4C72B0",
        label="HD cell")
ax.hist(z[~stats["is_hd"]], bins=np.arange(0, max(z) + 5, 5), color="0.6",
        label="not HD")
ax.set_xlabel("MVL z-score vs time-shift null")
ax.set_ylabel("units")
ax.set_title("Effect size relative to shuffles")
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 5, projection="polar")
axes[1, 1].remove()
edges = np.linspace(0, 2 * np.pi, 25)
cnt, _ = np.histogram(stats.loc[hd_cells, "pref_dir"], edges)
ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge",
       color="#4C72B0", edgecolor="w")
ax.set_title("Preferred directions of the\n%d HD cells" % len(hd_cells), pad=28)

ax = axes[1, 2]
ct = np.array([[np.sum(~stats["is_hd"] & ~stats["is_hd_dataset"]),
                np.sum(~stats["is_hd"] & stats["is_hd_dataset"])],
               [np.sum(stats["is_hd"] & ~stats["is_hd_dataset"]),
                np.sum(stats["is_hd"] & stats["is_hd_dataset"])]])
im = ax.imshow(ct, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, ct[i, j], ha="center", va="center",
                color="w" if ct[i, j] > ct.max() / 2 else "k", fontsize=14)
ax.set_xticks([0, 1], ["no", "yes"])
ax.set_yticks([0, 1], ["no", "yes"])
ax.set_xlabel("dataset's own is_head_direction flag")
ax.set_ylabel("this analysis")
ax.set_title("Agreement with the published labels\n(%.0f%% of units)"
             % (100 * (ct[0, 0] + ct[1, 1]) / ct.sum()))
fig.suptitle(f"{name}: classifying head-direction cells "
             f"({len(hd_cells)}/{len(stats)} units)", y=1.0)
fig.tight_layout()
fig.savefig("fig03_population_statistics.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 4: stability within and across environments
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ax = axes[0]
bins = np.linspace(-1, 1, 41)
ax.hist(stats.loc[hd_cells, "split_half_r"], bins=bins, color="#4C72B0",
        alpha=0.8, label="HD cells")
ax.hist(stats.loc[~stats["is_hd"], "split_half_r"], bins=bins, color="0.6",
        alpha=0.8, label="other units")
ax.set_xlabel("split-half correlation of tuning curve")
ax.set_ylabel("units")
ax.set_title("Within-session stability\n(interleaved 60 s blocks)")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(stats.loc[hd_cells, "cross_env_r"], bins=bins, color="#4C72B0",
        alpha=0.8, label="HD cells")
ax.hist(stats.loc[~stats["is_hd"], "cross_env_r"], bins=bins, color="0.6",
        alpha=0.8, label="other units")
ax.set_xlabel("square vs triangle correlation")
ax.set_ylabel("units")
ax.set_title("Across-environment stability")
ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(np.degrees(stats.loc[hd_cells, "pref_sq"]),
           np.degrees(stats.loc[hd_cells, "pref_tr"]), s=22, color="#4C72B0")
ax.plot([0, 360], [0, 360], "k--", lw=1)
ax.set_xlabel("preferred direction, square (deg)")
ax.set_ylabel("preferred direction, triangle (deg)")
ax.set_title("Preferred directions are preserved\nwhen the arena changes")
ax.set_xlim(0, 360)
ax.set_ylim(0, 360)

ax = axes[3]
shift = np.degrees(stats.loc[hd_cells, "pref_shift"])
ax.hist(shift, bins=np.arange(-180, 190, 10), color="#4C72B0")
ax.set_xlabel("preferred-direction shift, triangle - square (deg)")
ax.set_ylabel("HD cells")
ax.set_title("Shift is coherent across cells\n(median %.0f deg, IQR %.0f deg)"
             % (np.median(shift), np.subtract(*np.percentile(shift, [75, 25]))))
fig.suptitle(f"{name}: the directional signal is stable within and across environments",
             y=1.04)
fig.tight_layout()
fig.savefig("fig04_stability.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 5: population structure
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.4, wspace=0.5)

order = stats.loc[hd_cells].sort_values("pref_dir").index.values
M = np.stack([tc[u].values / tc[u].values.max() for u in order])
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, 360, 0, len(order)])
ax.set_xlabel("head direction (deg)")
ax.set_ylabel("HD cell (sorted by preferred direction)")
ax.set_title("Tuning curves tile the compass")
fig.colorbar(im, ax=ax, label="normalised rate", fraction=0.046, pad=0.03)

ax = fig.add_subplot(gs[0, 1])
d = np.degrees(R["pair_dphi"])
edges = np.arange(0, 190, 15)
idx = np.digitize(d, edges) - 1
for key, color, lab in [("pair_r", "#4C72B0", "wake"),
                        ("pair_r_nrem", "#C44E52", "NREM sleep")]:
    y = R[key]
    m = np.array([np.mean(y[idx == i]) for i in range(len(edges) - 1)])
    e = np.array([np.std(y[idx == i]) / np.sqrt(max(1, np.sum(idx == i)))
                  for i in range(len(edges) - 1)])
    ctr = edges[:-1] + 7.5
    ax.errorbar(ctr, m, yerr=e, color=color, marker="o", ms=4, label=lab)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("difference in preferred direction (deg)")
ax.set_ylabel("spike-count correlation (250 ms bins)")
ax.set_title("Co-firing is organised by tuning offset,\nand survives into sleep")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(d, R["pair_r"], s=1, alpha=0.15, color="#4C72B0")
ax.set_xlabel("difference in preferred direction (deg)")
ax.set_ylabel("wake spike-count correlation")
ax.set_title("All %d cell pairs (r = %.2f)"
             % (len(d), np.corrcoef(d, R["pair_r"])[0, 1]))

# Population activity as a moving bump
ax = fig.add_subplot(gs[1, :])
counts = R["counts_wake"]
t0 = R["triangle"].start[0] + 300
seg = nap.IntervalSet(t0, t0 + 40)
c = counts.restrict(seg)
Mt = c.values[:, [list(counts.columns).index(u) for u in order]].T
Mt = Mt / (Mt.max(axis=1, keepdims=True) + 1e-9)
ax.imshow(Mt, aspect="auto", origin="lower", cmap="Greys",
          extent=[0, 40, 0, len(order)], vmax=0.8)
pref_deg = np.degrees(stats.loc[order, "pref_dir"].values)
truehd = R["true_hd"].restrict(seg)
# Map the true head direction onto the cell ordering by nearest preferred direction.
row = np.interp(np.degrees(truehd.values), pref_deg, np.arange(len(order)))
row = wrap_nan(row / len(order) * 360) / 360 * len(order)
ax.plot(truehd.t - t0, row, color="#DD8452", lw=2, label="measured head direction")
ax.set_xlabel("time (s)")
ax.set_ylabel("HD cell (sorted by preferred direction)")
ax.set_title("Population activity is a single bump that moves with the head "
             "(40 s of foraging, 250 ms bins)")
ax.legend(loc="upper right", fontsize=8)
fig.suptitle(f"{name}: population structure of the head-direction ensemble", y=0.97)
fig.savefig("fig05_population_structure.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 6: decoding
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.3)

ax = fig.add_subplot(gs[0, :])
dec, tru = R["decoded"], R["true_hd"]
seg = nap.IntervalSet(t0, t0 + 120)
d1, t1 = dec.restrict(seg), tru.restrict(seg)
ax.plot(t1.t - t0, wrap_nan(np.degrees(t1.values)), "k-", lw=1.5, label="measured")
ax.plot(d1.t - t0, np.degrees(d1.values), ".", ms=2.5, color="#4C72B0",
        label="decoded from spikes")
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_title("Bayesian decoding, trained on the square arena and tested in the "
             "triangle arena (120 s shown)")
ax.legend(loc="upper right", fontsize=8, markerscale=3)

ax = fig.add_subplot(gs[1, 0])
bins = np.arange(-180, 185, 5)
ax.hist(R["err"], bins=bins, color="#4C72B0", density=True, label="decoder")
ax.hist(R["err_perm"], bins=bins, color="0.7", density=True, alpha=0.8,
        label="tuning curves permuted")
ax.set_xlabel("decoding error (deg)")
ax.set_ylabel("density")
ax.set_title("median |error| = %.1f deg\n(control %.0f deg)"
             % (np.median(np.abs(R["err"])), np.median(np.abs(R["err_perm"]))))
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
cum = np.sort(np.abs(R["err"]))
ax.plot(cum, np.linspace(0, 1, len(cum)), color="#4C72B0", label="decoder")
cum2 = np.sort(np.abs(R["err_perm"]))
ax.plot(cum2, np.linspace(0, 1, len(cum2)), color="0.6", label="control")
ax.set_xlabel("absolute decoding error (deg)")
ax.set_ylabel("cumulative fraction of bins")
ax.set_title("%.0f%% of bins within 30 deg"
             % (100 * np.mean(np.abs(R["err"]) < 30)))
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
for k, color in [("wake (triangle)", "#4C72B0"), ("REM", "#55A868"),
                 ("NREM", "#C44E52"), ("shuffled", "0.6")]:
    v = np.abs(R["drift"][k]) * R["bin_size"]  # degrees per bin
    x = np.sort(v)
    ax.plot(x, np.linspace(0, 1, len(x)), color=color, label=k)
ax.set_xlim(0, 180)
ax.set_xlabel("|step| per 200 ms bin (deg)")
ax.set_ylabel("cumulative fraction")
ax.set_title("The decoded signal moves smoothly\nduring REM, as it does in wake")
ax.legend(fontsize=8)

for col, (key, ep, color, title) in enumerate([
        ("dec_rem", R["rem"], "#55A868",
         "REM sleep: the internal direction sweeps smoothly"),
        ("dec_nrem", R["nrem"], "#C44E52",
         "NREM sleep: fast jumps between directions")]):
    lengths = ep.end - ep.start
    i = int(np.argmax(lengths))
    dur = min(120, lengths[i])
    win = nap.IntervalSet(ep.start[i], ep.start[i] + dur)
    dd = R[key].restrict(win)
    ax = fig.add_subplot(gs[2, col * 2:col * 2 + 2] if col == 0 else gs[2, 2])
    ax.plot(dd.t - win.start[0], wrap_nan(np.degrees(dd.values)), "-",
            lw=0.8, color=color, alpha=0.9)
    ax.plot(dd.t - win.start[0], np.degrees(dd.values), ".", ms=2, color=color)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("decoded direction (deg)")
    ax.set_ylim(0, 360)
    ax.set_title("%s\n(longest episode, %.0f s shown)" % (title, dur))
fig.suptitle(f"{name}: reading head direction out of the population", y=0.96)
fig.savefig("fig06_decoding.png", bbox_inches="tight")
plt.close(fig)
print("saved fig02-fig06")
