"""Stage 5: figures for the decoded replay trajectories."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import replaylib as R

SESSION = os.environ.get("SESSION", "Achilles-10252013")
FIG, CACHE = "figures", "cache"
DEC_BIN = 0.020

pf = np.load(f"{CACHE}/place_fields.npz")
place_ids, centers = pf["place_ids"], pf["centers"]
templates = {"right": pf["tc_right"], "left": pf["tc_left"]}
rip = np.load(f"{CACHE}/ripples.npz")
lfp_rate = float(rip["lfp_rate"])
best_ch = int(rip["best_ch"])

post_df = pd.read_csv(f"{CACHE}/replay_events_POSTEpoch.csv")
pre_df = pd.read_csv(f"{CACHE}/replay_events_PREEpoch.csv")
ctrl_df = pd.read_csv(f"{CACHE}/replay_events_POSTEpoch_idshuffle.csv")
stats = np.load(f"{CACHE}/replay_stats.npz")

h5, nwbfile, nwb = R.open_session(SESSION)
units = nwb["units"][list(place_ids)]
field_peak = centers[np.argmax(pf["tc_all"], axis=1)]
sort_by_field = np.argsort(field_peak)


def decode_event(row):
    ev = nap.IntervalSet(start=row.t_start, end=row.t_end)
    c = units.count(DEC_BIN, ev)
    tt = c.t - c.t[0]
    post = R.bayesian_decode(c.values, templates[row.direction], DEC_BIN)
    return post, tt, c


# ---------------------------------------------------------------- figure 4
# Six strongly significant POST events: LFP, spike raster ordered by place-field
# position, and the decoded posterior.
sig = post_df[post_df.significant].copy()
sig["absr"] = sig.r.abs()
examples = sig.sort_values("absr", ascending=False).head(6)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 6, height_ratios=[0.55, 1, 1.2], hspace=0.18, wspace=0.35)

for k, (_, row) in enumerate(examples.iterrows()):
    pad = 0.05
    seg = R.read_lfp(h5, [best_ch], row.t_start - pad, row.t_end + pad)
    filt, _ = R.ripple_envelope(seg.values[:, 0], lfp_rate)

    ax = fig.add_subplot(gs[0, k])
    ax.plot((seg.t - row.t_start) * 1000, seg.values[:, 0], "k", lw=0.6)
    ax.axvspan(0, (row.t_end - row.t_start) * 1000, color="tab:orange", alpha=0.2)
    ax.set_xticks([])
    ax.set_title(f"event {int(row.event)} ({row.direction})\n"
                 f"r={row.r:+.2f}, {abs(row.slope):.1f} m/s", fontsize=9)
    if k == 0:
        ax.set_ylabel("LFP")

    ax = fig.add_subplot(gs[1, k])
    for j, ui in enumerate(np.array(units.index)[sort_by_field]):
        st = units[ui].t
        st = st[(st >= row.t_start - pad) & (st <= row.t_end + pad)]
        if len(st):
            ax.plot((st - row.t_start) * 1000, np.full(len(st), j), "|",
                    color="k", ms=3, mew=0.8)
    ax.axvspan(0, (row.t_end - row.t_start) * 1000, color="tab:orange", alpha=0.2)
    ax.set_ylim(-1, len(place_ids))
    ax.set_xticks([])
    if k == 0:
        ax.set_ylabel("cell (by field position)")

    post, tt, _ = decode_event(row)
    ax = fig.add_subplot(gs[2, k])
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (tt[-1] + DEC_BIN) * 1000, centers[0], centers[-1]])
    dec = centers[np.argmax(post, axis=1)]
    ax.plot((tt + DEC_BIN / 2) * 1000, dec, "w.", ms=5)
    ax.set_xlabel("time in event (ms)")
    if k == 0:
        ax.set_ylabel("decoded position (m)")

fig.suptitle("Decoded trajectories during POST-sleep sharp-wave ripples "
             "(top: raw LFP; middle: place-cell raster; bottom: posterior)",
             fontsize=13, y=0.96)
fig.savefig(f"{FIG}/04_replay_examples.png", dpi=130, bbox_inches="tight")
print("wrote 04_replay_examples.png")

# ---------------------------------------------------------------- figure 5
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 26)
ax.hist(ctrl_df.r.abs(), bins=bins, density=True, histtype="step", lw=2,
        color="k", label=f"cell-ID shuffled (n={len(ctrl_df)})")
ax.hist(pre_df.r.abs(), bins=bins, density=True, alpha=0.55, color="tab:blue",
        label=f"PRE (n={len(pre_df)})")
ax.hist(post_df.r.abs(), bins=bins, density=True, alpha=0.55, color="tab:red",
        label=f"POST (n={len(post_df)})")
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.set_title("Sequence score\n"
             f"POST > control p = {float(stats['p_u_ctrl']):.1e}; "
             f"PRE > control p = {float(stats['p_u_prectrl']):.2f}", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1])
frac = [100 * ctrl_df.significant.mean(), 100 * pre_df.significant.mean(),
        100 * post_df.significant.mean()]
n = [len(ctrl_df), len(pre_df), len(post_df)]
err = [100 * np.sqrt(f / 100 * (1 - f / 100) / m) for f, m in zip(frac, n)]
ax.bar(["cell-ID\nshuffled", "PRE", "POST"], frac, yerr=err,
       color=["0.6", "tab:blue", "tab:red"], capsize=6)
ax.axhline(frac[0], color="k", ls="--", lw=1,
           label=f"empirical chance ({frac[0]:.1f}%)")
for i, (f, m) in enumerate(zip(frac, n)):
    ax.text(i, f + err[i] + 0.6, f"{f:.1f}%\n({int(f / 100 * m)}/{m})",
            ha="center", fontsize=9)
ax.set_ylabel("significant replay events (%)")
ax.set_title(f"Replay prevalence\nPOST vs control p = {float(stats['p_ctrl']):.1e}; "
             f"PRE vs control p = {float(stats['p_prectrl']):.2f}", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(frac) * 1.5)

ax = fig.add_subplot(gs[0, 2])
sp = np.abs(post_df.loc[post_df.significant, "slope"])
ax.hist(sp, bins=np.linspace(0, 30, 31), color="tab:red", alpha=0.8)
ax.axvline(np.median(sp), color="k", ls="--",
           label=f"median {np.median(sp):.1f} m/s")
ax.set_xlabel("replay speed |slope| (m/s)")
ax.set_ylabel("events")
ax.set_title("Virtual trajectory speed (POST)", fontsize=11)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 0])
sigp = post_df[post_df.significant]
ax.scatter(post_df.n_active, post_df.r.abs(), s=6, c="0.7", label="all candidates")
ax.scatter(sigp.n_active, sigp.r.abs(), s=8, c="tab:red", label="significant")
ax.set_xlabel("active place cells in event")
ax.set_ylabel("|weighted correlation|")
ax.set_title("Sequence score vs. event participation", fontsize=11)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 1])
forward = sigp.slope > 0
ax.bar(["forward\n(+slope)", "reverse\n(-slope)"],
       [forward.sum(), (~forward).sum()], color=["tab:green", "tab:purple"])
ax.set_ylabel("significant events")
ax.set_title("Direction of the decoded trajectory (POST)", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
tstart = post_df.t_start.values
t0 = tstart.min()
bw = 600.0
edges = np.arange(0, tstart.max() - t0 + bw, bw)
cand, _ = np.histogram(tstart - t0, bins=edges)
sigc, _ = np.histogram(post_df.loc[post_df.significant, "t_start"] - t0, bins=edges)
ax.plot(edges[:-1] / 60, cand / bw * 60, color="0.6", label="candidate events")
ax.plot(edges[:-1] / 60, sigc / bw * 60, color="tab:red", label="significant replay")
ax.set_xlabel("time into POST sleep (min)")
ax.set_ylabel("events / min")
ax.set_title("Replay across POST sleep", fontsize=11)
ax.legend(fontsize=9)

fig.suptitle("Replay statistics: POST-sleep ripples carry ordered spatial trajectories "
             "far above chance, PRE-sleep ripples do not", fontsize=13, y=0.97)
fig.savefig(f"{FIG}/05_replay_statistics.png", dpi=130, bbox_inches="tight")
print("wrote 05_replay_statistics.png")

# ---------------------------------------------------------------- figure 6
# Concatenated posteriors of the 24 best events, the "replay gallery".
best = sig.sort_values("absr", ascending=False).head(24)
fig, axes = plt.subplots(4, 6, figsize=(16, 9))
for ax, (_, row) in zip(axes.ravel(), best.iterrows()):
    post, tt, _ = decode_event(row)
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (tt[-1] + DEC_BIN) * 1000, centers[0], centers[-1]])
    ax.set_title(f"r={row.r:+.2f} p={row.pcyc:.3f}", fontsize=8, pad=2)
    ax.tick_params(labelsize=7)
for ax in axes[-1]:
    ax.set_xlabel("ms", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("pos (m)", fontsize=8)
fig.suptitle("Gallery of decoded POST-sleep replay events (posterior probability)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/06_replay_gallery.png", dpi=130, bbox_inches="tight")
print("wrote 06_replay_gallery.png")
