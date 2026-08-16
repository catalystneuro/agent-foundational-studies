"""Decoder validation, example replay events, and replay statistics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.stats import fisher_exact

from dandi_io import open_session
from swr_lib import (FS, bayesian_decode, linearize_position, ripple_envelope,
                     running_speed)

SESSION = "Achilles_10252013"
BIN = 0.020
TRACK = 1.6

nwbfile, nwb, h5 = open_session(SESSION)
units = nwb["units"]
pyr = units[units.cell_type == "excitatory"]
pos, TRACK = linearize_position(nwb)
speed = running_speed(pos)

pf = np.load("place_fields_Achilles_10252013.npz")
centers, all_units, place_units = pf["centers"], pf["units"], pf["place_units"]
sel = np.isin(all_units, place_units)
templates = {"rightward": np.nan_to_num(pf["tc_right"])[:, sel],
             "leftward": np.nan_to_num(pf["tc_left"])[:, sel]}
cells = all_units[sel]
res = pd.read_csv("replay_events_Achilles_10252013.csv")
rip = np.load("ripples_Achilles_10252013.npz")
CH = int(rip["channel"])
lfp = nap.Tsd(t=np.arange(np.load(f"lfp_ch{CH}_{SESSION}.npy").size) / FS,
              d=np.load(f"lfp_ch{CH}_{SESSION}.npy"))
filt, _ = ripple_envelope(lfp)

spike_t = np.concatenate([pyr[u].times() for u in cells])
spike_c = np.concatenate([np.full(pyr[u].shape[0], i) for i, u in enumerate(cells)])
o = np.argsort(spike_t)
spike_t, spike_c = spike_t[o], spike_c[o]


def counts(t0, t1, bin_size):
    n = int(round((t1 - t0) / bin_size))
    i0, i1 = np.searchsorted(spike_t, [t0, t0 + n * bin_size])
    tb = np.minimum(((spike_t[i0:i1] - t0) / bin_size).astype(int), n - 1)
    C = np.zeros((n, len(cells)))
    np.add.at(C, (tb, spike_c[i0:i1]), 1)
    return C


# --------------------------------------------- decoder validation on running laps
run_bin = 0.25
run_start, run_end, disp = pf["run_start"], pf["run_end"], pf["disp"]
true_pos, dec_pos = [], []
for s, e, dp in zip(run_start, run_end, disp):
    tmpl = templates["rightward" if dp > 0 else "leftward"]
    C = counts(s, e, run_bin)
    if C.shape[0] < 2:
        continue
    post = bayesian_decode(C, tmpl, run_bin)
    tb = s + (np.arange(C.shape[0]) + 0.5) * run_bin
    p_interp = np.interp(tb, pos.times(), pos.values)
    true_pos.append(p_interp)
    dec_pos.append(centers[np.argmax(post, axis=1)])
true_pos, dec_pos = np.concatenate(true_pos), np.concatenate(dec_pos)
err = np.abs(true_pos - dec_pos)
print(f"decoder validation on {len(true_pos)} running bins ({run_bin*1000:.0f} ms): "
      f"median error {np.median(err)*100:.1f} cm, r = "
      f"{np.corrcoef(true_pos, dec_pos)[0,1]:.2f}")

# ------------------------------------------------------------------ example events
ordered = res[res.significant & (res.epoch == "POST")].sort_values(
    "n_active", ascending=False)
fwd = ordered[ordered.forward].head(20).sort_values("r", key=abs, ascending=False)
rev = ordered[~ordered.forward].head(20).sort_values("r", key=abs, ascending=False)
examples = pd.concat([fwd.head(2), rev.head(2)])

M = np.stack([templates["rightward"][:, i] for i in range(len(cells))])
field_order = np.argsort(np.argmax(M, axis=1))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, height_ratios=[0.5, 1, 1])
for j, (_, ev) in enumerate(examples.iterrows()):
    t0, t1 = ev.start, ev.end
    tmpl = templates[ev.direction]
    order = np.argsort(np.argmax(tmpl.T, axis=1))
    C = counts(t0, t1, BIN)
    post = bayesian_decode(C, tmpl, BIN)

    ax = fig.add_subplot(gs[0, j])
    w = nap.IntervalSet(start=t0 - 0.05, end=t1 + 0.05)
    ax.plot((lfp.restrict(w).times() - t0) * 1000, lfp.restrict(w).values * 1e6,
            lw=0.6, color="k")
    ax.plot((filt.restrict(w).times() - t0) * 1000,
            filt.restrict(w).values * 1e6 - 500, lw=0.6, color="#e63946")
    ax.set_xlim(-50, (t1 - t0) * 1000 + 50)
    ax.set_xticks([])
    if j == 0:
        ax.set_ylabel("LFP (µV)")
    ax.set_title(f"{'forward' if ev.forward else 'reverse'} replay of the "
                 f"{ev.direction} run\n"
                 f"r = {ev.r:+.2f}, p = {max(ev.p_cycle, ev.p_id):.3f}", fontsize=9)

    ax = fig.add_subplot(gs[1, j])
    for i, ci in enumerate(order):
        st = spike_t[(spike_c == ci) & (spike_t >= t0 - 0.05) & (spike_t <= t1 + 0.05)]
        ax.plot((st - t0) * 1000, np.full(st.size, i), "|", ms=4, color="k")
    ax.set_xlim(-50, (t1 - t0) * 1000 + 50)
    ax.set_ylim(-1, len(cells))
    if j == 0:
        ax.set_ylabel("place cell (by field position)")
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[2, j])
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (t1 - t0) * 1000, 0, TRACK], vmin=0,
              vmax=np.percentile(post, 99.5))
    tb = (np.arange(post.shape[0]) + 0.5) * BIN * 1000
    ax.plot(tb, centers[np.argmax(post, axis=1)], "o-", color="#4cc9f0", ms=3, lw=1)
    ax.set_xlabel("time in event (ms)")
    if j == 0:
        ax.set_ylabel("decoded position (m)")
fig.suptitle(f"{SESSION}: example replay events during post-task sleep", y=0.95)
fig.savefig("fig04_replay_examples.png", dpi=150, bbox_inches="tight")
print("wrote fig04_replay_examples.png")

# ---------------------------------------------------------------------- summary
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
h = ax.hist2d(true_pos, dec_pos, bins=[np.linspace(0, TRACK, 26)] * 2,
              cmap="viridis")
ax.plot([0, TRACK], [0, TRACK], "w--", lw=1)
ax.set_xlabel("true position (m)")
ax.set_ylabel("decoded position (m)")
ax.set_title(f"decoder validation on running laps\nmedian error "
             f"{np.median(err)*100:.1f} cm", fontsize=10)
plt.colorbar(h[3], ax=ax, label="bins")

ax = fig.add_subplot(gs[0, 1])
order_ep = ["PRE", "Maze", "POST"]
frac = [res[res.epoch == e].significant.mean() * 100 for e in order_ep]
n_ev = [int((res.epoch == e).sum()) for e in order_ep]
ax.bar(order_ep, frac, color=["#8ecae6", "#fb8500", "#219ebc"])
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
for x, (f, n) in enumerate(zip(frac, n_ev)):
    ax.annotate(f"n={n}", (x, f), ha="center", textcoords="offset points",
                xytext=(0, 4), fontsize=8)
ax.set_ylabel("significant replay events (%)")
ax.set_title("trajectory content by epoch", fontsize=10)
ax.legend(fontsize=8)

a = res[res.epoch == "POST"].significant
b = res[res.epoch == "PRE"].significant
odds, p_fisher = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
print(f"POST vs PRE: odds ratio {odds:.2f}, Fisher p = {p_fisher:.2e}")

ax = fig.add_subplot(gs[0, 2])
for e, c in [("PRE", "#8ecae6"), ("Maze", "#fb8500"), ("POST", "#219ebc")]:
    v = np.sort(res[res.epoch == e].r.abs().values)
    ax.plot(v, 1 - np.arange(v.size) / v.size, lw=1.6, color=c, label=e)
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("fraction of events above")
ax.set_title("sequence scores (survival function)", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
sig = res[res.significant]
ax.hist([sig[sig.forward].r.abs(), sig[~sig.forward].r.abs()], bins=15,
        stacked=True, color=["#023047", "#e63946"],
        label=[f"forward ({sig.forward.sum()})",
               f"reverse ({(~sig.forward).sum()})"])
ax.set_xlabel("|r|")
ax.set_ylabel("significant events")
ax.set_title("replay direction", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(res.n_active, res.r.abs(), s=5,
           c=np.where(res.significant, "#023047", "#c8ccd0"))
ax.set_xlabel("active place cells in event")
ax.set_ylabel("|r|")
ax.set_title("event quality vs sequence score", fontsize=10)

# replay rate over the session
ax = fig.add_subplot(gs[1, 2])
edges = np.arange(0, res.start.max() + 600, 600)
for lab, mask, c in [("all candidate events", np.ones(len(res), bool), "#adb5bd"),
                     ("significant replay", res.significant.values, "#023047")]:
    n = np.histogram(res.start[mask], bins=edges)[0]
    ax.plot(edges[:-1] / 60, n / 10, lw=1.4, color=c, label=lab)
ep_df = nwbfile.epochs.to_dataframe()
for _, r in ep_df.iterrows():
    ax.axvspan(r.start_time / 60, r.stop_time / 60, alpha=0.12,
               color={"PREEpoch": "b", "MazeEpoch": "orange", "POSTEpoch": "g"}[r.label])
ax.set_xlabel("time (min)")
ax.set_ylabel("events / min")
ax.set_title("time course (blue PRE, orange maze, green POST)", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle(f"{SESSION}: replay statistics", y=0.96)
fig.savefig("fig05_replay_summary.png", dpi=150, bbox_inches="tight")
print("wrote fig05_replay_summary.png")
