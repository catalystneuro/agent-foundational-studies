# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sharp-wave ripples and hippocampal replay in DANDI:000044
#
# This notebook demonstrates two linked phenomena in rat dorsal CA1, using
# streamed data from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)**: brief (~50 ms) 140–200 Hz oscillations in the
#    pyramidal layer, riding on a negative-going sharp wave in stratum radiatum,
#    that occur during non-REM sleep and quiet wakefulness and are accompanied by
#    a large transient increase in population firing.
# 2. **Replay**: within those events, place cells fire in sequences that trace out
#    the trajectory the animal ran on the track, compressed roughly twentyfold in
#    time, in both the forward and the reverse direction. Replay is more frequent
#    in sleep *after* the track session than in sleep before it.
#
# ## Dataset
#
# [DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences"
# (Grosmark & Buzsáki, *Science* 2016; the CRCNS `hc-11` data set). Four rats,
# eight sessions of bilateral silicon-probe recordings from dorsal CA1. Each
# session is a PRE sleep epoch, a maze epoch on a novel linear or circular track,
# and a POST sleep epoch, with spike-sorted units labelled excitatory or
# inhibitory, 128-channel LFP at 1250 Hz, position tracking, and manually scored
# sleep states.
#
# The primary analysis uses `Achilles_10252013` (1.6 m linear track, 120
# pyramidal cells); the last section repeats the whole pipeline on all five
# linear-track sessions.
#
# ## Access
#
# The NWB files are 5–9 GB each, so nothing is downloaded in full. Files are
# streamed from S3 with `remfile` + a local disk cache, and the LFP is stored
# with one channel per chunk, so reading a single channel for a ten-hour session
# transfers well under 100 MB.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from IPython import get_ipython
from scipy.signal import welch
from scipy.stats import binomtest, fisher_exact, wilcoxon

from dandi_io import open_session
from pipeline import (BIN, LINEAR_SESSIONS, analyze_session, bayesian_decode,
                      get_lfp, run_all_sessions)
from swr_lib import FS

# Every figure is written to disk with savefig. Under a Jupyter kernel we also
# switch on the inline backend so the figures appear in the notebook; as a plain
# script (`python swr_replay_analysis.py`) get_ipython() is None, nothing is
# displayed, and the PNGs are the only output.
_ip = get_ipython()
if _ip is not None:
    _ip.run_line_magic("matplotlib", "inline")

SESSION = "Achilles_10252013"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# %% [markdown]
# ## 1. Run the pipeline on the primary session
#
# `analyze_session` performs every step in one pass: pick the ripple channel,
# stream it, detect SWRs, linearize the position, build direction-specific place
# fields, Bayesian-decode every candidate event against those fields, and compute
# the sleep reactivation statistics. Each step is defined in `swr_lib.py` /
# `pipeline.py` and is described as we plot it below.

# %%
out = analyze_session(SESSION)
summary = out["summary"]
pd.Series(summary)

# %% [markdown]
# ## 2. Session structure and raw LFP
#
# The recording is a ~10 h session: 5 h of pre-task sleep, 34 min on the track,
# then 4 h of post-task sleep. Non-REM, REM and wake are scored by the original
# authors and shipped in the NWB file.
#
# The ripple channel is chosen automatically as the channel with the largest
# ripple-band (130–250 Hz) envelope standard deviation during a two-minute slab
# of post-task non-REM sleep. The sawtooth pattern across channels reflects
# probe geometry: the maximum sits at the top of each shank, in the CA1
# pyramidal layer.

# %%
nwbfile, nwb, h5 = open_session(SESSION)
lfp, filt, env = out["lfp"], out["filt"], out["env"]
ep_df = nwbfile.epochs.to_dataframe()
st_df = nwbfile.processing["behavior"]["states"].to_dataframe()
elec = nwbfile.electrodes.to_dataframe()
CH = summary["channel"]

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
colors = {"PREEpoch": "#8ecae6", "MazeEpoch": "#fb8500", "POSTEpoch": "#219ebc"}
for _, r in ep_df.iterrows():
    ax.axvspan(r.start_time / 60, r.stop_time / 60, color=colors[r.label], alpha=0.6)
    ax.text((r.start_time + r.stop_time) / 120, 1.3, r.label.replace("Epoch", ""),
            ha="center", fontsize=9)
for lab, c in [("Awake", "#adb5bd"), ("Non-REM", "#023047"), ("REM", "#e63946")]:
    sub = st_df[st_df.label == lab]
    ax.barh(0.3, (sub.stop_time - sub.start_time) / 60, left=sub.start_time / 60,
            height=0.35, color=c, label=lab)
ax.set_ylim(0, 1.6)
ax.set_yticks([])
ax.set_xlabel("time (min)")
ax.set_title(f"{SESSION}: session structure and scored sleep states", pad=12)
ax.legend(ncol=3, fontsize=8, loc="upper left", framealpha=0.9)

ax = fig.add_subplot(gs[1, 0])
ax.plot(out["channel_sd"] * 1e6, ".", ms=4, color="#023047")
ax.axvline(CH, color="#e63946", lw=1)
ax.set_xlabel("channel")
ax.set_ylabel("ripple-band env. SD (µV)")
ax.set_title(f"ripple-channel selection (best = {CH})", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
ax.plot(out["pos"].times() / 60, out["pos"].values, lw=0.5, color="#fb8500")
ax.set_xlabel("time (min)")
ax.set_ylabel("linear position (m)")
ax.set_title("position on the linear track", fontsize=10)

# 1 s of raw LFP around a large ripple
pk = out["peaks"].times()[np.argsort(out["peaks"].values)[-40]]
w = nap.IntervalSet(start=pk - 0.5, end=pk + 0.5)
for row, (sig, lab, c) in enumerate([(lfp, "raw LFP (µV)", "k"),
                                     (filt, "130–250 Hz (µV)", "#e63946")]):
    ax = fig.add_subplot(gs[2 + row, :])
    s = sig.restrict(w)
    ax.plot((s.times() - pk) * 1000, s.values * 1e6, lw=0.7, color=c)
    ax.set_ylabel(lab)
    ax.set_xlim(-500, 500)
    if row == 1:
        ax.set_xlabel("time from ripple peak (ms)")
    else:
        ax.set_title(f"raw trace, channel {CH}", fontsize=10)
fig.savefig("fig01_session_overview.png", dpi=150)

# %% [markdown]
# ## 3. Detecting sharp-wave ripples
#
# The channel is band-pass filtered at 130–250 Hz, the Hilbert envelope is
# smoothed with an 8 ms Gaussian, and the envelope is z-scored against the
# distribution during immobility (so that high-frequency activity during running
# cannot inflate the threshold). Events are periods where the envelope exceeds
# 2 SD, containing a peak above 5 SD, lasting 20–200 ms. Detection itself runs
# over the whole session, which lets us ask afterwards how the event rate depends
# on behavioural state.

# %%
ripples, peaks, pfreq = out["ripples"], out["peaks"], out["pfreq"]
dur = ripples.end - ripples.start
print(f"{len(ripples)} ripples, {summary['ripple_rate']:.1f}/min overall, "
      f"median duration {summary['ripple_dur_ms']:.0f} ms, "
      f"median peak frequency {summary['ripple_freq']:.0f} Hz")

# stratum radiatum channel on the same shank, for the sharp wave
same = np.where(elec.group_name.values == elec.group_name.iloc[CH])[0]
CH_RAD = int(same[np.argmax(elec.shank_electrode_number.values[same])])
lfp_rad = get_lfp(nwbfile, SESSION, CH_RAD)
print(f"pyramidal-layer channel {CH}, stratum radiatum channel {CH_RAD}")

# %%
half = int(0.25 * FS)
pk_idx = np.searchsorted(lfp.times(), peaks.times())
pk_idx = pk_idx[(pk_idx > half) & (pk_idx < lfp.shape[0] - half)]
lags = np.arange(-half, half) / FS
rta = np.stack([lfp.values[i - half:i + half] for i in pk_idx]).mean(0)
rta_rad = np.stack([lfp_rad.values[i - half:i + half] for i in pk_idx]).mean(0)
rta_f = np.stack([filt.values[i - half:i + half] for i in pk_idx]).mean(0)

rng = np.random.default_rng(0)
inside = np.concatenate([lfp.values[i - 31:i + 31] for i in pk_idx[:4000]])
ctrl = rng.choice(np.arange(half, lfp.shape[0] - half), size=4000, replace=False)
outside = np.concatenate([lfp.values[i - 31:i + 31] for i in ctrl])
f_in, p_in = welch(inside, fs=FS, nperseg=62)
f_out, p_out = welch(outside, fs=FS, nperseg=62)

pyr = out["pyr"]
mua_t = np.sort(np.concatenate([pyr[i].times() for i in pyr.index]))
peth_bins = np.arange(-0.25, 0.2501, 0.005)
mua = np.zeros(peth_bins.size - 1)
for tp in peaks.times():
    i0, i1 = np.searchsorted(mua_t, [tp - 0.25, tp + 0.25])
    mua += np.histogram(mua_t[i0:i1] - tp, bins=peth_bins)[0]
mua /= 0.005 * len(peaks)

# %%
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.3)

order = np.argsort(peaks.values)
for j, q in enumerate([0.5, 0.75, 0.95]):
    bi = order[int(q * len(order))]
    t_pk = peaks.times()[bi]
    w = nap.IntervalSet(start=t_pk - 0.25, end=t_pk + 0.25)
    ax = fig.add_subplot(gs[0, j])
    ax.plot(lfp.restrict(w).times() - t_pk, lfp.restrict(w).values * 1e6,
            lw=0.7, color="k", label="raw")
    ax.plot(filt.restrict(w).times() - t_pk, filt.restrict(w).values * 1e6 - 900,
            lw=0.7, color="#e63946", label="130–250 Hz")
    e = env.restrict(w)
    ax.plot(e.times() - t_pk, e.values * 1e6 * 3 - 1600, lw=1.0, color="#0077b6",
            label="envelope")
    ax.axvspan(ripples.start[bi] - t_pk, ripples.end[bi] - t_pk,
               color="#ffd166", alpha=0.4, zorder=0)
    ax.set_xlabel("time from peak (s)")
    if j == 0:
        ax.set_ylabel("µV (traces offset)")
        ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{int(q*100)}th percentile event, {peaks.values[bi]:.1f} SD",
                 fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.plot(lags, rta * 1e6, color="k", lw=1.2, label=f"pyramidal layer (ch {CH})")
ax.plot(lags, rta_rad * 1e6, color="#0077b6", lw=1.2, label=f"radiatum (ch {CH_RAD})")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("LFP (µV)")
ax.set_title(f"ripple-triggered average LFP (n={len(pk_idx)})", fontsize=10)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 1])
ax.plot(lags, rta_f * 1e6, color="#e63946", lw=1.0)
ax.set_xlim(-0.1, 0.1)
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("filtered LFP (µV)")
ax.set_title("ripple-triggered average, ripple band", fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.semilogy(f_in, p_in, color="#e63946", lw=1.2, label="inside ripples")
ax.semilogy(f_out, p_out, color="0.4", lw=1.2, label="random control")
ax.set_xlim(0, 400)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (V²/Hz)")
ax.set_title("LFP power spectrum", fontsize=10)
ax.legend(fontsize=8)

for j, (v, lab, ttl) in enumerate([
        (dur * 1000, "duration (ms)", f"durations (median {summary['ripple_dur_ms']:.0f} ms)"),
        (pfreq[np.isfinite(pfreq)], "peak frequency (Hz)",
         f"ripple frequency (median {summary['ripple_freq']:.0f} Hz)"),
        (peaks.values, "peak envelope (SD)", "event amplitude")]):
    ax = fig.add_subplot(gs[2, j])
    ax.hist(v, bins=40, color="#023047")
    ax.set_xlabel(lab)
    if j == 0:
        ax.set_ylabel("count")
    ax.set_title(ttl, fontsize=10)

ax = fig.add_subplot(gs[3, 0])
ax.plot(peth_bins[:-1] + 0.0025, mua, color="#fb8500", lw=1.2)
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("pyramidal MUA (Hz)")
ax.set_title("population firing around ripples", fontsize=10)

ax = fig.add_subplot(gs[3, 1])
labs = ["Non-REM", "REM", "Awake\nimmobile", "Awake\nrunning"]
vals = [summary["rate_nrem"], summary["rate_rem"], summary["rate_awake"],
        summary["rate_run"]]
ax.bar(labs, vals, color=["#023047", "#e63946", "#adb5bd", "#fb8500"])
ax.set_ylabel("ripples / min")
ax.set_title("state dependence", fontsize=10)
ax.tick_params(axis="x", labelsize=8)

ax = fig.add_subplot(gs[3, 2])
speed = out["speed"]
dt_pos = float(np.median(np.diff(out["pos"].times())))
pk_maze = nap.Ts(peaks.times()).restrict(speed.time_support)
v_at = speed.values[np.clip(np.searchsorted(speed.times(), pk_maze.times()),
                            0, speed.shape[0] - 1)]
edges = np.array([0, 2, 5, 10, 20, 30, 45, 100]) / 100
occ = np.histogram(speed.values, bins=edges)[0] * dt_pos
cnt = np.histogram(v_at, bins=edges)[0]
rate_v = np.where(occ > 10, cnt / np.maximum(occ, 1e-9) * 60, np.nan)
ctr = (edges[:-1] + edges[1:]) / 2 * 100
ax.plot(ctr, rate_v, "o-", color="#fb8500")
for xx, yy, nn in zip(ctr, rate_v, cnt):
    if np.isfinite(yy):
        ax.annotate(f"{nn}", (xx, yy), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, ha="center")
ax.set_xlabel("running speed (cm/s)")
ax.set_ylabel("ripples / min")
ax.set_title("speed dependence on the maze\n(n events above each point)", fontsize=10)

fig.suptitle(f"{SESSION}: sharp-wave ripple detection and validation", y=0.93)
fig.savefig("fig02_ripple_detection.png", dpi=150)

# %% [markdown]
# Everything about these events matches the textbook description of a
# sharp-wave ripple. The ripple-triggered average shows a positive deflection in
# the pyramidal layer against a large negative sharp wave in stratum radiatum,
# with a ~160 Hz oscillation on top; the power spectrum inside events has a clear
# 150–200 Hz peak absent from randomly chosen control windows; pyramidal
# population firing rises about fivefold at the ripple peak; and the event rate
# is highest in non-REM sleep, intermediate during quiet wakefulness, effectively
# zero in REM, and effectively zero while the animal runs.

# %% [markdown]
# ## 4. Place fields: the replay template
#
# Position is linearized by projecting the 2-D tracking onto the track axis. (The
# file ships a linearized series, but it is only defined while the animal runs;
# refitting the same projection from the 2-D series recovers the coordinate at
# every tracked sample, including the immobile periods at the reward wells where
# awake ripples happen.) Running epochs are periods above 5 cm/s, split by
# travel direction, and tuning curves are computed separately for each direction
# because CA1 place fields on a linear track are strongly directional.
#
# A cell counts as a place cell in a direction if its peak rate is at least 1 Hz,
# its Skaggs spatial information is at least 0.4 bits/spike, and its split-half
# tuning-curve correlation is at least 0.3.

# %%
tc, centers, stats = out["tc"], out["centers"], out["stats"]
place_units, track = out["place_units"], out["track"]
print(f"{len(place_units)} place cells out of {summary['n_pyr']} pyramidal cells; "
      f"{summary['n_laps']} full traversals")

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.5, height_ratios=[1.2, 1, 1])

for j, d in enumerate(["rightward", "leftward"]):
    ax = fig.add_subplot(gs[0, j])
    M = np.stack([np.nan_to_num(tc[d][u].values) for u in place_units])
    M = M / np.maximum(M.max(axis=1, keepdims=True), 1e-9)
    im = ax.imshow(M[np.argsort(np.argmax(M, axis=1))], aspect="auto",
                   origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, track, 0, len(place_units)])
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted)" if j == 0 else "")
    ax.set_title(f"{d} runs", fontsize=10)
    if j == 1:
        plt.colorbar(im, ax=ax, label="norm. rate")

ax = fig.add_subplot(gs[0, 2])
ax.hist(stats.si, bins=30, color="#adb5bd", label="all")
ax.hist(stats.query("place_cell").si, bins=30, color="#023047", label="place cells")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("count")
ax.set_title("spatial information", fontsize=10)
ax.legend(fontsize=8)

pc = stats.query("place_cell and peak > 3").sort_values("com")
for k, q in enumerate([0.25, 0.5, 0.75]):
    r = pc.iloc[int(q * (len(pc) - 1))]
    ax = fig.add_subplot(gs[1, k])
    for d, c in [("rightward", "#0077b6"), ("leftward", "#e63946")]:
        ax.plot(centers, tc[d][r.unit].values, color=c, lw=1.4, label=d)
    ax.set_title(f"unit {int(r.unit)}  (SI {r.si:.2f} bits/spike)", fontsize=9)
    ax.set_xlabel("position (m)")
    ax.set_ylabel("rate (Hz)")
    if k == 0:
        ax.legend(fontsize=7)

# one traversal, cells ordered by field position
run_ep, pos = out["run_ep"], out["pos"]
disp = np.array([pos.restrict(run_ep[i]).values[-1] - pos.restrict(run_ep[i]).values[0]
                 for i in range(len(run_ep))])
lap = int(np.argmax(disp > 0.5 * track))
t0, t1 = run_ep.start[lap] - 1, run_ep.end[lap] + 1
w = nap.IntervalSet(start=t0, end=t1)
M = np.stack([np.nan_to_num(tc["rightward"][u].values) for u in place_units])
order_u = np.array(place_units)[np.argsort(np.argmax(M, axis=1))]

ax = fig.add_subplot(gs[2, 0])
for i, u in enumerate(order_u):
    s = pyr[u].restrict(w).times()
    ax.plot(s - t0, np.full(s.size, i), "|", ms=4, color="k")
axp = ax.twinx()
axp.plot(pos.restrict(w).times() - t0, pos.restrict(w).values, color="#fb8500", lw=1)
axp.set_ylabel("position (m)", color="#fb8500")
ax.set_xlabel("time (s)")
ax.set_ylabel("cell (by field position)")
ax.set_title("sequential activation during one traversal", fontsize=9)

ax = fig.add_subplot(gs[2, 1])
ax.scatter(stats.peak, stats.stability, s=8,
           c=np.where(stats.place_cell, "#023047", "#adb5bd"))
ax.axhline(0.3, color="0.5", ls="--", lw=0.8)
ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("peak rate (Hz)")
ax.set_ylabel("split-half stability (r)")
ax.set_title("place-cell selection", fontsize=9)

ax = fig.add_subplot(gs[2, 2])
ax.hist(stats.query("place_cell").com, bins=20, color="#023047")
ax.set_xlabel("field peak position (m)")
ax.set_ylabel("count")
ax.set_title("field distribution along the track", fontsize=9)

fig.suptitle(f"{SESSION}: direction-specific place fields "
             f"({len(place_units)} place cells)", y=0.94)
fig.savefig("fig03_place_fields.png", dpi=150)

# %% [markdown]
# ## 5. Decoding replay
#
# Every ripple that falls outside a running epoch is a candidate replay event.
# Short ripples are padded symmetrically to at least 100 ms, spikes are binned at
# 20 ms, and events with fewer than 5 active place cells or fewer than 5 time
# bins are dropped. Each event is decoded with a flat-prior Poisson Bayesian
# decoder against each direction's template, and the sequence score is the
# posterior-weighted correlation between decoded position and time. The
# direction whose template gives the larger |r| is assigned to the event.
#
# Significance requires *both* of two shuffles to reject at p < 0.05:
#
# - **column-cycle shuffle**: each time bin's posterior is circularly shifted by
#   an independent random offset, which preserves the per-bin position
#   distribution but destroys sequence structure;
# - **field-identity shuffle**: place fields are randomly reassigned among cells,
#   which preserves each cell's spike train and the set of fields but destroys
#   the specific cell-to-position mapping.
#
# Before trusting the decoder we check it against ground truth: decoding the
# animal's actual position during running laps.

# %%
res = out["res"]
templates = {d: np.stack([np.nan_to_num(tc[d][u].values) for u in place_units], 1)
             for d in tc}
cells = np.array(place_units)
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


run_bin, true_pos, dec_pos = 0.25, [], []
for i in range(len(run_ep)):
    if abs(disp[i]) < 0.5 * track:
        continue
    C = counts(run_ep.start[i], run_ep.end[i], run_bin)
    if C.shape[0] < 2:
        continue
    post = bayesian_decode(C, templates["rightward" if disp[i] > 0 else "leftward"],
                           run_bin)
    tb = run_ep.start[i] + (np.arange(C.shape[0]) + 0.5) * run_bin
    true_pos.append(np.interp(tb, pos.times(), pos.values))
    dec_pos.append(centers[np.argmax(post, axis=1)])
true_pos, dec_pos = np.concatenate(true_pos), np.concatenate(dec_pos)
err = np.abs(true_pos - dec_pos)
print(f"decoder validation on {len(true_pos)} running bins: median error "
      f"{np.median(err)*100:.1f} cm, r = {np.corrcoef(true_pos, dec_pos)[0,1]:.2f}")

print(res.groupby("epoch").agg(n=("r", "size"), n_sig=("significant", "sum"),
                               frac_sig=("significant", "mean")))

# %% [markdown]
# ### Example events
#
# Each column below is one post-task-sleep replay event: the raw and ripple-band
# LFP, the spike raster with place cells ordered by field position, and the
# decoded posterior. The diagonal streak in the posterior is a trajectory swept
# across the track in roughly 100–200 ms, some forward and some reverse.

# %%
ordered = res[res.significant & (res.epoch == "POST")].sort_values("n_active",
                                                                  ascending=False)
fwd = ordered[ordered.forward].head(20).sort_values("r", key=abs, ascending=False)
rev = ordered[~ordered.forward].head(20).sort_values("r", key=abs, ascending=False)
examples = pd.concat([fwd.head(2), rev.head(2)])

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, height_ratios=[0.5, 1, 1])
for j, (_, ev) in enumerate(examples.iterrows()):
    t0, t1 = ev.start, ev.end
    tmpl = templates[ev.direction]
    order = np.argsort(np.argmax(tmpl.T, axis=1))
    post = bayesian_decode(counts(t0, t1, BIN), tmpl, BIN)

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
    ax.set_title(f"{'forward' if ev.forward else 'reverse'} replay of the\n"
                 f"{ev.direction} run · r = {ev.r:+.2f}", fontsize=9)

    ax = fig.add_subplot(gs[1, j])
    for i, ci in enumerate(order):
        s = spike_t[(spike_c == ci) & (spike_t >= t0 - 0.05) & (spike_t <= t1 + 0.05)]
        ax.plot((s - t0) * 1000, np.full(s.size, i), "|", ms=4, color="k")
    ax.set_xlim(-50, (t1 - t0) * 1000 + 50)
    ax.set_ylim(-1, len(cells))
    ax.set_xticklabels([])
    if j == 0:
        ax.set_ylabel("place cell (by field position)")

    ax = fig.add_subplot(gs[2, j])
    ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
              extent=[0, (t1 - t0) * 1000, 0, track], vmin=0,
              vmax=np.percentile(post, 99.5))
    ax.plot((np.arange(post.shape[0]) + 0.5) * BIN * 1000,
            centers[np.argmax(post, axis=1)], "o-", color="#4cc9f0", ms=3, lw=1)
    ax.set_xlabel("time in event (ms)")
    if j == 0:
        ax.set_ylabel("decoded position (m)")
fig.suptitle(f"{SESSION}: example replay events during post-task sleep", y=0.95)
fig.savefig("fig04_replay_examples.png", dpi=150)

# %% [markdown]
# ### Replay statistics

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
h = ax.hist2d(true_pos, dec_pos, bins=[np.linspace(0, track, 26)] * 2, cmap="viridis")
ax.plot([0, track], [0, track], "w--", lw=1)
ax.set_xlabel("true position (m)")
ax.set_ylabel("decoded position (m)")
ax.set_title(f"decoder validation on running laps\nmedian error "
             f"{np.median(err)*100:.1f} cm", fontsize=10)
plt.colorbar(h[3], ax=ax, label="bins")

ax = fig.add_subplot(gs[0, 1])
order_ep = ["PRE", "Maze", "POST"]
frac = [res[res.epoch == e].significant.mean() * 100 for e in order_ep]
ax.bar(order_ep, frac, color=["#8ecae6", "#fb8500", "#219ebc"])
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
for x, e in enumerate(order_ep):
    ax.annotate(f"n={(res.epoch == e).sum()}", (x, frac[x]), ha="center",
                textcoords="offset points", xytext=(0, 4), fontsize=8)
ax.set_ylabel("significant replay events (%)")
ax.set_title("trajectory content by epoch", fontsize=10)
ax.legend(fontsize=8)

a = res[res.epoch == "POST"].significant
b = res[res.epoch == "PRE"].significant
odds, p_fisher = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
print(f"POST vs PRE: {a.mean()*100:.1f}% vs {b.mean()*100:.1f}%, "
      f"odds ratio {odds:.2f}, Fisher p = {p_fisher:.2e}")

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
ax.hist([sig[sig.forward].r.abs(), sig[~sig.forward].r.abs()], bins=15, stacked=True,
        color=["#023047", "#e63946"],
        label=[f"forward ({sig.forward.sum()})", f"reverse ({(~sig.forward).sum()})"])
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

ax = fig.add_subplot(gs[1, 2])
edges_t = np.arange(0, res.start.max() + 600, 600)
for lab, mask, c in [("all candidate events", np.ones(len(res), bool), "#adb5bd"),
                     ("significant replay", res.significant.values, "#023047")]:
    ax.plot(edges_t[:-1] / 60, np.histogram(res.start[mask], bins=edges_t)[0] / 10,
            lw=1.4, color=c, label=lab)
for _, r in ep_df.iterrows():
    ax.axvspan(r.start_time / 60, r.stop_time / 60, alpha=0.12,
               color={"PREEpoch": "b", "MazeEpoch": "orange", "POSTEpoch": "g"}[r.label])
ax.set_xlabel("time (min)")
ax.set_ylabel("events / min")
ax.set_title("time course (blue PRE, orange maze, green POST)", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle(f"{SESSION}: replay statistics", y=0.96)
fig.savefig("fig05_replay_summary.png", dpi=150)

# %% [markdown]
# ## 6. A template-free cross-check: explained variance
#
# Decoded replay depends on the place-field template, so it is worth confirming
# the same conclusion without one. The classic measure (Kudrimoti, Barnes &
# McNaughton 1999) asks how much of the pairwise correlation structure among
# cells during running is present in post-task sleep once the pre-task
# correlation structure is partialled out (EV), compared with the time-reversed
# control that partials out POST instead (reverse EV). Any effect of stable
# anatomy or firing rates affects both equally; only experience-driven
# reactivation makes EV exceed reverse EV.

# %%
from pipeline import explained_variance

ev, rev, v = explained_variance(pyr[list(cells)], out["epochs"], out["states"],
                                out["run_ep"])
print(f"EV = {ev*100:.1f}%, reverse EV = {rev*100:.1f}%")

fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)
iu = np.triu_indices(len(cells), 1)
R = {k: np.zeros((len(cells), len(cells))) for k in v}
for k in v:
    R[k][iu] = v[k]
    R[k] += R[k].T
field_pos = np.array([centers[np.argmax(np.nan_to_num(tc["rightward"][u].values))]
                      for u in cells])
srt = np.argsort(field_pos)
for j, (k, lab) in enumerate([("pre", "PRE sleep"), ("run", "RUN"),
                              ("post", "POST sleep")]):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(R[k][np.ix_(srt, srt)], cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("cell (by field position)")
    if j == 0:
        ax.set_ylabel("cell (by field position)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="pairwise correlation" if j == 2 else None)

ax = fig.add_subplot(gs[1, :2])
ax.scatter(v["run"], v["pre"], s=3, alpha=0.3, color="#8ecae6", label="PRE sleep")
ax.scatter(v["run"], v["post"], s=3, alpha=0.3, color="#023047", label="POST sleep")
for k, c in [("pre", "#8ecae6"), ("post", "#023047")]:
    m, b_ = np.polyfit(v["run"], v[k], 1)
    xs = np.linspace(v["run"].min(), v["run"].max(), 10)
    ax.plot(xs, m * xs + b_, color=c, lw=2)
ax.set_xlabel("pairwise correlation during RUN")
ax.set_ylabel("pairwise correlation in sleep")
ax.legend(fontsize=8, markerscale=3)
ax.set_title("reinstatement of run-time correlation structure", fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.bar(["EV", "reverse EV"], [ev * 100, rev * 100], color=["#023047", "#adb5bd"])
ax.set_ylabel("explained variance (%)")
ax.set_title("EV / reverse EV", fontsize=10)
fig.suptitle(f"{SESSION}: sleep reactivation of run-time correlations "
             f"({len(cells)} place cells, 100 ms bins)", y=0.96)
fig.savefig("fig06_reactivation.png", dpi=150)

# %% [markdown]
# ## 7. All five linear-track sessions
#
# The same pipeline is run unchanged on every session of DANDI:000044 that used a
# linear track (the three circular-maze sessions are excluded because the
# linearization and the direction split would need different handling). Results
# are cached per session, so re-running this cell is cheap.

# %%
summary_all, allev = run_all_sessions(LINEAR_SESSIONS)
cols = ["session", "track", "n_pyr", "n_place", "n_ripples", "ripple_rate",
        "ripple_dur_ms", "ripple_freq", "rate_nrem", "rate_rem", "n_events",
        "frac_sig_PRE", "frac_sig_POST", "frac_forward", "ev", "rev"]
print(summary_all[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

a = allev[allev.epoch == "POST"].significant
b = allev[allev.epoch == "PRE"].significant
odds, p = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
w = wilcoxon(summary_all.frac_sig_POST, summary_all.frac_sig_PRE)
n_up = int((summary_all.frac_sig_POST > summary_all.frac_sig_PRE).sum())
p_sign = binomtest(n_up, len(summary_all), 0.5, alternative="greater").pvalue
print(f"\npooled POST vs PRE: {a.mean()*100:.1f}% vs {b.mean()*100:.1f}% "
      f"(odds ratio {odds:.2f}, Fisher p = {p:.2e})")
print(f"paired across {len(summary_all)} sessions: POST > PRE in "
      f"{n_up}/{len(summary_all)} (sign test p = {p_sign:.3f}; "
      f"Wilcoxon p = {w.pvalue:.3f})")

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
x = np.arange(len(summary_all))
short = [s.split("_")[0] + "\n" + s.split("_")[1][:4] for s in summary_all.session]

ax = fig.add_subplot(gs[0, 0])
for i, (k, lab, c) in enumerate([("rate_nrem", "non-REM", "#023047"),
                                 ("rate_awake", "awake immobile", "#adb5bd"),
                                 ("rate_rem", "REM", "#e63946"),
                                 ("rate_run", "running", "#fb8500")]):
    ax.bar(x + (i - 1.5) * 0.2, summary_all[k], 0.2, label=lab, color=c)
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("ripples / min")
ax.set_title("ripple rate by state", fontsize=10)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.bar(x - 0.2, summary_all.ripple_dur_ms, 0.4, color="#023047", label="duration (ms)")
ax.bar(x + 0.2, summary_all.ripple_freq, 0.4, color="#e63946", label="peak freq (Hz)")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_title("ripple properties", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.bar(x - 0.2, summary_all.n_pyr, 0.4, color="#adb5bd", label="pyramidal cells")
ax.bar(x + 0.2, summary_all.n_place, 0.4, color="#023047", label="place cells")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_title("recorded population", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
for i in range(len(summary_all)):
    ax.plot([0, 1], [summary_all.frac_sig_PRE[i] * 100,
                     summary_all.frac_sig_POST[i] * 100], "o-", color="#023047",
            alpha=0.8)
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["PRE sleep", "POST sleep"])
ax.set_ylabel("significant replay events (%)")
ax.set_title(f"replay increases after the track\n"
             f"(POST > PRE in {n_up}/{len(summary_all)} sessions, "
             f"sign test p = {p_sign:.3f})", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.bar(x - 0.2, summary_all.ev * 100, 0.4, color="#023047", label="EV")
ax.bar(x + 0.2, summary_all.rev * 100, 0.4, color="#adb5bd", label="reverse EV")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("explained variance (%)")
ax.set_title("reactivation of run-time correlations", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
fwd_frac = allev[allev.significant].groupby("session").forward.mean() * 100
ax.bar(range(len(fwd_frac)), fwd_frac, color="#023047")
ax.axhline(50, color="k", ls="--", lw=1)
ax.set_xticks(range(len(fwd_frac)))
ax.set_xticklabels([s.split("_")[0] + "\n" + s.split("_")[1][:4]
                    for s in fwd_frac.index], fontsize=8)
ax.set_ylabel("forward replay (%)")
ax.set_title("forward vs reverse", fontsize=10)

fig.suptitle(f"DANDI:000044 — {len(summary_all)} linear-track sessions, "
             f"{int(summary_all.n_ripples.sum())} ripples, "
             f"{len(allev)} decoded events", y=0.96)
fig.savefig("fig07_multisession.png", dpi=150)

# %% [markdown]
# ## Conclusions
#
# Sharp-wave ripples are recovered from the raw CA1 LFP with the expected
# signature in every session: ~50 ms events at ~160 Hz, a radiatum sharp wave, a
# severalfold transient increase in pyramidal population firing, a rate that is
# highest in non-REM sleep and near zero in REM and during running.
#
# Decoding the population activity inside those events against direction-specific
# place fields shows that a substantial minority of them contain a coherent
# spatial trajectory, well above what either of two shuffle controls allows.
# Those trajectories run in both directions along the track, sweep it in
# 100–200 ms (roughly twenty times faster than the animal ran it), and are more
# frequent in post-task sleep than in pre-task sleep. The template-free explained
# variance measure agrees: the pairwise correlation structure of running is
# reinstated in post-task sleep far more than the reverse control allows. Awake
# ripples during immobility on the track carry the highest fraction of decodable
# trajectories of all.
#
# Some events in pre-task sleep also pass the sequence test, at a rate slightly
# above the nominal 5%. That is consistent with the "preplay" literature, but it
# is also what one expects if the shuffles are slightly liberal, so the safer
# statement is the paired comparison: replay is reliably more common after the
# experience than before it, in every session tested.
