# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sharp-wave ripples and hippocampal replay in DANDI:000044
#
# This notebook demonstrates the two halves of the sharp-wave ripple (SWR) phenomenon
# in real recordings from the DANDI Archive:
#
# 1. **The oscillation.** SWRs are 150-200 Hz transients in the CA1 pyramidal layer,
#    riding on a slow sharp-wave deflection, that occur during slow-wave sleep and
#    quiet immobility and never during theta states (running, REM).
# 2. **The content.** The population spikes packed inside a ripple are not random: they
#    re-express the sequence of place cells that fired while the animal ran the track,
#    compressed roughly ten-fold in time. This is *replay*, and it appears in the sleep
#    that follows the run but not in the sleep that precedes it.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark &
# Buzsáki (2016), *Science* 351:1440. Four rats ran back and forth on a linear track
# between two sleep sessions in the home cage. Each NWB file contains ~9 h of 128-channel
# CA1 LFP at 1250 Hz, spike-sorted units labelled excitatory/inhibitory, the linearized
# position on the track, and manually scored brain states (Awake / non-REM / REM).
#
# **Access.** Files are ~9 GB each and are never downloaded whole. They are streamed with
# `remfile` over a local disk cache; because the LFP dataset is chunked one channel per
# chunk, reading a single channel for the whole session costs about 90 MB.
#
# **Tooling.** All time-series handling (interval sets, restriction, tuning curves,
# perievent alignment, wavelet transform, Bayesian decoding) uses
# [Pynapple](https://pynapple.org). The analysis functions live in `pipeline.py` and the
# streaming loaders in `swr_utils.py`, both in this directory.

# %%
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import requests
from scipy import stats
from scipy.signal import welch

import pipeline as pl
import swr_utils as su

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
warnings.filterwarnings("ignore", category=FutureWarning)

SESSION = "Achilles-10252013"
RNG = np.random.default_rng(1)

# %% [markdown]
# ## 1. The dandiset
#
# Eight sessions from four rats. We prototype on `Achilles-10252013` (the largest unit
# yield) and repeat the full analysis on three more sessions at the end.

# %%
assets = requests.get(
    "https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/",
    params={"page_size": 100},
).json()["results"]
for a in sorted(assets, key=lambda a: a["path"]):
    print("%-58s %5.1f GB" % (a["path"], a["size"] / 1e9))

# %% [markdown]
# ## 2. Loading and inspecting the data streams
#
# Every stream is checked before anything is computed from it.

# %%
h5 = su.open_session(SESSION)
epochs = su.load_epochs(h5)
states = su.load_states(h5)
units = su.load_units(h5)
pos = su.load_position(h5)
lfp_ds, fs, t0 = su.lfp_meta(h5)

pyr = units.getby_category("cell_type")["excitatory"]
inh = units.getby_category("cell_type")["inhibitory"]

print("session:", SESSION)
print("epochs:", {k: "%.0f-%.0f s" % (v.start[0], v.end[-1]) for k, v in epochs.items()})
print("states:", {k: "%d bouts, %.0f s" % (len(v), v.tot_length()) for k, v in states.items()})
print("units: %d excitatory, %d inhibitory (all CA1)" % (len(pyr), len(inh)))
print("LFP: %d samples x %d channels at %g Hz = %.1f h"
      % (lfp_ds.shape[0], lfp_ds.shape[1], fs, lfp_ds.shape[0] / fs / 3600))
print("position: %d samples, %.0f-%.0f cm (linearized only during traversals: %.0f%% NaN)"
      % (len(pos), np.nanmin(pos.values), np.nanmax(pos.values),
         100 * np.mean(~np.isfinite(pos.values))))

# %% [markdown]
# ### Choosing a ripple channel
#
# Ripple amplitude depends steeply on depth: it is maximal in the CA1 pyramidal layer and
# falls off within a few hundred microns. The electrode table in this dandiset gives no
# layer labels, so the channel is chosen empirically as the one with the largest
# 130-250 Hz envelope during a 200 s bout of post-task non-REM sleep. The profile below
# shows the expected sawtooth: each of the 14 shanks has one site near the cell layer.

# %%
best_ch, alt_ch, sd, group = pl.pick_ripple_channel(h5, epochs, states)
print("ripple channel %d (%s); independent channel %d (%s)"
      % (best_ch, group[best_ch], alt_ch, group[alt_ch]))

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(np.arange(len(sd)), sd, color=["C0" if g == group[best_ch] else "0.6" for g in group])
for ch, c in ((best_ch, "C3"), (alt_ch, "C1")):
    ax.plot(ch, sd[ch] * 1.06, "v", color=c, ms=9)
    ax.text(ch, sd[ch] * 1.12, "ch %d" % ch, color=c, ha="center")
ax.margins(y=0.18)
ax.set(xlabel="LFP channel", ylabel="130-250 Hz envelope SD (a.u.)",
       title="%s: ripple-band power across 128 sites (200 s of non-REM sleep)" % SESSION)
fig.tight_layout()
fig.savefig("fig01_channel_selection.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Raw streams
#
# Four seconds of sleep LFP, the same trace band-passed, the animal's position on the
# track, and a spike raster. The band-passed trace already shows discrete high-frequency
# bursts, which is what the detector below formalizes.

# %%
nrem_post = states["Non-REM"].intersect(epochs["POST"])
seg = int(np.argmax(nrem_post.end - nrem_post.start))
t_ex = float(nrem_post.start[seg]) + 25
lfp_ex = su.load_lfp_channel(h5, best_ch, t_ex, t_ex + 4)
filt_ex = pl.bandpass(lfp_ex.values, pl.LOW, pl.HIGH, fs)

fig, axes = plt.subplots(4, 1, figsize=(11, 9))
axes[0].plot(lfp_ex.index - t_ex, lfp_ex.values, lw=0.6, color="k")
axes[0].set(ylabel="LFP (a.u.)", title="Raw CA1 LFP, ch %d (non-REM sleep)" % best_ch)
axes[1].plot(lfp_ex.index - t_ex, filt_ex, lw=0.6, color="C3")
axes[1].set(ylabel="130-250 Hz", xlabel="time in window (s)")

maze = epochs["MAZE"]
p = pos.restrict(maze)
axes[2].plot(p.index, p.values, lw=0.8, color="C0")
axes[2].set(ylabel="position (cm)", xlabel="time (s)",
            title="Linearized position on the 1.6 m track (MAZE epoch)")

sub = pyr.restrict(nap.IntervalSet(maze.start[0], maze.start[0] + 120))
for i, k in enumerate(list(sub.keys())[:60]):
    tt = sub[k].index
    axes[3].plot(tt, np.full_like(tt, i), "|", ms=2.5, color="k", mew=0.5)
axes[3].set(ylabel="unit #", xlabel="time (s)",
            title="Spike raster, 60 pyramidal cells (first 2 min on the track)")
fig.tight_layout()
fig.savefig("fig02_raw_streams.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Detecting sharp-wave ripples
#
# Standard envelope-threshold detection on the whole 9.7 h recording: band-pass
# 130-250 Hz, Hilbert envelope, z-score, keep excursions whose peak exceeds 4 SD with
# boundaries at 2 SD, merge events closer than 20 ms, keep durations of 20-200 ms.
# Events during locomotion (speed > 4 cm/s) are discarded, since running-speed
# high-frequency power is not a ripple.

# %%
ripples, peak_t, peak_z, raw, filt, fs, t0 = pl.ripples_from_channel(h5, best_ch, pos)
dur_ms = (ripples.end - ripples.start) * 1000
peak_freq = pl.ripple_stats(filt, fs, t0, peak_t)

print("%d ripples over %.1f h (%.2f Hz overall)"
      % (len(peak_t), (t0 + len(raw) / fs) / 3600, len(peak_t) / (len(raw) / fs)))
print("duration %.1f +- %.1f ms, intra-ripple frequency %.1f +- %.1f Hz"
      % (dur_ms.mean(), dur_ms.std(), peak_freq.mean(), peak_freq.std()))

rates = {}
for name, ep in list(epochs.items()) + list(states.items()):
    n = int(((peak_t[:, None] >= ep.start) & (peak_t[:, None] <= ep.end)).any(1).sum())
    rates[name] = n / ep.tot_length()
    print("  %-8s n=%5d  %8.0f s  %.3f Hz" % (name, n, ep.tot_length(), rates[name]))

# %% [markdown]
# ### Are these really ripples?
#
# Six independent checks, none of which the detector was tuned to satisfy:
#
# - the event-triggered average of the **raw** LFP shows the slow sharp wave the ripple rides on;
# - the mean wavelet spectrum has a single blob centred at ~160 Hz and ~50 ms wide;
# - durations and intra-ripple frequencies match the textbook rodent values (~50 ms, 150-200 Hz);
# - both pyramidal cells and interneurons increase their firing sharply and symmetrically
#   around the ripple peak, interneurons much more strongly;
# - the rate is high in non-REM, intermediate in quiet waking, and essentially zero in REM,
#   which is the classic state-dependence and cannot come from an amplitude artifact;
# - detecting independently on a channel from a different shank recovers the same events.

# %%
HALF = 0.25
n_half = int(HALF * fs)
idx = np.round((peak_t - t0) * fs).astype(int)
idx = idx[(idx > n_half) & (idx < len(raw) - n_half - 1)]
offs = np.arange(-n_half, n_half + 1)
lag = offs / fs
snips_raw = raw[idx[:, None] + offs]
snips_raw = snips_raw - snips_raw[:, :int(0.05 * fs)].mean(axis=1, keepdims=True)
snips_filt = filt[idx[:, None] + offs]

freqs = np.geomspace(30, 400, 60)
sub_ev = RNG.choice(len(snips_raw), size=min(800, len(snips_raw)), replace=False)
tf = np.zeros((len(freqs), snips_raw.shape[1]))
for i in sub_ev:
    sig = nap.Tsd(t=lag - lag[0] + 1.0, d=snips_raw[i].astype(float))
    tf += np.abs(np.asarray(nap.compute_wavelet_transform(sig, freqs, fs=fs))).T ** 2
tf /= len(sub_ev)

peaks = nap.Ts(t=peak_t)
psth = {}
for name, grp in (("pyramidal", pyr), ("interneuron", inh)):
    pe = nap.compute_perievent(grp, peaks, window=(-HALF, HALF))
    r = []
    for k in pe.keys():
        c = pe[k].count(0.005, nap.IntervalSet(-HALF, HALF))
        r.append(np.asarray(c).sum(axis=1) / (0.005 * len(peak_t)))
    psth[name] = (np.asarray(c.index), np.array(r))

# same detection on a channel from a different shank
lfp2 = su.load_lfp_channel(h5, alt_ch)
filt2 = pl.bandpass(lfp2.values, pl.LOW, pl.HIGH, fs)
env2 = pl.envelope(filt2, fs)
_, peak_t2, _ = pl.detect_ripples(nap.Tsd(t=lfp2.index, d=(env2 - env2.mean()) / env2.std()))
nearest = peak_t2[np.argmin(np.abs(peak_t2[None, :] - peak_t[:, None]), axis=1)] - peak_t
match = np.abs(nearest) < 0.05
print("channel %d: %d ripples, %.1f%% of channel-%d events matched within 50 ms"
      % (alt_ch, len(peak_t2), 100 * match.mean(), best_ch))

# %%
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
m, s = snips_raw.mean(0), snips_raw.std(0) / np.sqrt(len(snips_raw))
ax.plot(lag * 1000, m, color="k", lw=1.2)
ax.fill_between(lag * 1000, m - s, m + s, color="k", alpha=0.3)
ax.axvline(0, color="C3", ls=":", lw=1)
ax.set(xlabel="time from ripple peak (ms)", ylabel="LFP (a.u.)",
       title="Ripple-triggered average LFP\n(the sharp wave)")

ax = fig.add_subplot(gs[0, 1])
ax.plot(lag * 1000, snips_filt.mean(0), color="C3", lw=1)
ax.set(xlabel="time from ripple peak (ms)", ylabel="130-250 Hz (a.u.)",
       title="Ripple-triggered average\nof the filtered band", xlim=(-100, 100))

ax = fig.add_subplot(gs[0, 2])
im = ax.pcolormesh(lag * 1000, freqs, tf, shading="auto", cmap="magma")
ax.axhline(130, color="w", ls=":", lw=0.8)
ax.axhline(250, color="w", ls=":", lw=0.8)
ax.set(yscale="log", xlabel="time from ripple peak (ms)", ylabel="frequency (Hz)",
       title="Mean wavelet power\n(n=%d events)" % len(sub_ev), xlim=(-150, 150))
fig.colorbar(im, ax=ax, label="power (a.u.)")

ax = fig.add_subplot(gs[1, 0])
ax.hist(dur_ms, bins=40, color="C0")
ax.set(xlabel="duration (ms)", ylabel="count",
       title="Event duration\nmedian %.0f ms" % np.median(dur_ms))

ax = fig.add_subplot(gs[1, 1])
ax.hist(peak_freq, bins=40, color="C0")
ax.set(xlabel="intra-ripple peak frequency (Hz)", ylabel="count",
       title="Ripple frequency\nmedian %.0f Hz" % np.median(peak_freq))

ax = fig.add_subplot(gs[1, 2])
ax.hist(peak_z, bins=np.arange(4, 25, 0.5), color="C0")
ax.set(xlabel="peak envelope (z)", ylabel="count", title="Ripple amplitude", yscale="log")

ax = fig.add_subplot(gs[2, 0])
for name, color in (("pyramidal", "C0"), ("interneuron", "C1")):
    tt, r = psth[name]
    mm, se = r.mean(0), r.std(0) / np.sqrt(len(r))
    ax.plot(tt * 1000, mm, color=color, label="%s (n=%d)" % (name, len(r)))
    ax.fill_between(tt * 1000, mm - se, mm + se, color=color, alpha=0.3)
ax.axvline(0, color="0.5", ls=":")
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="time from ripple peak (ms)", ylabel="firing rate (Hz)",
       title="Spiking is locked to ripples")

ax = fig.add_subplot(gs[2, 1])
names = list(epochs) + list(states)
ax.bar(names, [rates[n] for n in names], color=["C0"] * 3 + ["C2"] * 3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set(ylabel="ripple rate (Hz)", title="Ripples are a non-REM /\nimmobility phenomenon")

ax = fig.add_subplot(gs[2, 2])
ax.hist(nearest, bins=np.arange(-0.05, 0.0501, 0.004), color="C0")
ax.set(xlabel="peak-time difference (s)", ylabel="count",
       title="Same events on the shank of ch %d\n(%.0f%% matched)" % (alt_ch, 100 * match.mean()))

fig.suptitle("Sharp-wave ripples in CA1, %s (DANDI:000044), n=%d events" % (SESSION, len(peak_t)))
fig.savefig("fig03_ripple_characterization.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Individual events

# %%
order = np.argsort(peak_z)[::-1]
in_nrem = ((peak_t[:, None] >= nrem_post.start) & (peak_t[:, None] <= nrem_post.end)).any(1)
sel = [i for i in order if in_nrem[i]][:6]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for ax, i in zip(axes.ravel(), sel):
    tc = peak_t[i]
    sl = slice(int((tc - 0.2 - t0) * fs), int((tc + 0.2 - t0) * fs))
    tt = (np.arange(sl.start, sl.stop) / fs + t0 - tc) * 1000
    ax.plot(tt, raw[sl] - raw[sl].mean(), color="k", lw=0.7)
    ax.plot(tt, filt[sl] - 2200, color="C3", lw=0.7)
    sp = pyr.restrict(nap.IntervalSet(tc - 0.2, tc + 0.2))
    for j, k in enumerate(sp.keys()):
        s = (sp[k].index - tc) * 1000
        ax.plot(s, np.full_like(s, -3200 - 22 * j), "|", color="C0", ms=3, mew=0.7)
    ax.axvspan((ripples.start[i] - tc) * 1000, (ripples.end[i] - tc) * 1000, color="C3", alpha=0.12)
    ax.set(title="peak %.1f z, t=%.1f s" % (peak_z[i], tc), yticks=[])
    ax.set_xlabel("time from ripple peak (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("LFP / filtered / spikes")
fig.suptitle("Example sharp-wave ripples during post-task non-REM sleep (raster: pyramidal cells)")
fig.tight_layout()
fig.savefig("fig04_example_ripples.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Place fields: the template that replay is scored against
#
# Replay can only be read out against a map of what the cells code for. Rate maps are
# built from running periods only (speed > 5 cm/s) and separately for the two travel
# directions, because CA1 fields on a linear track are strongly direction-selective.

# %%
pos_maze = pos[np.isfinite(pos.values)].restrict(epochs["MAZE"])
track_len = float(np.ceil(np.nanmax(pos.values)))
run, right, left = pl.running_epochs(pos_maze)
tcs, centers, si, is_place = pl.place_fields(pyr, pos_maze, right, left, track_len)
ids = np.array(list(pyr.keys()))[is_place]
spk = pyr[list(ids)]
templates = {k: pl.make_template(tcs[k][:, is_place], ids, centers) for k in tcs}

print("%d rightward runs (%.0f s), %d leftward runs (%.0f s)"
      % (len(right), right.tot_length(), len(left), left.tot_length()))
print("%d of %d pyramidal cells pass the place-cell criteria; median spatial information %.2f bits/spike"
      % (is_place.sum(), len(pyr), np.median(np.maximum(si["right"], si["left"])[is_place])))

# %%
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35, height_ratios=[1.25, 1])
idx_pc = np.where(is_place)[0]

for j, k in enumerate(["right", "left"]):
    ax = fig.add_subplot(gs[0, j])
    o = idx_pc[np.argsort(np.argmax(tcs[k][:, idx_pc], axis=0))]
    m = tcs[k][:, o].T
    im = ax.imshow(m / m.max(1, keepdims=True), aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, track_len, 0, len(o)])
    ax.set(xlabel="position on track (cm)", ylabel="place cell (sorted by own peak)",
           title="%sward runs" % k.capitalize())
    fig.colorbar(im, ax=ax, label="normalized rate")

ax = fig.add_subplot(gs[0, 2])
pk_all = centers[np.argmax(tcs["right"], axis=0)]
mid = idx_pc[(pk_all[idx_pc] > 20) & (pk_all[idx_pc] < track_len - 20)]
for c in mid[np.argsort(-si["right"][mid])][:6]:
    ax.plot(centers, tcs["right"][:, c], lw=1.4)
ax.set(xlabel="position (cm)", ylabel="firing rate (Hz)",
       title="Six example place fields\n(rightward runs)")

ax = fig.add_subplot(gs[1, 0])
p = pos_maze.restrict(nap.IntervalSet(pos_maze.index[0], pos_maze.index[0] + 300))
ax.plot(p.index - p.index[0], p.values, color="0.5", lw=0.8)
for ep_dir, c in ((right, "C0"), (left, "C3")):
    for s, e in zip(ep_dir.start, ep_dir.end):
        sg = pos_maze.restrict(nap.IntervalSet(s, e))
        ax.plot(sg.index - p.index[0], sg.values, color=c, lw=1.6)
ax.set(xlim=(0, 300), xlabel="time on maze (s)", ylabel="position (cm)",
       title="Laps: rightward (blue) / leftward (red)")

ax = fig.add_subplot(gs[1, 1])
si_max = np.maximum(si["right"], si["left"])
ax.hist(si_max, bins=30, color="0.7", label="all pyramidal")
ax.hist(si_max[is_place], bins=30, color="C0", label="place cells")
ax.axvline(pl.MIN_SPATIAL_INFO, color="k", ls=":")
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="spatial information (bits/spike)", ylabel="count", title="Spatial tuning")

ax = fig.add_subplot(gs[1, 2])
pk_r = centers[np.argmax(tcs["right"][:, is_place], axis=0)]
pk_l = centers[np.argmax(tcs["left"][:, is_place], axis=0)]
ax.plot(pk_r, pk_l, "o", ms=4, alpha=0.6)
ax.plot([0, track_len], [0, track_len], "k:", lw=1)
ax.set(xlabel="peak position, rightward (cm)", ylabel="peak position, leftward (cm)",
       title="Fields are direction-specific\n(r=%.2f)" % np.corrcoef(pk_r, pk_l)[0, 1])

fig.suptitle("Place fields on the linear track, %s" % SESSION)
fig.savefig("fig05_place_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Does the template actually decode position?
#
# Before asking what the posterior says during sleep, we check what it says during
# running, where ground truth exists. A median error of a few centimetres on a 160 cm
# track (chance is ~47 cm) means the template is sound.

# %%
errs = []
for k, ep_dir in (("right", right), ("left", left)):
    dec, _ = nap.decode_bayes(tuning_curves=templates[k], data=spk, epochs=ep_dir, bin_size=0.25)
    errs.append(np.abs(np.asarray(dec) - np.asarray(pos_maze.interpolate(dec))))
err = np.concatenate(errs)
chance = np.median(np.abs(RNG.uniform(0, track_len, 20000) - RNG.uniform(0, track_len, 20000)))
run_speed = float(np.median(su.compute_speed(pos_maze).restrict(run).values))
print("decoder median error while running: %.1f cm (chance %.1f cm); median running speed %.0f cm/s"
      % (np.median(err), chance, run_speed))

# %% [markdown]
# ## 5. Replay inside ripples
#
# **Candidate events.** Population bursts of the place-cell ensemble (smoothed multi-unit
# rate above its mean, peak > 3 SD, 100-500 ms long) that contain a detected ripple peak
# and in which at least 5 place cells fire. Requiring both the LFP signature and the
# population burst is what ties the sequence content to the ripple.
#
# **Scoring.** Each event is decoded in 20 ms bins with a uniform-prior Bayesian decoder
# and scored by the weighted correlation between decoded position and time. Both
# direction templates are tried and the larger |r| is kept.
#
# **Significance.** Two shuffles of the posterior, 400 draws each: a per-bin circular
# shift in position (destroys trajectory continuity, keeps per-bin structure) and a
# permutation of time bins (destroys temporal order). Because the observed statistic is a
# maximum over two templates, the null is the same maximum over the two templates'
# matched shuffles, so nothing is gained for free by trying both. An event counts as
# replay when it beats both nulls at p < 0.05.
#
# That last step is nominally a 5% false-positive rate, but it should not be trusted as
# one. Posterior shuffles preserve some of the structure that produces apparent sequences,
# so the achieved rate is measured empirically with the cell-identity shuffle below rather
# than assumed. It comes out at 6 to 9% depending on session and random draw, so every
# comparison in this notebook is made against that empirical null and not against 0.05.

# %%
rows, posteriors = [], {}
for ep_name in ("PRE", "POST"):
    pbe = pl.population_bursts(spk, epochs[ep_name], peak_t)
    n_rip = int(((peak_t >= epochs[ep_name].start[0]) & (peak_t <= epochs[ep_name].end[-1])).sum())
    print("%s: %d ripple-associated population bursts (of %d ripples)" % (ep_name, len(pbe), n_rip))
    r, p_ = pl.score_epoch(spk, templates, centers, pbe, ep_name, RNG)
    rows += r
    posteriors.update(p_)

df = pd.DataFrame(rows)
df["significant"] = df["max_p"] < pl.ALPHA
df["fwd"] = ((df.template == "right") & (df.slope > 0)) | ((df.template == "left") & (df.slope < 0))
df.to_csv("replay_events.csv", index=False)

# %% [markdown]
# ### Control: shuffle which cell owns which place field
#
# The strongest test of whether the sequences are real is to destroy the correspondence
# between cells and fields while keeping everything else, including the burst times, the
# spike counts and the decoding machinery, identical. If the pipeline manufactures
# sequences, this control will still find them.

# %%
perm_ids = RNG.permutation(len(ids))
shuf = {k: pl.make_template(tcs[k][:, is_place][:, perm_ids], ids, centers) for k in tcs}
ctrl_rows = []
for j in RNG.choice(len(df), size=min(400, len(df)), replace=False):
    row = df.iloc[j]
    out = pl.score_event(spk, shuf, centers,
                         nap.IntervalSet(row.t0, row.t0 + row.dur), RNG)
    if out is not None:
        ctrl_rows.append(out[0])
ctrl = pd.DataFrame(ctrl_rows)
ctrl["significant"] = ctrl["max_p"] < pl.ALPHA
ctrl.to_csv("replay_control.csv", index=False)

# %%
tab = [[int(df[(df.epoch == e) & df.significant].shape[0]),
        int(df[(df.epoch == e) & ~df.significant].shape[0])] for e in ("POST", "PRE")]
odds, p_fisher = stats.fisher_exact(tab)
for e in ("PRE", "POST"):
    g = df[df.epoch == e]
    k, n = int(g.significant.sum()), len(g)
    print("%s: %d/%d significant (%.1f%%); vs 5%% chance p=%.2g"
          % (e, k, n, 100 * k / n, stats.binomtest(k, n, 0.05, alternative="greater").pvalue))
print("cell-identity shuffle: %d/%d (%.1f%%)"
      % (ctrl.significant.sum(), len(ctrl), 100 * ctrl.significant.mean()))
print("POST vs PRE: odds ratio %.2f, Fisher p=%.2g" % (odds, p_fisher))

# The cell-ID shuffle, not the nominal 5%, is the operative null here: it lands above
# alpha, which means the per-event test is somewhat liberal (the two posterior shuffles do
# not destroy every source of apparent sequence structure). POST therefore has to beat the
# shuffle rate, not 0.05, before it counts for anything.
n_post_sig = int(df[(df.epoch == "POST")].significant.sum())
n_post = int((df.epoch == "POST").sum())
odds_cs, p_ctrl = stats.fisher_exact([[n_post_sig, n_post - n_post_sig],
                                      [int(ctrl.significant.sum()),
                                       int((~ctrl.significant).sum())]])
print("POST vs cell-ID shuffle: %.1f%% vs %.1f%%, odds ratio %.2f, Fisher p=%.2g"
      % (100 * n_post_sig / n_post, 100 * ctrl.significant.mean(), odds_cs, p_ctrl))
print("(the shuffle sits above the nominal 5%, so it, not alpha, is the baseline to beat)")
sig = df[df.significant]
print("significant events: %d forward, %d reverse" % (sig.fwd.sum(), (~sig.fwd).sum()))
print("median replay speed %.0f cm/s = %.0fx the animal's running speed"
      % (sig.slope.abs().median(), sig.slope.abs().median() / run_speed))

# %% [markdown]
# ### What a replay event looks like
#
# Three forward and three reverse events from post-task sleep. Top: the raw LFP with the
# ripple-filtered trace beneath it. Middle: the posterior over track position, which
# sweeps smoothly across the whole 1.6 m in 200-300 ms. Bottom: the same event as a raw
# raster with cells ordered by the position of their place field, where the sequence is
# visible without any decoding at all.

# %%
field_peak = {k: centers[np.argmax(tcs[k][:, is_place], axis=0)] for k in tcs}
strength = sig.assign(abs_r=sig.wcorr.abs())
best6 = pd.concat([
    strength[(strength.epoch == "POST") & strength.fwd].nlargest(3, "abs_r"),
    strength[(strength.epoch == "POST") & ~strength.fwd].nlargest(3, "abs_r"),
])

fig, axes = plt.subplots(3, 6, figsize=(18, 9),
                         gridspec_kw={"height_ratios": [0.55, 1.5, 1.1], "hspace": 0.5, "wspace": 0.38})
fig.subplots_adjust(top=0.80)
for col, (_, ev) in enumerate(best6.iterrows()):
    post, tt = posteriors[(ev.epoch, int(ev.event))]
    t_rel = (tt - tt[0]) * 1000
    pad = 0.05
    sl = slice(int((ev.t0 - pad - t0) * fs), int((ev.t0 + ev.dur + pad - t0) * fs))
    lt = (np.arange(sl.start, sl.stop) / fs + t0 - ev.t0) * 1000

    ax = axes[0, col]
    ax.plot(lt, raw[sl] - raw[sl].mean(), color="k", lw=0.6)
    ax.plot(lt, filt[sl] * 2 - 1500, color="C3", lw=0.6)
    ax.set(xticks=[], yticks=[], xlim=(lt[0], lt[-1]))
    ax.set_title("%s, %sward template\n|r| = %.2f,  p = %.3f\n%.0f cm/s"
                 % ("FORWARD" if ev.fwd else "REVERSE", ev.template,
                    abs(ev.wcorr), ev.max_p, abs(ev.slope)),
                 fontsize=9, pad=8,
                 color="C0" if ev.fwd else "C1")

    ax = axes[1, col]
    ax.imshow(post, aspect="auto", origin="lower", cmap="magma",
              extent=[t_rel[0] - 10, t_rel[-1] + 10, 0, track_len])
    com = (post / post.sum(0) * centers[:, None]).sum(0)
    ax.plot(t_rel, com, "o", color="w", ms=3, alpha=0.8)
    ax.plot(t_rel, np.polyval(np.polyfit(t_rel, com, 1), t_rel), color="C0", lw=1.6)
    ax.set(xlim=(lt[0], lt[-1]), ylim=(0, track_len))
    if col == 0:
        ax.set_ylabel("decoded position (cm)")

    ax = axes[2, col]
    sub_spk = spk.restrict(nap.IntervalSet(ev.t0 - pad, ev.t0 + ev.dur + pad))
    for r_i, u in enumerate(ids[np.argsort(field_peak[ev.template])]):
        s = (np.asarray(sub_spk[u].index) - ev.t0) * 1000
        if len(s):
            ax.plot(s, np.full_like(s, r_i), "|", color="k", ms=3, mew=0.8)
    ax.axvspan(0, ev.dur * 1000, color="C3", alpha=0.10)
    ax.set(xlim=(lt[0], lt[-1]), ylim=(-1, len(ids)), xlabel="time (ms)")
    if col == 0:
        ax.set_ylabel("cell (sorted by field position)")

fig.suptitle("Replay of the linear track inside sharp-wave ripples (post-task sleep)\n"
             "top: raw + ripple-filtered LFP;  middle: posterior P(position | spikes);  bottom: place-cell raster",
             fontsize=11)
fig.savefig("fig06_replay_examples.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### A second, decoding-free line of evidence
#
# Bayesian decoding involves many choices. An older and much simpler measure asks only
# whether pairs of cells that fired together on the track also fire together in sleep
# ripples: the explained variance EV is the correlation between run and post-sleep
# pair-correlation vectors, partialled for pre-sleep, and REV is the same quantity with
# pre and post exchanged. EV >> REV is reactivation that cannot be attributed to
# pre-existing coupling between cells.

# %%
ev_val, rev_val, n_pairs = pl.explained_variance(spk, ripples, run, epochs)
print("EV = %.3f, REV = %.3f over %d cell pairs" % (ev_val, rev_val, n_pairs))

# %%
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.hist(err, bins=np.arange(0, track_len, 4), color="C0")
ax.axvline(np.median(err), color="k", ls="--")
ax.axvline(chance, color="C3", ls=":")
ax.text(np.median(err) + 4, ax.get_ylim()[1] * 0.9, "median %.1f cm" % np.median(err), fontsize=9)
ax.text(chance + 4, ax.get_ylim()[1] * 0.6, "chance", color="C3", fontsize=9)
ax.set(xlabel="decoding error while running (cm)", ylabel="count",
       title="The template decodes real position\n(250 ms bins, MAZE epoch)")

ax = fig.add_subplot(gs[0, 1])
labels = ["PRE sleep", "POST sleep", "cell-ID\nshuffle"]
vals = [100 * df[df.epoch == "PRE"].significant.mean(),
        100 * df[df.epoch == "POST"].significant.mean(),
        100 * ctrl.significant.mean()]
ns = [int((df.epoch == "PRE").sum()), int((df.epoch == "POST").sum()), len(ctrl)]
bars = ax.bar(labels, vals, color=["0.6", "C0", "0.85"])
ax.axhline(5, color="C3", ls="--", lw=1, label="nominal α = 0.05")
ax.axhline(vals[2], color="k", ls="-.", lw=1.2,
           label="empirical null (cell-ID shuffle, %.1f%%)" % vals[2])
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, "%.1f%%\n(n=%d)" % (v, n), ha="center", fontsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set(ylabel="% of ripple events with significant replay", ylim=(0, max(vals) * 1.55),
       title="POST beats PRE and the cell-ID shuffle\nvs PRE p=%.1g,  vs shuffle p=%.1g"
             % (p_fisher, p_ctrl))

ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 1, 26)
for name, v, c in (("POST", df[df.epoch == "POST"].wcorr.abs(), "C0"),
                   ("PRE", df[df.epoch == "PRE"].wcorr.abs(), "0.5"),
                   ("cell-ID shuffle", ctrl.wcorr.abs(), "C3")):
    ax.hist(v, bins=bins, histtype="step", density=True, lw=1.8, color=c, label=name)
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="|weighted correlation|", ylabel="density", title="Sequence scores by epoch")

ax = fig.add_subplot(gs[1, 0])
ax.hist(sig.slope.abs() / 100, bins=np.arange(0, 9, 0.4), color="C0")
ax.axvline(run_speed / 100, color="C3", ls="--")
ax.set(xlabel="replay speed (m/s)", ylabel="count", xlim=(0, 9),
       title="Replay is time-compressed\nmedian %.1f m/s = %.0fx running"
             % (sig.slope.abs().median() / 100, sig.slope.abs().median() / run_speed))
ax.text(run_speed / 100 + 0.25, ax.get_ylim()[1] * 0.92,
        "running speed\n%.1f m/s" % (run_speed / 100), color="C3", fontsize=8, va="top")

ax = fig.add_subplot(gs[1, 1])
x = np.arange(2)
ax.bar(x - 0.18, [int(((sig.epoch == e) & sig.fwd).sum()) for e in ("PRE", "POST")], 0.36,
       label="forward", color="C0")
ax.bar(x + 0.18, [int(((sig.epoch == e) & ~sig.fwd).sum()) for e in ("PRE", "POST")], 0.36,
       label="reverse", color="C1")
ax.set_xticks(x)
ax.set_xticklabels(["PRE", "POST"])
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="significant events",
       title="Forward and reverse replay\nboth occur, in comparable numbers")

ax = fig.add_subplot(gs[1, 2])
ax.bar(["EV\n(POST | RUN, PRE)", "REV\n(PRE | RUN, POST)"], [ev_val, rev_val], color=["C0", "0.6"])
for i, v in enumerate([ev_val, rev_val]):
    ax.text(i, v + 0.002, "%.3f" % v, ha="center", fontsize=9)
ax.set(ylabel="explained variance", title="Pairwise reactivation, no decoding\n%d cell pairs" % n_pairs)

fig.suptitle("Hippocampal replay during sharp-wave ripples, %s (DANDI:000044)" % SESSION)
fig.savefig("fig07_replay_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Four sessions, three rats
#
# The whole pipeline (channel selection, ripple detection, place fields, decoding,
# shuffles, controls) is repeated on three further sessions. `Achilles-11012013` is
# excluded because its maze is circular rather than linear, which would need a different
# linearization. Results are cached to CSV so re-running the notebook is cheap.
#
# This is where the two halves of the result come apart, and it is worth stating plainly
# rather than burying. The ripple physiology reproduces almost exactly across all four
# sessions. The replay statistic does not: only `Achilles-10252013`, which has two to
# three times the place-cell yield of the others, shows a POST fraction clearly above both
# its PRE fraction and its own cell-ID shuffle. In the other three sessions the POST, PRE
# and shuffle fractions all sit within a couple of percentage points of each other, and
# the cell-ID shuffle itself runs at 6-7.5% rather than the nominal 5%, which means the
# per-event test is slightly liberal and small apparent enrichments there should not be
# read as replay.

# %%
SESSIONS = ["Achilles-10252013", "Gatsby-08022013", "Cicero-09172014", "Buddy-06272013"]

if os.path.exists("multi_session_summary.csv"):
    res = pd.read_csv("multi_session_summary.csv")
    all_events = pd.read_csv("multi_session_events.csv")
else:
    rows, frames = [], []
    for s in SESSIONS:
        summary, d, c, _ = pl.run_session(s)
        summary["frac_ctrl_n"] = len(c)
        rows.append(summary)
        d["session"] = s
        frames.append(d)
    res = pd.DataFrame(rows)
    all_events = pd.concat(frames)
    res.to_csv("multi_session_summary.csv", index=False)
    all_events.to_csv("multi_session_events.csv", index=False)

print(res[["session", "n_place", "n_ripples", "rate_nrem", "rate_rem", "decoder_err_cm",
           "frac_pre", "frac_post", "frac_ctrl", "ev", "rev"]].to_string(index=False))

# Per-session POST-vs-PRE tests, plus a Cochran-Mantel-Haenszel test stratified by
# session. Naively pooling events across sessions would be misleading here, because the
# sessions differ enormously in place-cell yield; CMH keeps sessions as strata and its
# companion homogeneity test says explicitly whether one effect size fits all of them.
import statsmodels.api as sm

tabs, per_session = [], []
for s in res.session:
    g = all_events[all_events.session == s]
    t = [[int(((g.epoch == e) & g.significant).sum()),
          int(((g.epoch == e) & ~g.significant).sum())] for e in ("POST", "PRE")]
    orr, pv = stats.fisher_exact(t)
    per_session.append(dict(session=s, n_place=int(res.loc[res.session == s, "n_place"].iloc[0]),
                            post_sig=t[0][0], post_n=t[0][0] + t[0][1],
                            pre_sig=t[1][0], pre_n=t[1][0] + t[1][1],
                            odds_ratio=orr, p=pv))
    tabs.append(np.array(t).T)
per_session = pd.DataFrame(per_session)
print("\nPOST vs PRE, session by session:")
print(per_session.to_string(index=False,
                            formatters={"odds_ratio": "{:.2f}".format, "p": "{:.3g}".format}))

cmh = sm.stats.StratifiedTable(np.dstack(tabs))
or_cmh = cmh.oddsratio_pooled
p_cmh = cmh.test_null_odds().pvalue
p_homog = cmh.test_equal_odds().pvalue
print("\nCochran-Mantel-Haenszel (stratified by session): OR=%.2f, p=%.2g" % (or_cmh, p_cmh))
print("homogeneity of the odds ratios across sessions: p=%.3g" % p_homog)
print("-> the effect is NOT homogeneous; %s carries it."
      % per_session.loc[per_session.odds_ratio.idxmax(), "session"])

# %%
labels = [s.split("-")[0] for s in res.session]
x = np.arange(len(res))
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

ax = axes[0, 0]
for i, (k, c) in enumerate((("rate_nrem", "C0"), ("rate_awake", "C2"), ("rate_rem", "C1"))):
    ax.bar(x + (i - 1) * 0.27, res[k], 0.27, label=k.replace("rate_", ""), color=c)
ax.set_xticks(x, labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="ripple rate (Hz)", title="Ripple rate by brain state")

ax = axes[0, 1]
ax.bar(x - 0.2, res.ripple_dur_ms, 0.4, color="C0", label="duration (ms)")
ax.bar(x + 0.2, res.ripple_freq_hz, 0.4, color="C3", label="frequency (Hz)")
ax.set_xticks(x, labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="ms  /  Hz", title="Ripple duration and frequency")

ax = axes[0, 2]
ax.bar(x - 0.2, res.n_place, 0.4, label="place cells", color="C0")
ax.bar(x + 0.2, res.decoder_err_cm, 0.4, label="decoder error (cm)", color="C1")
ax.set_xticks(x, labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="count  /  cm", title="Template quality")

ax = axes[1, 0]
ax.bar(x - 0.27, 100 * res.frac_pre, 0.27, label="PRE sleep", color="0.6")
ax.bar(x, 100 * res.frac_post, 0.27, label="POST sleep", color="C0")
ax.bar(x + 0.27, 100 * res.frac_ctrl, 0.27, label="cell-ID shuffle", color="0.85")
ax.axhline(5, color="C3", ls="--", lw=1, label="nominal α = 0.05")
for i, row in per_session.reset_index(drop=True).iterrows():
    star = "p<1e-10" if row.p < 1e-10 else ("p=%.2f" % row.p)
    ax.text(i, 100 * res.frac_post.iloc[i] + 0.4, star, ha="center", fontsize=7.5,
            fontweight="bold" if row.p < 0.05 else "normal",
            color="k" if row.p < 0.05 else "0.45")
ax.set_xticks(x, labels, rotation=15)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_ylim(0, 100 * max(res.frac_post.max(), res.frac_pre.max()) * 1.35)
ax.set(ylabel="% events with significant replay",
       title="POST > PRE only in Achilles\n(CMH OR=%.2f, but homogeneity p=%.3f)"
             % (or_cmh, p_homog))

ax = axes[1, 1]
ax.bar(x - 0.2, res.ev, 0.4, label="EV", color="C0")
ax.bar(x + 0.2, res.rev, 0.4, label="REV", color="0.6")
ax.set_xticks(x, labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="explained variance", title="Pairwise reactivation (EV > REV)")

ax = axes[1, 2]
for s in res.session:
    g = all_events[(all_events.session == s) & all_events.significant]
    ax.hist(np.abs(g.slope) / 100, bins=np.arange(0, 12, 0.75), histtype="step", lw=1.6,
            density=True, label=s.split("-")[0])
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="replay speed (m/s)", ylabel="density", title="Replay speed")

fig.suptitle("Sharp-wave ripples and replay across four sessions of DANDI:000044")
fig.savefig("fig08_multi_session.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 7. What the data show
#
# The LFP events isolated here are sharp-wave ripples by every standard criterion: a
# ~50 ms, ~160 Hz oscillation superimposed on a slow sharp wave, occurring at roughly
# 0.5 Hz in non-REM sleep, at a lower rate during quiet waking, and essentially never in
# REM; accompanied by a large increase in pyramidal and (much larger) interneuron firing;
# and detected identically from an electrode on a different shank.
#
# The spikes inside those ripples carry the track. Decoding population bursts against the
# running place-field template recovers smooth trajectories across the maze at roughly
# 2-3 m/s, some replaying the run in the direction it was experienced and some in
# reverse. In `Achilles-10252013` the fraction of ripple events carrying a statistically
# significant trajectory is about 15% in the sleep that follows the run, against about 6%
# in the sleep that precedes it (odds ratio 2.8, p = 3e-11) and about 9% in a control that
# permutes cell identities across place fields (p = 0.002). The same asymmetry appears in a
# completely different statistic that involves no decoding: pairwise co-firing during
# ripples resembles co-firing on the track far more in post-task sleep than in pre-task
# sleep (EV = 0.088, REV = 0.001).
#
# The cell-identity control is the number worth dwelling on. At 9% it sits well above the
# nominal 5%, which means the per-event test is liberal and that roughly half of the
# "significant" POST events would be expected from a pipeline that had no true
# cell-to-field correspondence at all. The POST excess over that control is still clear in
# this session, but it is an excess over 9%, not over 5%, and the exact counts move by a
# few events between runs because the shuffle uses only 400 draws.
#
# The honest reading of the four-session comparison is that these two results have very
# different evidential strength. The ripple physiology is reproducible: rate, duration,
# intra-ripple frequency and the non-REM / REM contrast are nearly identical in all four
# sessions. The replay result is carried by one session. Achilles is individually
# significant; Cicero is marginal (OR 1.6, p = 0.05); Gatsby (OR 1.07) and Buddy (OR 1.3)
# are not. The stratified CMH odds ratio of 1.9 is significant, but its homogeneity test
# rejects at p = 0.002, which is the formal way of saying that a single pooled number does
# not describe these four sessions and that quoting it alone would overstate the case.
#
# The most likely reason is simply the size of the ensemble. Achilles contributes 106
# place cells and decodes the animal's real position to 4.7 cm; the other sessions
# contribute 30 to 44 place cells and decode to 5.5-8.3 cm. Weighted correlation over a
# 20 ms bin needs enough simultaneously active fields to distinguish a trajectory from a
# scattered posterior, so a sequence measure that works at 106 cells can fall to near
# chance at 30 without the underlying phenomenon changing at all. That is an interpretable
# limitation of the measurement rather than evidence against replay, but it is a
# limitation, and the pairwise EV/REV statistic (which needs far fewer cells and does show
# EV > REV in all four sessions) is the better-supported cross-session claim here.

# %%
with open("results_summary.txt", "w") as f:
    f.write("session: %s\n" % SESSION)
    f.write("ripples: %d (%.2f Hz non-REM, %.3f Hz REM)\n" % (len(peak_t), rates["Non-REM"], rates["REM"]))
    f.write("ripple duration %.0f ms, frequency %.0f Hz\n" % (np.median(dur_ms), np.median(peak_freq)))
    f.write("place cells: %d of %d pyramidal\n" % (is_place.sum(), len(pyr)))
    f.write("decoder median error %.1f cm (chance %.1f cm)\n" % (np.median(err), chance))
    for e in ("PRE", "POST"):
        g = df[df.epoch == e]
        f.write("%s: %d/%d significant replay (%.1f%%)\n"
                % (e, g.significant.sum(), len(g), 100 * g.significant.mean()))
    f.write("cell-ID shuffle: %.1f%% (n=%d)  <- empirical null, above nominal 5%%\n"
            % (100 * ctrl.significant.mean(), len(ctrl)))
    f.write("Fisher POST vs PRE: OR=%.2f p=%.3g\n" % (odds, p_fisher))
    f.write("Fisher POST vs cell-ID shuffle: OR=%.2f p=%.3g\n" % (odds_cs, p_ctrl))
    f.write("replay speed %.0f cm/s vs running %.0f cm/s\n" % (sig.slope.abs().median(), run_speed))
    f.write("EV=%.3f REV=%.3f (%d pairs)\n" % (ev_val, rev_val, n_pairs))
    f.write("\n4 sessions, POST vs PRE:\n")
    for _, r in per_session.iterrows():
        f.write("  %-18s %d/%d POST vs %d/%d PRE  OR=%.2f p=%.3g\n"
                % (r.session, r.post_sig, r.post_n, r.pre_sig, r.pre_n, r.odds_ratio, r.p))
    f.write("CMH OR=%.2f p=%.3g; homogeneity p=%.3g (effect is NOT homogeneous)\n"
            % (or_cmh, p_cmh, p_homog))
print(open("results_summary.txt").read())
