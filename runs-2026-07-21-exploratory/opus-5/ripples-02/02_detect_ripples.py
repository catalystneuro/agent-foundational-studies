"""Detect sharp-wave ripples across the whole session and validate them."""

import os

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

from dandi_io import open_session
from swr_lib import (FS, detect_ripples, linearize_position, load_lfp_channel,
                     ripple_envelope, ripple_peak_frequency, running_speed)

SESSION = "Achilles_10252013"
CH = int(np.load("ripple_channel.npy")[0])

nwbfile, nwb, h5 = open_session(SESSION)
ep_df = nwbfile.epochs.to_dataframe()
maze = nap.IntervalSet(start=ep_df.query("label=='MazeEpoch'").start_time.values,
                       end=ep_df.query("label=='MazeEpoch'").stop_time.values)
pre = nap.IntervalSet(start=ep_df.query("label=='PREEpoch'").start_time.values,
                      end=ep_df.query("label=='PREEpoch'").stop_time.values)
post = nap.IntervalSet(start=ep_df.query("label=='POSTEpoch'").start_time.values,
                       end=ep_df.query("label=='POSTEpoch'").stop_time.values)

cache = f"lfp_ch{CH}_{SESSION}.npy"
if os.path.exists(cache):
    d = np.load(cache)
    lfp = nap.Tsd(t=np.arange(d.size) / FS, d=d)
else:
    lfp = load_lfp_channel(nwbfile, CH)
    np.save(cache, lfp.values)
print("LFP loaded:", lfp.shape, lfp.time_support)

# deepest channel on the same shank -> stratum radiatum, for the sharp wave
elec = nwbfile.electrodes.to_dataframe()
shank = elec.group_name.iloc[CH]
same = np.where(elec.group_name.values == shank)[0]
CH_RAD = int(same[np.argmax(elec.shank_electrode_number.values[same])])
cache_r = f"lfp_ch{CH_RAD}_{SESSION}.npy"
if os.path.exists(cache_r):
    lfp_rad = nap.Tsd(t=lfp.times(), d=np.load(cache_r))
else:
    lfp_rad = load_lfp_channel(nwbfile, CH_RAD)
    np.save(cache_r, lfp_rad.values)
print(f"pyramidal-layer ch {CH}, radiatum ch {CH_RAD} (shank {shank})")

# ------------------------------------------------------------------ detection
filt, env = ripple_envelope(lfp)

pos, TRACK = linearize_position(nwb)
speed = running_speed(pos)
run_ep = speed.threshold(0.05).time_support.merge_close_intervals(0.5).drop_short_intervals(0.3)
session = nap.IntervalSet(start=0.0, end=float(lfp.times()[-1]))
immobile = session.set_diff(run_ep)
print(f"running: {run_ep.tot_length():.0f} s over {len(run_ep)} bouts; "
      f"immobility reference epoch {immobile.tot_length():.0f} s")

# events are detected session-wide; the z-score reference is immobility only
ripples, peaks = detect_ripples(env, session, ref_ep=immobile)
detect_ep = session
dur = ripples.end - ripples.start
pfreq = ripple_peak_frequency(filt, peaks)
print(f"{len(ripples)} ripples, {len(ripples)/detect_ep.tot_length()*60:.1f}/min, "
      f"median duration {np.median(dur)*1000:.0f} ms, "
      f"median peak freq {np.nanmedian(pfreq):.0f} Hz")

np.savez("ripples_Achilles_10252013.npz", start=ripples.start, end=ripples.end,
         peak_t=peaks.times(), peak_z=peaks.values, peak_freq=pfreq, channel=CH)

# ------------------------------------------- ripple-triggered averages and PSD
units = nwb["units"]
pyr = units[units.cell_type == "excitatory"]

half = int(0.25 * FS)
pk_idx = np.searchsorted(lfp.times(), peaks.times())
pk_idx = pk_idx[(pk_idx > half) & (pk_idx < lfp.shape[0] - half)]
seg = np.stack([lfp.values[i - half : i + half] for i in pk_idx])
seg_rad = np.stack([lfp_rad.values[i - half : i + half] for i in pk_idx])
seg_f = np.stack([filt.values[i - half : i + half] for i in pk_idx])
lags = (np.arange(-half, half)) / FS

# power spectrum inside vs outside ripples (Welch on concatenated snippets)
from scipy.signal import welch
inside = np.concatenate([lfp.values[i - 31 : i + 31] for i in pk_idx[:4000]])
rng = np.random.default_rng(0)
ctrl_idx = rng.choice(np.arange(half, lfp.shape[0] - half), size=4000, replace=False)
outside = np.concatenate([lfp.values[i - 31 : i + 31] for i in ctrl_idx])
f_in, p_in = welch(inside, fs=FS, nperseg=62)
f_out, p_out = welch(outside, fs=FS, nperseg=62)

# MUA around ripple peaks
mua_t = np.sort(np.concatenate([pyr[i].times() for i in pyr.index]))
peth_bins = np.arange(-0.25, 0.2501, 0.005)
counts = np.zeros(peth_bins.size - 1)
for tp in peaks.times():
    i0, i1 = np.searchsorted(mua_t, [tp - 0.25, tp + 0.25])
    counts += np.histogram(mua_t[i0:i1] - tp, bins=peth_bins)[0]
mua_counts = counts / (0.005 * len(peaks))

# ----------------------------------------------------------------- fig 02
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.3)

# three representative events (around the median amplitude), raw + filtered + envelope
order = np.argsort(peaks.values)
best = order[[int(0.5 * len(order)), int(0.75 * len(order)), int(0.95 * len(order))]]
for j, bi in enumerate(best):
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
    r = ripples[bi]
    ax.axvspan(r.start[0] - t_pk, r.end[0] - t_pk, color="#ffd166", alpha=0.4, zorder=0)
    ax.set_xlabel("time from peak (s)")
    if j == 0:
        ax.set_ylabel("µV (traces offset)")
        ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"event {bi}, {peaks.values[bi]:.1f} SD", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.plot(lags, seg.mean(0) * 1e6, color="k", lw=1.2, label=f"pyr. layer (ch {CH})")
ax.plot(lags, seg_rad.mean(0) * 1e6, color="#0077b6", lw=1.2,
        label=f"radiatum (ch {CH_RAD})")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("LFP (µV)")
ax.set_title(f"ripple-triggered average LFP (n={len(pk_idx)})", fontsize=10)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 1])
ax.plot(lags, seg_f.mean(0) * 1e6, color="#e63946", lw=1.0)
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

ax = fig.add_subplot(gs[2, 0])
ax.hist(dur * 1000, bins=40, color="#023047")
ax.set_xlabel("duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"durations (median {np.median(dur)*1000:.0f} ms)", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
ax.hist(pfreq[np.isfinite(pfreq)], bins=40, color="#023047")
ax.set_xlabel("peak frequency (Hz)")
ax.set_title(f"ripple frequency (median {np.nanmedian(pfreq):.0f} Hz)", fontsize=10)

ax = fig.add_subplot(gs[2, 2])
ax.hist(peaks.values, bins=np.arange(5, 30, 0.5), color="#023047")
ax.set_xlabel("peak envelope (SD)")
ax.set_title("event amplitude", fontsize=10)

ax = fig.add_subplot(gs[3, 0])
ax.plot(np.arange(-0.25, 0.25, 0.005) + 0.0025, mua_counts, color="#fb8500", lw=1.2)
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("pyramidal MUA (Hz)")
ax.set_title("population firing around ripples", fontsize=10)

# ripple rate per behavioural state
st = nwbfile.processing["behavior"]["states"].to_dataframe()
ax = fig.add_subplot(gs[3, 1])
rates, labels = [], []
state_eps = {}
for lab in ["Non-REM", "REM", "Awake"]:
    sub = st[st.label == lab]
    state_eps[lab] = nap.IntervalSet(start=sub.start_time.values,
                                     end=sub.stop_time.values).intersect(immobile)
state_eps["Awake\nrunning"] = run_ep
for lab, ep in state_eps.items():
    n = len(nap.Ts(peaks.times()).restrict(ep))
    rates.append(n / ep.tot_length() * 60)
    labels.append(f"{lab}\n({ep.tot_length()/60:.0f} min)")
ax.bar(labels, rates, color=["#023047", "#e63946", "#adb5bd", "#fb8500"])
ax.tick_params(axis="x", labelsize=8)
ax.set_ylabel("ripples / min")
ax.set_title("state dependence", fontsize=10)

# speed dependence: occupancy from the tracking samples themselves, so that
# short excursions above threshold are not lost to interval bookkeeping
ax = fig.add_subplot(gs[3, 2])
dt_pos = float(np.median(np.diff(pos.times())))
pk_maze = nap.Ts(peaks.times()).restrict(speed.time_support)
v_at_ripple = speed.values[np.clip(np.searchsorted(speed.times(), pk_maze.times()),
                                   0, speed.shape[0] - 1)]
edges = np.array([0, 2, 5, 10, 20, 30, 45, 100]) / 100
occ = np.histogram(speed.values, bins=edges)[0] * dt_pos
cnt = np.histogram(v_at_ripple, bins=edges)[0]
rate_v = np.where(occ > 10, cnt / np.maximum(occ, 1e-9) * 60, np.nan)
ctr = (edges[:-1] + edges[1:]) / 2
ax.plot(ctr * 100, rate_v, "o-", color="#fb8500")
for x, y, n in zip(ctr * 100, rate_v, cnt):
    if np.isfinite(y):
        ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, ha="center")
ax.set_xlabel("running speed (cm/s)")
ax.set_ylabel("ripples / min")
ax.set_title("speed dependence on the maze\n(n events above each point)", fontsize=10)

fig.suptitle(f"{SESSION}: sharp-wave ripple detection and validation", y=0.93)
fig.savefig("fig02_ripple_detection.png", dpi=150, bbox_inches="tight")
print("wrote fig02_ripple_detection.png")
print("ripple rate by state:", dict(zip(state_eps, np.round(rates, 2))))
