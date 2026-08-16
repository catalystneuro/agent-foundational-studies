# %% [markdown]
# # Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples
#
# This notebook demonstrates **hippocampal replay** — the compressed re-expression
# of a waking spatial experience by hippocampal place cells during sharp-wave
# ripple (SWR) events of subsequent rest. We use a public Buzsáki-lab recording
# from the DANDI Archive, build a place-cell map of a linear track, detect ripples
# in the post-behaviour sleep LFP, and use a Bayesian decoder to read out the
# position "represented" by the population during each ripple. Replay appears as a
# posterior probability that sweeps smoothly across the track within the ~50-150 ms
# of a single ripple, far faster than the animal ever ran.
#
# **Dataset:** DANDI:000044, Grosmark, Long & Buzsáki, *"Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences"* (Science 2016).
# Session `Achilles_10252013`: bilateral CA1 silicon-probe recording of a rat that
# ran back and forth on a 1.6 m linear track (MAZE epoch) flanked by long
# rest/sleep epochs (PRE, POST). 137 sorted units (120 putative pyramidal), 128-ch
# LFP at 1250 Hz, and a linearized position signal.
#
# **Pipeline:** (1) place fields from track running, (2) SWR detection from the
# ripple-band LFP envelope, (3) memoryless Bayesian decoding of position in 20 ms
# bins within each ripple, (4) a weighted-correlation replay score benchmarked
# against a within-event column-shuffle null.

# %%
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from tqdm import tqdm

import replay_lib as rl

np.random.seed(0)
RNG = np.random.default_rng(0)

# %% [markdown]
# ## 1. Load the session (streaming from DANDI)
#
# The NWB file (~8.7 GB) is streamed from S3 with `remfile` disk caching; only the
# bytes we touch are fetched. Pynapple wraps spikes, LFP, position, epochs, and
# behavioural states as native objects.

# %%
nwb, h5, io = rl.load_session()
epochs = rl.get_epochs(nwb)
states = rl.get_states(h5)
print("Epochs (s):", {k: (round(v.start[0], 1), round(v.end[0], 1))
                      for k, v in epochs.items()})
print("Behavioural-state totals (s):",
      {k: round(float(np.sum(v.end - v.start)), 1) for k, v in states.items()})

pyr = rl.get_pyramidal_units(nwb)
print(f"Putative pyramidal (excitatory) CA1 units: {len(pyr)}")

# %% [markdown]
# ## 2. Place fields from track running
#
# The linearized position is valid only while the rat is on the track (NaN
# elsewhere), so we reconstruct contiguous on-track segments, keep epochs with
# running speed > 5 cm/s, and compute 1-D rate maps (50 bins over 1.6 m). Cells
# with a peak rate ≥ 1 Hz are retained as the decoding ensemble.

# %%
maze = epochs["MAZE"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_all = lin.index.values
x_all = np.asarray(lin.values).ravel()
m = (t_all >= maze.start[0]) & (t_all <= maze.end[0])
t_all, x_all = t_all[m], x_all[m]
good = ~np.isnan(x_all)
tg, xg = t_all[good], x_all[good]

dt_nom = np.median(np.diff(t_all))
brk = np.where(np.diff(tg) > 5 * dt_nom)[0]
valid = nap.IntervalSet(start=tg[np.r_[0, brk + 1]], end=tg[np.r_[brk, len(tg) - 1]])
pos = nap.Tsd(t=tg, d=xg, time_support=valid)
print(f"On-track time: {float(np.sum(valid.end - valid.start)):.0f} s "
      f"across {len(valid)} laps/segments")

speed = rl.compute_speed(pos)
run = valid.intersect(speed.threshold(0.05).time_support)
print(f"Running time (>5 cm/s): {float(np.sum(run.end - run.start)):.0f} s")

tc = nap.compute_1d_tuning_curves(pyr, pos, nb_bins=50, ep=run, minmax=(0, 1.6))
tc_smooth = tc.copy()
tc_smooth.iloc[:, :] = gaussian_filter1d(tc.values, 1.0, axis=0)
place_cells = tc_smooth.columns[tc_smooth.values.max(0) >= 1.0]
tc_smooth = tc_smooth[place_cells]
pyr_dec = pyr[list(place_cells)]
pos_bins = tc_smooth.index.values
print(f"Place cells used for decoding: {len(place_cells)}")

# %%
tcv = tc_smooth.values
order = np.argsort(np.argmax(tcv, axis=0))
norm = tcv / (tcv.max(0, keepdims=True) + 1e-9)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im = axs[0].imshow(norm[:, order].T, aspect="auto", origin="lower",
                   extent=[0, 1.6, 0, len(order)], cmap="viridis")
axs[0].set(xlabel="Linearized position (m)", ylabel="Place cell (sorted by peak)",
           title=f"Place-field map: {len(order)} CA1 pyramidal cells")
plt.colorbar(im, ax=axs[0], label="normalized rate")
axs[1].plot(pos.index.values - maze.start[0], pos.values, ".", ms=1)
axs[1].set(xlabel="Time in MAZE (s)", ylabel="Position (m)",
           title="Linearized trajectory (on-track running)")
plt.tight_layout()
plt.savefig("fig1_place_fields.png", dpi=130)
plt.close()

# %% [markdown]
# The rate maps tile the whole 1.6 m track: each cell fires at a distinct location
# and, sorted by peak position, they form a clean diagonal. This is the spatial
# code the decoder will invert.

# %% [markdown]
# ## 3. Sharp-wave ripple detection in POST sleep
#
# We pick the CA1 channel with the strongest 150-250 Hz power, band-pass filter it,
# and take the Hilbert envelope. Ripples are envelope excursions crossing 5 SD
# (extended out to 2 SD), 15-450 ms long. Detection is run over the entire POST
# epoch.

# %%
# Channel selection: ripple-band RMS across all channels in a POST sample window
lfp = h5["processing/ecephys/LFP/LFP/data"]
fs = rl.LFP_RATE
post = epochs["POST"]
i0 = int((post.start[0] + 500) * fs)
seg = lfp[i0:i0 + int(120 * fs), :].astype(np.float32)
rms = np.array([np.sqrt(np.mean(rl.bandpass(seg[:, c], 150, 250) ** 2))
                for c in range(seg.shape[1])])
ripple_ch = int(np.argmax(rms))
print(f"Selected ripple channel: {ripple_ch}")

# Read that single channel across all of POST and detect ripples
raw = lfp[int(post.start[0] * fs):int(post.end[0] * fs), ripple_ch].astype(np.float32)
t_lfp = post.start[0] + np.arange(len(raw)) / fs
filt = rl.bandpass(raw, 150, 250)
env = gaussian_filter1d(np.abs(hilbert(filt)), int(0.008 * fs))
env_tsd = nap.Tsd(t=t_lfp, d=env, time_support=post)

ripples, rip_peaks = rl.detect_ripples(env_tsd, post, low_thr=2.0, high_thr=5.0,
                                        min_dur=0.015, max_dur=0.45, merge_gap=0.03)
rip_dur = ripples.end - ripples.start
print(f"Detected {len(ripples)} ripples in POST "
      f"({len(ripples) / ((post.end[0]-post.start[0])/60):.1f}/min); "
      f"median duration {np.median(rip_dur)*1000:.0f} ms")

# %%
# Example ripple + duration/amplitude distributions
pk = rip_peaks.index.values[np.argmax(rip_peaks.values)]
w = 0.15
mm = (t_lfp > pk - w) & (t_lfp < pk + w)
fig, axs = plt.subplots(2, 2, figsize=(12, 6),
                        gridspec_kw={"width_ratios": [1.4, 1]})
axs[0, 0].plot((t_lfp[mm] - pk) * 1000, raw[mm], "k", lw=0.6)
axs[0, 0].set(ylabel="raw LFP (a.u.)", title="Example sharp-wave ripple")
axs[1, 0].plot((t_lfp[mm] - pk) * 1000, filt[mm], "C3", lw=0.6)
axs[1, 0].set(ylabel="150-250 Hz", xlabel="Time from ripple peak (ms)")
axs[0, 1].hist(rip_dur * 1000, bins=40, color="C0")
axs[0, 1].set(xlabel="Ripple duration (ms)", ylabel="count",
              title=f"N = {len(ripples)} ripples")
axs[1, 1].hist(rip_peaks.values, bins=40, color="C3")
axs[1, 1].set(xlabel="Peak envelope (SD)", ylabel="count")
plt.tight_layout()
plt.savefig("fig2_ripples.png", dpi=130)
plt.close()

# %% [markdown]
# ## 4. Bayesian decoding of position during ripples
#
# For every ripple with ≥ 5 active place cells and ≥ 80 ms duration we decode
# position in 20 ms bins with Pynapple's memoryless Bayesian decoder (a flat
# spatial prior, so the read-out is driven purely by which place cells fire). If a
# ripple replays a trajectory, the posterior should march across the track.

# %%
spike_counts = pyr_dec.count(ep=ripples)
n_active = (spike_counts.values > 0).sum(1)
candidates = np.where((n_active >= 5) & (rip_dur >= 0.08))[0]
print(f"Candidate replay events (>=5 cells, >=80 ms): {len(candidates)}")

BIN = 0.02
results = []      # per-event scores
posteriors = {}   # keep posteriors for plotting
for ev in tqdm(candidates, desc="decoding ripples"):
    s, e = ripples.start[ev], ripples.end[ev]
    ep1 = nap.IntervalSet(s - 0.005, e + 0.005)
    decoded, proba = nap.decode_1d(tc_smooth, pyr_dec, ep1, BIN)
    P = proba.values.T                       # (position, time)
    tt = proba.index.values
    if P.shape[1] < 4:
        continue
    wc, pval, sd = rl.score_event(P, pos_bins, tt, n_shuffle=500, rng=RNG)
    results.append(dict(event=ev, start=s, end=e, n_active=int(n_active[ev]),
                        n_bins=P.shape[1], wc=wc, p=pval))
    posteriors[ev] = (P, tt, decoded)

res = pd.DataFrame(results)
res["abs_wc"] = res["wc"].abs()
res["significant"] = res["p"] < 0.05
res.to_csv("replay_scores.csv", index=False)

n_sig = int(res["significant"].sum())
print(f"\nScored {len(res)} events. Significant replays (p<0.05): "
      f"{n_sig} ({100*n_sig/len(res):.0f}%)")
print(f"Forward (wc>0): {int(((res.wc>0)&res.significant).sum())}, "
      f"Reverse (wc<0): {int(((res.wc<0)&res.significant).sum())}")
print(f"Mean |wc| real = {res.abs_wc.mean():.3f}")

# %% [markdown]
# ## 5. Population statistics: real replay vs. shuffle
#
# We compare the observed weighted correlations against a null in which each
# ripple's decoded posterior is column-cycle shuffled (position circularly shifted
# independently per time bin), which destroys spatial continuity while preserving
# per-bin certainty. Real ripples carry systematically more line-like structure.

# %%
# Build a pooled shuffle distribution of |wc| for the plotted null
null_abs = []
for ev in res["event"].values:
    P, tt, _ = posteriors[ev]
    for _ in range(20):
        shifts = RNG.integers(0, P.shape[0], size=P.shape[1])
        Ps = np.stack([np.roll(P[:, j], shifts[j]) for j in range(P.shape[1])], axis=1)
        null_abs.append(abs(rl.weighted_correlation(Ps, pos_bins, tt)))
null_abs = np.array(null_abs)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
axs[0].hist(null_abs, bins=40, density=True, alpha=0.6, color="gray",
            label="shuffle")
axs[0].hist(res["abs_wc"], bins=40, density=True, alpha=0.6, color="C3",
            label="observed ripples")
axs[0].axvline(res["abs_wc"].mean(), color="C3", ls="--")
axs[0].axvline(null_abs.mean(), color="gray", ls="--")
axs[0].set(xlabel="|weighted correlation|", ylabel="density",
           title="Replay score: real vs shuffle")
axs[0].legend()

axs[1].hist(res.loc[res.significant, "wc"], bins=30, color="C0")
axs[1].axvline(0, color="k", lw=0.8)
axs[1].set(xlabel="weighted correlation (signed)", ylabel="count",
           title=f"Significant replays (n={n_sig})\nleft=reverse, right=forward")

frac_sig_by_thr = [np.mean(res["p"] < a) for a in [0.05, 0.01]]
axs[2].bar(["observed\np<0.05", "chance\n(0.05)", "observed\np<0.01",
            "chance\n(0.01)"],
           [frac_sig_by_thr[0], 0.05, frac_sig_by_thr[1], 0.01],
           color=["C3", "gray", "C3", "gray"])
axs[2].set(ylabel="fraction of events", title="Significant-event fraction")
plt.tight_layout()
plt.savefig("fig4_population_stats.png", dpi=130)
plt.close()

# %% [markdown]
# ## 6. Example replay trajectories
#
# The clearest events: the six most significant replays with the strongest
# weighted correlation, split so both forward (wc>0) and reverse (wc<0) sweeps are
# shown. The dashed line is the weighted-correlation fit; the cyan trace is the
# per-bin MAP estimate over the "hot" posterior.

# %%
sig = res[res.significant].copy()
fwd = sig[sig.wc > 0].sort_values("abs_wc", ascending=False)
rev = sig[sig.wc < 0].sort_values("abs_wc", ascending=False)
pick = list(fwd["event"].values[:3]) + list(rev["event"].values[:3])

fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, ev in zip(axs.ravel(), pick):
    P, tt, decoded = posteriors[ev]
    row = res[res.event == ev].iloc[0]
    rel = (tt - tt[0]) * 1000
    ax.imshow(P, aspect="auto", origin="lower",
              extent=[rel[0], rel[-1], 0, 1.6], cmap="hot")
    ax.plot((decoded.index.values - tt[0]) * 1000, decoded.values,
            "c.-", ms=5, lw=1)
    # weighted-correlation line fit
    w = P / P.sum()
    mx = np.sum(w * pos_bins[:, None]); mt = np.sum(w * tt[None, :])
    b = (np.sum(w * (pos_bins[:, None] - mx) * (tt[None, :] - mt)) /
         np.sum(w * (tt[None, :] - mt) ** 2))
    ax.plot(rel, (mx + b * (tt - mt)), "w--", lw=1.5)
    kind = "forward" if row.wc > 0 else "reverse"
    ax.set(title=f"{kind}: wc={row.wc:.2f}, p={row.p:.3f}, {row.n_active} cells",
           xlabel="time in ripple (ms)", ylabel="decoded position (m)",
           ylim=(0, 1.6))
plt.tight_layout()
plt.savefig("fig3_example_replays.png", dpi=130)
plt.close()

# %% [markdown]
# ## Summary
#
# Place cells recorded while the rat ran a 1.6 m track tile the environment with
# well-separated firing fields. During sharp-wave ripples of subsequent rest, a
# Bayesian decoder trained on those fields reads out positions that sweep
# coherently across the track within tens of milliseconds. The observed
# weighted-correlation replay scores exceed a column-shuffle null far more often
# than the 5% expected by chance, and both forward and reverse trajectories occur.
# This is hippocampal replay: the offline, temporally compressed reactivation of
# waking spatial experience during SWRs, decoded directly from public DANDI data.

# %%
print("Done. Figures written:")
for f in ["fig1_place_fields.png", "fig2_ripples.png",
          "fig3_example_replays.png", "fig4_population_stats.png"]:
    print(" ", f)
io.close()
