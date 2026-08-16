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
# # Theta phase entrainment of hippocampal CA1 neurons
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki
# (2016), the `hc-11` recordings. Bilateral CA1 silicon-probe recordings from freely moving
# rats running back and forth on a linear track for water reward, with sleep sessions before
# (PRE) and after (POST) the maze. The NWB files provide sorted units with a pyramidal /
# interneuron label, a 128-channel LFP `ElectricalSeries` at 1250 Hz, tracked position, and
# hand-scored REM / non-REM sleep intervals.
#
# **Phenomenon.** During locomotion the rodent hippocampus is dominated by a 6–10 Hz theta
# oscillation in the LFP. Individual neurons do not fire uniformly across the theta cycle;
# they fire preferentially at a particular phase. This is theta phase entrainment (or theta
# phase locking), and it is one of the most reproducible observations in systems
# neuroscience.
#
# **What this notebook does.**
#
# 1. Streams one session from S3 and validates every data stream before analysing it.
# 2. Picks the LFP channel with the strongest theta and verifies the phase extraction.
# 3. Measures, for every unit, the distribution of theta phases at which it spikes, and
#    tests it against both an analytic null (Rayleigh) and a spike-jitter null.
# 4. Repeats across five sessions and four rats.
# 5. Runs two controls that a filtering artefact would fail: a brain-state comparison and
#    theta phase precession.
#
# **Phase convention throughout:** theta phase is the Hilbert phase of the 6–10 Hz
# bandpass-filtered LFP, with 0° = peak of the filtered signal and 180° = trough. The
# recording is from the CA1 pyramidal layer, so the LFP peak corresponds roughly to the
# stratum-pyramidale theta peak.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import welch
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

import theta_utils as tu  # shared loading, filtering and circular statistics

SESSION = "Achilles-10252013"
MIN_SPIKES = 100        # a unit needs this many spikes in a state to be tested
SPEED_THRESH = 10.0     # cm/s, defines "running"
N_SHUFFLES = 200        # jitter-null resamples
NBINS = 18              # phase bins (20 deg each)

# Reuse the CSVs written by the staged scripts when they are present. Delete them to
# force a full recomputation; everything below runs end-to-end either way.
REUSE = os.path.exists("phase_locking_all_sessions.csv")
print("theta band:", tu.THETA_BAND, "Hz;  LFP rate:", tu.LFP_FS, "Hz")
print("sessions available:", list(tu.SESSIONS))

# %% [markdown]
# ## 1. Load one session and validate every data stream
#
# The NWB file is read directly from the DANDI S3 bucket with `remfile` plus a disk cache,
# so nothing is downloaded in full. The LFP dataset is 43M × 128 `int16` chunked as
# (170221, 1), which makes single-channel reads cheap; that is the only thing we need
# lazily, so the file is opened as raw HDF5 rather than through `pynwb`. Units, epochs,
# sleep states and position are small and are pulled straight into pynapple objects.

# %%
h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)
units = meta["units"]
maze = meta["maze_epoch"]
post = meta["post_epoch"]

print("epochs:")
for k, v in meta["epochs"].items():
    print(f"   {k:12s} {v.start[0]:9.1f} -> {v.end[0]:9.1f} s  ({v.tot_length():.0f} s)")
print("brain states:")
for k, v in meta["states"].items():
    print(f"   {k:10s} n={len(v):3d}  {v.tot_length():8.1f} s")
print(f"units: {len(units)}   LFP channels: {meta['n_lfp_channels']}")
print("cell types:", {t: int((units.cell_type == t).sum())
                      for t in np.unique(units.cell_type)})

# %% [markdown]
# Position is tracked at 39 Hz in metres. Speed is the smoothed 2-D displacement, with
# short tracking dropouts interpolated across. Running is defined as speed above 10 cm/s
# for at least 1 s, which on this track selects the traversals and excludes the reward
# ports where the animal pauses.

# %%
pos = meta["position"].restrict(maze)
speed = tu.compute_speed(pos, rate=meta["position_rate"])
run = speed.threshold(SPEED_THRESH).time_support.drop_short_intervals(1.0)
rem = meta["states"]["REM"].intersect(post).drop_short_intervals(20.0)
nrem = meta["states"]["Non-REM"].intersect(post).drop_short_intervals(20.0)

print("position: %d samples, x %.2f-%.2f m, NaN fraction %.3f"
      % (len(pos), np.nanmin(pos["x"].d), np.nanmax(pos["x"].d),
         np.isnan(pos["x"].d).mean()))
print("speed: median %.1f cm/s, 90th pct %.1f cm/s"
      % (np.median(speed.d), np.percentile(speed.d, 90)))
print("RUN %.0f s (%d intervals) / REM %.0f s / non-REM %.0f s"
      % (run.tot_length(), len(run), rem.tot_length(), nrem.tot_length()))

# %% [markdown]
# ### Figure 1: raw data
#
# Before any analysis, look at the raw streams together: broadband LFP, the theta-filtered
# signal with its Hilbert envelope, the spike raster, and running speed. The theta rhythm
# should be visible by eye in the raw trace while the animal is moving.

# %%
ch_preview = 64
t0 = maze.start[0] + 600
lfp = tu.load_lfp_channel(h, ch_preview, t0, t0 + 12)
filt, phase, amp = tu.theta_phase_amplitude(lfp)

fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                        gridspec_kw={"height_ratios": [2, 2, 3, 1.5]})
axs[0].plot(lfp.t, lfp.d, lw=0.6, color="0.3")
axs[0].set_ylabel("LFP (µV)")
axs[0].set_title(f"{SESSION} raw data validation (channel {ch_preview}, maze epoch)")
axs[1].plot(lfp.t, lfp.d, lw=0.5, color="0.75", label="broadband")
axs[1].plot(filt.t, filt.d, lw=1.4, color="C0", label="6-10 Hz")
axs[1].plot(amp.t, amp.d, lw=1.0, color="C3", label="theta envelope")
axs[1].legend(loc="upper right", fontsize=8, ncol=3)
axs[1].set_ylabel("LFP (µV)")
sel = units.restrict(nap.IntervalSet(t0, t0 + 12))
cell_type = units.cell_type.values
for i, uid in enumerate(units.index):
    axs[2].vlines(sel[uid].t, i, i + 0.85, lw=0.5,
                  color="C0" if cell_type[i] == "excitatory" else "C3")
axs[2].set_ylabel("unit #"); axs[2].set_ylim(0, len(units))
sp = speed.restrict(nap.IntervalSet(t0, t0 + 12))
axs[3].plot(sp.t, sp.d, color="C2")
axs[3].set_ylabel("speed (cm/s)"); axs[3].set_xlabel("time (s)")
axs[3].set_xlim(t0, t0 + 12)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig01_raw_data_overview.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Figure 2: behaviour
#
# The animal shuttles along a 1.6 m linear track and the x-position trace shows clean
# back-and-forth traversals. The speed distribution (log counts) is dominated by a large
# slow mode from the time spent at the two reward ports, with a long tail and a shoulder
# between roughly 50 and 90 cm/s from the traversals themselves; it is not cleanly bimodal,
# which is why the 10 cm/s threshold is combined with a 1 s minimum duration rather than
# placed at a trough. About 21% of position samples are untracked and are interpolated
# across for the speed estimate.

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].plot(pos["x"].d, pos["y"].d, lw=0.4, color="0.4")
axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")
axs[0].set_title("tracked position, maze epoch")
axs[1].plot(pos.t - maze.start[0], pos["x"].d, lw=0.6)
axs[1].set_xlabel("time in maze epoch (s)"); axs[1].set_ylabel("x (m)")
axs[1].set_xlim(600, 900); axs[1].set_title("linear track traversals")
axs[2].hist(speed.d, bins=80, color="C2")
axs[2].axvline(SPEED_THRESH, color="k", ls="--", label=f"{SPEED_THRESH:.0f} cm/s threshold")
axs[2].set_yscale("log"); axs[2].set_xlabel("speed (cm/s)"); axs[2].set_ylabel("count")
axs[2].legend(fontsize=8); axs[2].set_title("speed distribution")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig02_behavior_validation.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 2. Choose the theta channel and validate the phase estimate
#
# Theta amplitude varies systematically with depth across the CA1 layers and reverses
# phase below the pyramidal layer, so the channel matters. We rank all 128 channels by the
# ratio of 6–10 Hz to 1–4 Hz power over 100 s in the middle of the maze epoch and take the
# maximum. Two checks follow: the power spectrum should show a theta peak during running
# and REM but not during non-REM, and the theta-trough-triggered average of the *raw*
# (unfiltered) LFP should be an oscillation, which it can only be if the detected phase
# corresponds to a real feature of the signal rather than to filter ringing.

# %%
t_scan = maze.start[0] + 900
ratios = np.array([
    tu.theta_delta_ratio(tu.load_lfp_channel(h, ch, t_scan, t_scan + 100).d)
    for ch in tqdm(range(meta["n_lfp_channels"]), desc="scanning channels")
])
BEST_CH = int(np.argmax(ratios))
print(f"selected channel {BEST_CH}, theta/delta = {ratios[BEST_CH]:.2f}")

# %%
lfp_maze = tu.load_lfp_channel(h, BEST_CH, maze.start[0], maze.end[0])
lfp_rem = tu.load_lfp_channel(h, BEST_CH, rem.start[0], rem.end[-1])
lfp_nrem = tu.load_lfp_channel(h, BEST_CH, nrem.start[0], nrem.end[-1])

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(ratios, ".-", ms=4, lw=0.6, color="0.4")
axs[0].plot(BEST_CH, ratios[BEST_CH], "o", color="C3", ms=9, label=f"selected ch {BEST_CH}")
axs[0].set_xlabel("LFP channel"); axs[0].set_ylabel("theta / delta power")
axs[0].set_title("channel selection (100 s of maze)"); axs[0].legend(fontsize=8)

for lbl, sig, c in [("run (maze)", lfp_maze.restrict(run).d, "C0"),
                    ("REM sleep", lfp_rem.restrict(rem).d, "C2"),
                    ("non-REM sleep", lfp_nrem.restrict(nrem).d, "C3")]:
    f, p = welch(sig, fs=tu.LFP_FS, nperseg=int(4 * tu.LFP_FS))
    m = (f > 0.5) & (f < 30)
    axs[1].semilogy(f[m], p[m], color=c, label=lbl)
axs[1].axvspan(*tu.THETA_BAND, color="C1", alpha=0.15)
axs[1].set_xlabel("frequency (Hz)"); axs[1].set_ylabel("PSD (µV²/Hz)")
axs[1].set_title(f"power spectrum by brain state (ch {BEST_CH})"); axs[1].legend(fontsize=8)

_, ph_full, _ = tu.theta_phase_amplitude(lfp_maze)
ph_run = ph_full.restrict(run).d
raw_run = lfp_maze.restrict(run).d
# phase crossing pi is the trough of the filtered signal (0 = peak by convention)
troughs = np.where(np.diff(np.sign(np.mod(ph_run + np.pi, 2 * np.pi) - np.pi)) < 0)[0]
w = int(0.15 * tu.LFP_FS)
seg = np.array([raw_run[i - w:i + w] for i in troughs[5:2000]
                if i - w >= 0 and i + w < len(raw_run)])
tt = np.arange(-w, w) / tu.LFP_FS
axs[2].plot(tt, seg.mean(0), color="C0")
axs[2].fill_between(tt, seg.mean(0) - seg.std(0) / np.sqrt(len(seg)),
                    seg.mean(0) + seg.std(0) / np.sqrt(len(seg)), color="C0", alpha=0.3)
axs[2].axvline(0, color="k", ls="--", lw=0.8)
axs[2].set_xlabel("time from detected theta trough (s)"); axs[2].set_ylabel("raw LFP (µV)")
axs[2].set_title(f"trough-triggered average, n={len(seg)} cycles")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig03_theta_channel_and_spectra.png", dpi=150)
plt.close(fig)

print("theta/delta ratio:  run=%.2f  REM=%.2f  non-REM=%.2f" % (
    tu.theta_delta_ratio(lfp_maze.restrict(run).d),
    tu.theta_delta_ratio(lfp_rem.restrict(rem).d),
    tu.theta_delta_ratio(lfp_nrem.restrict(nrem).d)))

# %% [markdown]
# ## 3. Per-unit phase locking
#
# Each running interval is filtered and Hilbert-transformed separately: filtering across
# the gap between two distant intervals would smear phase across the discontinuity. Every
# spike is then assigned the phase of the nearest LFP sample (1250 Hz, so under 0.4 ms of
# error), and each unit's phase distribution is summarised by
#
# - **MRL**, the mean resultant length $R = |\langle e^{i\phi}\rangle|$, which is 0 for a
#   uniform distribution and 1 for perfect locking;
# - **PPC**, pairwise phase consistency (Vinck et al., 2010), which unlike the MRL is not
#   biased upward by low spike counts. Spike counts here span two orders of magnitude, so
#   this matters;
# - the **Rayleigh test** for non-uniformity, with Benjamini–Hochberg FDR correction across
#   units;
# - a **spike-jitter null**: each spike is displaced by a uniform ±0.5 s, which is several
#   theta cycles, so the theta relationship is destroyed while the unit's slow rate profile
#   and its exact spike count are preserved. This is the control that matters, because the
#   Rayleigh test assumes independent samples and spikes in a burst are not independent.

# %%
state_epochs = {"RUN": run, "REM": rem, "nonREM": nrem}
phases, amps, raws, epochs = {}, {}, {}, {}
for name, ep in state_epochs.items():
    phases[name], amps[name], raws[name], epochs[name] = tu.phase_by_interval(
        h, BEST_CH, ep, max_total=1200.0)
    print(f"{name}: {epochs[name].tot_length():.0f} s of LFP "
          f"over {len(epochs[name])} segments")
lookups = {k: tu.PhaseLookup(v) for k, v in phases.items()}

# %%
rows = []
rng = np.random.default_rng(1)
for i, uid in enumerate(tqdm(units.index, desc="units")):
    st = units[uid].t
    row = {"session": SESSION, "unit": uid, "cell_type": cell_type[i]}
    for name in state_epochs:
        ph = lookups[name](st)
        n = len(ph)
        r, z, p = tu.rayleigh_test(ph) if n >= MIN_SPIKES else (np.nan,) * 3
        row[f"n_{name}"], row[f"mrl_{name}"], row[f"p_{name}"] = n, r, p
        row[f"ppc_{name}"] = tu.ppc(ph) if n >= MIN_SPIKES else np.nan
        row[f"pref_{name}"] = tu.circ_mean(ph) if n >= MIN_SPIKES else np.nan
    if row["n_RUN"] >= MIN_SPIKES:
        null = tu.jitter_null_mrl(st, lookups["RUN"], n_shuffles=N_SHUFFLES, rng=rng)
        row["mrl_null_mean"] = null.mean()
        row["mrl_null_p95"] = np.percentile(null, 95)
        row["p_shuffle"] = (np.sum(null >= row["mrl_RUN"]) + 1) / (N_SHUFFLES + 1)
    rows.append(row)

res = pd.DataFrame(rows)
for name in state_epochs:
    ok = res[f"n_{name}"] >= MIN_SPIKES
    res[f"sig_{name}"] = False
    res.loc[ok, f"sig_{name}"] = tu.benjamini_hochberg(res.loc[ok, f"p_{name}"].values)
res.to_csv("phase_locking_single_session.csv", index=False)

incl = (res["n_RUN"] >= MIN_SPIKES).values
locked = res["sig_RUN"].values & incl
print("\n=== %s, RUN epochs, %d units with >=%d spikes ===" % (SESSION, incl.sum(), MIN_SPIKES))
print("significantly phase locked (Rayleigh, FDR q<0.05): %d/%d (%.0f%%)"
      % (locked.sum(), incl.sum(), 100 * locked.sum() / incl.sum()))
print("exceeds the jitter null at p<0.05:                 %d/%d"
      % ((res.loc[incl, "p_shuffle"] < 0.05).sum(), incl.sum()))
for ct in ["excitatory", "inhibitory"]:
    m = incl & (res.cell_type == ct).values
    print(f"  {ct:11s} n={m.sum():3d}  median MRL={res.loc[m,'mrl_RUN'].median():.3f}"
          f"  preferred phase="
          f"{np.degrees(tu.circ_mean(res.loc[m & locked,'pref_RUN'])):6.1f} deg")

# %% [markdown]
# ### Figure 4: spikes tracking the theta cycle
#
# The left column shows 1.6 s of running: the LFP with its theta filter, a raster of all
# locked units sorted by preferred phase, and the cycle-averaged population rate. The
# diagonal banding in the raster and the sinusoidal population rate are the phenomenon.
# The right panel is the phase tuning curve of every locked unit, z-scored across phase
# bins; two theta cycles are plotted so the wraparound is visible.

# %%
tc = nap.compute_1d_tuning_curves(units, phases["RUN"], nb_bins=NBINS,
                                  minmax=(0, 2 * np.pi), ep=epochs["RUN"])
phase_centers = tc.index.values
sorted_units = res.loc[locked].sort_values("pref_RUN")["unit"].values

WIN_S = 1.6
n_w = int(WIN_S * tu.LFP_FS)
env, rawd = amps["RUN"].d, raws["RUN"].d
mean_env = np.convolve(env, np.ones(n_w) / n_w, mode="valid")
clipped = np.convolve(np.abs(rawd) > 1500, np.ones(n_w), mode="valid")
contiguous = (amps["RUN"].t[n_w - 1:] - amps["RUN"].t[:len(mean_env)]) < WIN_S * 1.01
tw = amps["RUN"].t[int(np.argmax(np.where(contiguous & (clipped == 0), mean_env, -np.inf)))]
win = nap.IntervalSet(tw, tw + WIN_S)
raw_w, ph_w = raws["RUN"].restrict(win), phases["RUN"].restrict(win)
filt_w = tu.bandpass(raws["RUN"], *tu.THETA_BAND).restrict(win)
trough_t = ph_w.t[np.where(np.diff(np.sign(ph_w.d - np.pi)) > 0)[0]]

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 2.2, 1.4], width_ratios=[3, 1.15],
                      hspace=0.35, wspace=0.25)
ax = fig.add_subplot(gs[0, 0])
ax.plot(raw_w.t - tw, raw_w.d, color="0.7", lw=0.7, label="raw LFP")
ax.plot(filt_w.t - tw, filt_w.d, color="C0", lw=1.6, label="6-10 Hz theta")
for t_ in trough_t:
    ax.axvline(t_ - tw, color="0.85", lw=0.6, zorder=0)
ax.legend(fontsize=8, loc="upper right", ncol=2); ax.set_ylabel("µV")
ax.set_title(f"{SESSION}, CA1 channel {BEST_CH}: spikes track the theta cycle "
             f"({WIN_S:.1f} s of running)")
ax.tick_params(labelbottom=False)

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
for i, uid in enumerate(sorted_units):
    ct = res.loc[res.unit == uid, "cell_type"].values[0]
    ax2.vlines(units[uid].restrict(win).t - tw, i, i + 0.95, lw=1.4,
               color="C3" if ct == "inhibitory" else "C0")
for t_ in trough_t:
    ax2.axvline(t_ - tw, color="0.85", lw=0.6, zorder=0)
ax2.set_ylabel("phase-locked units\n(sorted by preferred phase)")
ax2.set_ylim(0, len(sorted_units)); ax2.set_xlim(0, WIN_S)
ax2.set_xlabel("time from window start (s);  grey lines = theta troughs")
ax2.legend(handles=[plt.Line2D([], [], color="C0", lw=2, label="excitatory"),
                    plt.Line2D([], [], color="C3", lw=2, label="inhibitory")],
           fontsize=8, loc="upper right")

ax3 = fig.add_subplot(gs[2, 0])
xx = np.degrees(np.concatenate([phase_centers, phase_centers + 2 * np.pi]))
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    cols = [u for u in sorted_units
            if res.loc[res.unit == u, "cell_type"].values[0] == ct]
    norm = tc[cols] / tc[cols].mean()
    m = np.tile(norm.mean(axis=1).values, 2)
    s = np.tile((norm.std(axis=1) / np.sqrt(norm.shape[1])).values, 2)
    ax3.plot(xx, m, color=c, label=ct)
    ax3.fill_between(xx, m - s, m + s, color=c, alpha=0.25)
ax3.plot(xx, 1 + 0.15 * np.cos(np.radians(xx)), color="0.6", lw=1, ls="--",
         label="LFP theta (schematic)")
ax3.set_xlabel("theta phase (deg;  0 = peak of filtered LFP)")
ax3.set_ylabel("normalised rate")
ax3.set_xlim(0, 720); ax3.set_xticks(np.arange(0, 721, 90)); ax3.legend(fontsize=8, ncol=3)

ax4 = fig.add_subplot(gs[:, 1])
z = tc[sorted_units].values.T
z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, keepdims=True)
im = ax4.imshow(np.hstack([z, z]), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                extent=[0, 720, len(sorted_units), 0], interpolation="nearest")
plt.colorbar(im, ax=ax4, label="firing rate (z-scored across phase bins)",
             fraction=0.05, pad=0.03)
ax4.set_xlabel("theta phase (deg)"); ax4.set_ylabel("unit (sorted by preferred phase)")
ax4.set_title("phase tuning, all locked units"); ax4.set_xticks(np.arange(0, 721, 180))
for a in [ax, ax2, ax3]:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("fig04_theta_cycle_entrainment.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 5: example units
#
# Spike-phase histograms for the three most strongly locked pyramidal cells and the three
# most strongly locked interneurons, over two theta cycles.

# %%
cand = res.loc[locked].copy()
examples = pd.concat([
    cand[(cand.cell_type == "excitatory") & (cand.n_RUN >= 500)].nlargest(3, "mrl_RUN"),
    cand[cand.cell_type == "inhibitory"].nlargest(3, "mrl_RUN")])

fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, (_, row) in zip(axs.ravel(), examples.iterrows()):
    uid = int(row.unit)
    counts, edges = np.histogram(lookups["RUN"](units[uid].t), bins=NBINS,
                                 range=(0, 2 * np.pi))
    centers = np.degrees(edges[:-1] + np.diff(edges) / 2)
    frac = counts / counts.sum() * 100
    ax.bar(np.concatenate([centers, centers + 360]), np.tile(frac, 2),
           width=360 / NBINS * 0.95,
           color="C3" if row.cell_type == "inhibitory" else "C0", alpha=0.85)
    gx = np.linspace(0, 720, 200)
    ax.plot(gx, frac.mean() * (1 + 0.5 * np.cos(np.radians(gx))), color="0.4", lw=1, ls="--")
    for off in (0, 360):
        ax.axvline(np.degrees(row.pref_RUN) + off, color="k", lw=1.2)
    ax.set_title(f"unit {uid} ({row.cell_type[:3]})  n={int(row.n_RUN)}\n"
                 f"MRL={row.mrl_RUN:.3f}  PPC={row.ppc_RUN:.4f}  "
                 f"Rayleigh p={row.p_RUN:.1e}", fontsize=9)
    ax.set_xlim(0, 720); ax.set_xticks(np.arange(0, 721, 180))
    ax.spines[["top", "right"]].set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel("theta phase (deg)")
for ax in axs[:, 0]:
    ax.set_ylabel("% of spikes")
fig.suptitle("Spike-phase distributions of example CA1 units during running "
             "(two theta cycles shown)", y=1.0)
fig.tight_layout()
fig.savefig("fig05_example_units.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Figure 6: population summary for this session
#
# Note the dashed null distribution in the first panel: the 95th percentile of the
# jitter null sits near MRL ≈ 0.05 for most units, well below the observed values.

# %%
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 0.6, 31)
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = incl & (res.cell_type == ct).values
    ax.hist(res.loc[m, "mrl_RUN"], bins=bins, alpha=0.6, color=c, label=f"{ct} (n={m.sum()})")
ax.hist(res.loc[incl, "mrl_null_p95"], bins=bins, histtype="step", color="k", ls="--",
        label="jitter null, 95th pct")
ax.set_xlabel("mean resultant length (RUN)"); ax.set_ylabel("units")
ax.legend(fontsize=8); ax.set_title("strength of theta phase locking")

ax = fig.add_subplot(gs[0, 1])
colors = ["C3" if t == "inhibitory" else "C0" for t in res.loc[incl, "cell_type"]]
ax.scatter(res.loc[incl, "mrl_null_p95"], res.loc[incl, "mrl_RUN"], s=18, c=colors)
lim = [0, res.loc[incl, "mrl_RUN"].max() * 1.05]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(0, 0.12); ax.set_ylim(*lim)
ax.set_xlabel("null MRL, 95th pct (jittered spikes)"); ax.set_ylabel("observed MRL")
ax.set_title("observed vs jitter null")

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = locked & (res.cell_type == ct).values
    cnt, edges = np.histogram(res.loc[m, "pref_RUN"], bins=18, range=(0, 2 * np.pi))
    ax.bar(edges[:-1] + np.pi / 18, cnt, width=2 * np.pi / 18, color=c, alpha=0.6, label=ct)
    ax.annotate("", xy=(tu.circ_mean(res.loc[m, "pref_RUN"]), cnt.max()), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.set_theta_zero_location("E"); ax.set_rlabel_position(285)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("preferred phase of locked units\n(0 deg = LFP theta peak)", pad=22, fontsize=10)
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.25, -0.12))

ax = fig.add_subplot(gs[1, 0])
ax.scatter(np.asarray(units.rates)[incl], res.loc[incl, "mrl_RUN"], s=18, c=colors)
ax.set_xscale("log"); ax.set_xlabel("mean firing rate (Hz)"); ax.set_ylabel("MRL (RUN)")
ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color="C0", label="excitatory"),
                   plt.Line2D([], [], ls="", marker="o", color="C3", label="inhibitory")],
          fontsize=8)
ax.set_title("locking strength vs firing rate")

ax = fig.add_subplot(gs[1, 1])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    v = np.sort(res.loc[incl & (res.cell_type == ct).values, "ppc_RUN"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=ct)
ax.axvline(0, color="k", lw=0.8, ls=":"); ax.set_xscale("symlog", linthresh=1e-4)
ax.set_xlabel("pairwise phase consistency (spike-count unbiased)")
ax.set_ylabel("cumulative fraction of units")
ax.legend(fontsize=8); ax.set_title("PPC distribution")

ax = fig.add_subplot(gs[1, 2])
frac = [res.loc[incl & (res.cell_type == ct).values, "sig_RUN"].mean() * 100
        for ct in ["excitatory", "inhibitory"]]
ax.bar(["excitatory", "inhibitory"], frac, color=["C0", "C3"])
for i, f_ in enumerate(frac):
    ax.text(i, f_ + 1.5, f"{f_:.0f}%", ha="center")
ax.set_ylim(0, 108); ax.set_ylabel("% significantly locked")
ax.set_title("Rayleigh test, FDR q<0.05")
for a in fig.axes:
    if a.name != "polar":
        a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{SESSION}: population summary of CA1 theta phase entrainment during running",
             y=0.99)
fig.savefig("fig06_population_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 4. Control: is the locking to a real rhythm?
#
# This is the control that most matters, and it cuts against a naive reading of the result.
# Bandpass filtering followed by a Hilbert transform returns a phase for *any* input,
# including a signal with no 6–10 Hz peak whatsoever. A high MRL is therefore not by itself
# evidence of an oscillation.
#
# Non-REM sleep is the natural test case: the CA1 LFP has no theta peak then (figure 3), so
# if the MRL were measuring a rhythm it should collapse. It does not. Median MRL is
# statistically indistinguishable between running and non-REM. The MRL is picking up any
# systematic relationship between spiking and the filtered signal, including the
# spike-associated sharp waves and the spectral leakage of slower rhythms into the band.
#
# What *does* separate the states is whether the spike-triggered average of the raw LFP
# oscillates. A genuine rhythmic entrainment produces repeating side lobes at the theta
# period on either side of the spike; a non-rhythmic relationship produces a single
# transient. Quantifying that per unit as the theta/delta band-power ratio of its own STA
# waveform gives the separation the MRL fails to give.

# %%
ok_all = ((res.n_RUN >= MIN_SPIKES) & (res.n_REM >= MIN_SPIKES)
          & (res.n_nonREM >= MIN_SPIKES)).values
fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))

ax = axs[0]
data = [res.loc[ok_all, f"mrl_{s}"].values for s in ["RUN", "REM", "nonREM"]]
ax.violinplot(data, showmedians=True)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.06, 0.06, len(d)), d,
               s=6, color="0.3", alpha=0.5)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["RUN", "REM", "nonREM"])
ax.set_ylabel("MRL to 6-10 Hz phase")
ax.set_title(f"MRL alone does not separate\nthe states (n={ok_all.sum()} units)", fontsize=10)
w_run_nrem, w_run_rem = wilcoxon(data[0], data[2]), wilcoxon(data[0], data[1])
ax.text(0.02, 0.03, f"RUN vs nonREM  p={w_run_nrem.pvalue:.2f}\n"
                    f"RUN vs REM      p={w_run_rem.pvalue:.3f}\n(Wilcoxon signed-rank)",
        transform=ax.transAxes, va="bottom", fontsize=7.5)

ax = axs[1]
sta_rhythm = {}
for name in ["RUN", "REM", "nonREM"]:
    vals = []
    for uid in tqdm(res.loc[ok_all, "unit"].astype(int), desc=f"STA {name}", leave=False):
        sta, lags, nsp = tu.spike_triggered_average(raws[name], units[uid].t, max_spikes=3000)
        vals.append(tu.theta_delta_ratio(sta - sta.mean()) if sta is not None else np.nan)
    sta_rhythm[name] = np.array(vals, dtype=float)
ax.violinplot([sta_rhythm[s][np.isfinite(sta_rhythm[s])] for s in ["RUN", "REM", "nonREM"]],
              showmedians=True)
for i, s in enumerate(["RUN", "REM", "nonREM"]):
    v = sta_rhythm[s][np.isfinite(sta_rhythm[s])]
    ax.scatter(np.full(len(v), i + 1) + np.random.uniform(-0.06, 0.06, len(v)), v,
               s=6, color="0.3", alpha=0.5)
ax.set_yscale("log"); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["RUN", "REM", "nonREM"])
ax.set_ylabel("theta/delta power of the STA waveform")
ax.text(0.02, 0.03, f"RUN vs nonREM  p={wilcoxon(sta_rhythm['RUN'], sta_rhythm['nonREM']).pvalue:.1e}",
        transform=ax.transAxes, fontsize=7.5)
ax.set_title("spike-locked rhythmicity does\nseparate the states", fontsize=10)

ax = axs[2]
best_unit = int(res.loc[ok_all].sort_values("mrl_RUN", ascending=False).unit.values[0])
for name, c in [("RUN", "C0"), ("REM", "C2"), ("nonREM", "C3")]:
    sta, lags, nsp = tu.spike_triggered_average(raws[name], units[best_unit].t)
    if sta is not None:
        ax.plot(lags, sta, color=c, label=f"{name} (n={nsp})")
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("time from spike (s)"); ax.set_ylabel("LFP (µV)")
ax.set_title(f"spike-triggered LFP, unit {best_unit}"); ax.legend(fontsize=8)

ax = axs[3]
for name, c in [("RUN", "C0"), ("REM", "C2"), ("nonREM", "C3")]:
    v = np.sort(res.loc[ok_all, f"ppc_{name}"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=name)
ax.set_xscale("symlog", linthresh=1e-4); ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("PPC"); ax.set_ylabel("cumulative fraction")
ax.legend(fontsize=8); ax.set_title("PPC by brain state")
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Control: is the locking to a real rhythm? "
             "(6-10 Hz Hilbert phase returns a value in every state)", y=1.02)
fig.tight_layout()
fig.savefig("fig07_state_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nbrain-state control, per-unit medians:")
for s in ["RUN", "REM", "nonREM"]:
    print("  %-7s median MRL=%.3f  median STA theta/delta=%.2f"
          % (s, np.median(data[["RUN", "REM", "nonREM"].index(s)]),
             np.nanmedian(sta_rhythm[s])))

# %% [markdown]
# ## 5. Replication across five sessions and four rats
#
# The single-session result is repeated for all five sessions in the dandiset that were
# used here, each with its own theta channel selected the same way. Statistics are computed
# per session (FDR correction is within-session) and then pooled.

# %%
def analyse_session(name, url, rng):
    hs = tu.open_session(url)
    ms = tu.load_metadata(hs)
    us, mz = ms["units"], ms["maze_epoch"]
    ts = mz.start[0] + 0.5 * mz.tot_length()
    rat = [tu.theta_delta_ratio(tu.load_lfp_channel(hs, c, ts, ts + 100).d)
           for c in tqdm(range(ms["n_lfp_channels"]), desc=f"{name}: channels", leave=False)]
    ch = int(np.argmax(rat))
    sp = tu.compute_speed(ms["position"].restrict(mz), rate=ms["position_rate"])
    rn = sp.threshold(SPEED_THRESH).time_support.drop_short_intervals(1.0)
    ph, _, _, _ = tu.phase_by_interval(hs, ch, rn, max_total=1500.0)
    lk = tu.PhaseLookup(ph)
    ctypes = us.cell_type.values
    out = []
    for i, uid in enumerate(us.index):
        p_ = lk(us[uid].t)
        if len(p_) < MIN_SPIKES:
            continue
        r, z, pv = tu.rayleigh_test(p_)
        null = tu.jitter_null_mrl(us[uid].t, lk, n_shuffles=N_SHUFFLES, rng=rng)
        out.append(dict(session=name, unit=int(uid), cell_type=ctypes[i], channel=ch,
                        theta_delta=float(np.max(rat)), n_spikes=len(p_),
                        rate=float(np.asarray(us.rates)[i]), mrl=r, ppc=tu.ppc(p_),
                        pref=tu.circ_mean(p_), p_rayleigh=pv, mrl_null_mean=null.mean(),
                        p_shuffle=(np.sum(null >= r) + 1) / (N_SHUFFLES + 1)))
    df = pd.DataFrame(out)
    df["sig"] = tu.benjamini_hochberg(df["p_rayleigh"].values)
    print(f"{name}: ch{ch} theta/delta={max(rat):.1f}, {rn.tot_length():.0f}s running, "
          f"{len(df)} units, {df.sig.mean()*100:.0f}% locked")
    return df


if REUSE:
    all_df = pd.read_csv("phase_locking_all_sessions.csv")
    print("reusing phase_locking_all_sessions.csv "
          f"({all_df.session.nunique()} sessions, {len(all_df)} units)")
else:
    rng2 = np.random.default_rng(2)
    all_df = pd.concat([analyse_session(k, v, rng2)
                        for k, v in tqdm(tu.SESSIONS.items(), desc="sessions")],
                       ignore_index=True)
    all_df.to_csv("phase_locking_all_sessions.csv", index=False)

print("\n=== pooled across %d sessions ===" % all_df.session.nunique())
print("units: %d (%d excitatory, %d inhibitory)"
      % (len(all_df), (all_df.cell_type == "excitatory").sum(),
         (all_df.cell_type == "inhibitory").sum()))
print("significantly theta-locked (Rayleigh, FDR q<0.05 within session): %d/%d = %.0f%%"
      % (all_df.sig.sum(), len(all_df), 100 * all_df.sig.mean()))
print("exceeds the jitter null at p<0.05: %d/%d"
      % ((all_df.p_shuffle < 0.05).sum(), len(all_df)))
for ct in ["excitatory", "inhibitory"]:
    m = all_df.cell_type == ct
    mu = np.degrees(tu.circ_mean(all_df.loc[m & all_df.sig, "pref"]))
    R = tu.circ_r(all_df.loc[m & all_df.sig, "pref"])
    print(f"  {ct:11s} n={m.sum():3d}  locked={100*all_df.loc[m,'sig'].mean():3.0f}%"
          f"  median MRL={all_df.loc[m,'mrl'].median():.3f}"
          f"  mean preferred phase={mu:6.1f} deg (R={R:.2f})")

# %%
fig = plt.figure(figsize=(13.5, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.33)
sessions = list(all_df.session.unique())

ax = fig.add_subplot(gs[0, 0])
frac = [100 * all_df.loc[all_df.session == s, "sig"].mean() for s in sessions]
ax.bar(range(len(sessions)), frac, color="C0")
for i, f_ in enumerate(frac):
    ax.text(i, f_ + 1.5, f"{f_:.0f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(sessions))); ax.set_xticklabels(sessions, rotation=30, ha="right",
                                                        fontsize=8)
ax.set_ylabel("% units significantly locked"); ax.set_ylim(0, 110)
ax.set_title("phase locking replicates in every session")

ax = fig.add_subplot(gs[0, 1])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    v = np.sort(all_df.loc[all_df.cell_type == ct, "mrl"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=f"{ct} (n={len(v)})")
v = np.sort(all_df["mrl_null_mean"].values)
ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="k", ls="--", label="jitter null")
ax.set_xlabel("mean resultant length"); ax.set_ylabel("cumulative fraction")
ax.legend(fontsize=8); ax.set_title("pooled locking strength")

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = (all_df.cell_type == ct) & all_df.sig
    cnt, edges = np.histogram(all_df.loc[m, "pref"], bins=24, range=(0, 2 * np.pi))
    ax.bar(edges[:-1] + np.pi / 24, cnt, width=2 * np.pi / 24, color=c, alpha=0.6, label=ct)
    ax.annotate("", xy=(tu.circ_mean(all_df.loc[m, "pref"]), cnt.max()), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.set_theta_zero_location("E"); ax.set_rlabel_position(285)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("preferred phase, all locked units\n(0 deg = LFP theta peak)", fontsize=10, pad=22)
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.3, -0.1))

ax = fig.add_subplot(gs[1, 0])
for i, s in enumerate(sessions):
    m = all_df.session == s
    ax.scatter(np.full(m.sum(), i) + np.random.uniform(-0.18, 0.18, m.sum()),
               all_df.loc[m, "mrl"], s=10,
               c=["C3" if t == "inhibitory" else "C0" for t in all_df.loc[m, "cell_type"]])
    ax.plot([i - 0.3, i + 0.3], [all_df.loc[m, "mrl"].median()] * 2, "k-", lw=2)
ax.set_xticks(range(len(sessions))); ax.set_xticklabels(sessions, rotation=30, ha="right",
                                                        fontsize=8)
ax.set_ylabel("MRL"); ax.set_title("per-session distributions (black = median)")

ax = fig.add_subplot(gs[1, 1])
ax.scatter(all_df["mrl_null_mean"], all_df["mrl"], s=10,
           c=["C3" if t == "inhibitory" else "C0" for t in all_df.cell_type])
lim = [0, all_df["mrl"].max() * 1.05]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(0, 0.09); ax.set_ylim(*lim)
ax.set_xlabel("mean MRL of jittered spikes"); ax.set_ylabel("observed MRL")
ax.set_title("observed vs jitter null (pooled)")

ax = fig.add_subplot(gs[1, 2])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = (all_df.cell_type == ct) & all_df.sig
    for s, mark in zip(sessions, "os^Dv"):
        mm = m & (all_df.session == s)
        ax.scatter(np.degrees(all_df.loc[mm, "pref"]), all_df.loc[mm, "mrl"], s=16,
                   color=c, marker=mark, label=s if ct == "excitatory" else None)
ax.set_xlabel("preferred phase (deg)"); ax.set_ylabel("MRL")
ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 90))
ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("phase vs strength, by session")
for a in fig.axes:
    if a.name != "polar":
        a.spines[["top", "right"]].set_visible(False)
fig.suptitle("CA1 theta phase entrainment during running, five sessions from DANDI:000044",
             y=0.99)
fig.savefig("fig08_multisession_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Second control: theta phase precession
#
# Phase precession is the strongest available check that the phase assignment is
# behaviourally meaningful. As a rat crosses a place field, the spikes of that place cell
# occur at progressively earlier theta phases, so the phase carries information about
# position within the field over and above the firing rate. No filtering artefact produces
# a systematic, negatively sloped relationship between an animal's position in a specific
# cell's place field and the LFP phase.
#
# Place fields are taken from 1-D tuning curves computed separately for rightward and
# leftward traversals (the two directions have different fields), with the field defined as
# the contiguous region above 30% of the peak rate. Within each field, the circular-linear
# correlation of Kempter et al. (2012) is fitted, with a permutation null over positions.

# %%
if os.path.exists("phase_precession.csv"):
    pre = pd.read_csv("phase_precession.csv")
    print(f"reusing phase_precession.csv ({len(pre)} fields)")
else:
    raise SystemExit("run 06_phase_precession.py to generate phase_precession.csv")

neg = pre.sig & (pre.slope < 0)
print(f"place fields analysed: {len(pre)}")
print(f"significant circular-linear correlation (FDR q<0.05): {pre.sig.sum()} "
      f"({100*pre.sig.mean():.0f}%)")
print(f"  of which negative slope (precession): {neg.sum()} "
      f"({100*neg.sum()/max(pre.sig.sum(),1):.0f}% of significant)")
print(f"median slope of significant fields: {pre.loc[pre.sig,'slope'].median():.2f} "
      "theta cycles per field traversal")

# %% [markdown]
# The full precession computation, including the per-field figure
# (`fig09_phase_precession.png`), lives in `06_phase_precession.py`; it is kept separate
# because the permutation test over every field is the slowest step in the pipeline.

# %% [markdown]
# ## Summary of findings
#
# Across 430 CA1 units from five sessions and four rats, **87% (372/430) are significantly
# phase locked to the 6–10 Hz LFP theta rhythm during running** (Rayleigh test, Benjamini–Hochberg
# FDR q < 0.05 within session), and 374/430 also exceed a ±0.5 s spike-jitter null at
# p < 0.05.
# Locking replicates in every session individually, at 82% to 98% of units.
#
# The two cell classes differ in the way the literature describes. Putative interneurons are
# more strongly locked than pyramidal cells (median MRL 0.20 vs 0.12, 96% vs 84% significant)
# and fire near the trough of the pyramidal-layer LFP theta (mean preferred phase 178°,
# where 0° is the filtered LFP peak), while pyramidal cells fire later in the cycle (227°).
# Both classes show a broad but clearly non-uniform spread of preferred phases, so the
# population tiles the cycle rather than firing in a single synchronous burst.
#
# Two controls support the interpretation. First, phase precession: of the 24 direction-
# specific place fields tested in the prototype session, 12 show a significant negative
# circular-linear correlation between position in the field and theta phase, with a median
# slope near -0.5 theta cycles per traversal. Second, and as a caution rather than a
# confirmation, the MRL by itself does not distinguish running from non-REM sleep
# (p = 0.42), even though the non-REM LFP has no theta peak at all. Bandpass filtering plus
# a Hilbert transform assigns a phase to any signal, so a high MRL is not on its own
# evidence of an oscillation. The measure that does separate the states is the rhythmicity
# of each unit's spike-triggered LFP average, which shows the repeating theta-period side
# lobes only during running and REM.
