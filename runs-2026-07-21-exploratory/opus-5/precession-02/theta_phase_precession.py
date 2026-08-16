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
# # Theta phase precession in hippocampal CA1 place cells
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — *Diversity in
# neural firing dynamics supports both rigid and learned hippocampal sequences*
# (Grosmark & Buzsáki, Science 2016), the dataset also distributed as `hc-11` on
# CRCNS. Four rats ran back and forth on linear tracks for water reward while
# bilateral silicon probes recorded CA1 spikes and local field potentials.
#
# **Phenomenon.** As a rat traverses the firing field of a CA1 place cell, the
# cell's spikes occur at progressively earlier phases of the ongoing 6–12 Hz
# theta rhythm (O'Keefe & Recce, 1993). Position within the field is therefore
# encoded twice: by *which* cells fire (a rate code) and by *when in the theta
# cycle* they fire (a phase code). This notebook demonstrates the effect from
# raw NWB data: it extracts theta phase from the LFP, builds place fields from
# the linearized position, and regresses spike phase on within-field position
# using the circular-linear method of Kempter et al. (2012).
#
# **What is done here**
#
# 1. Stream the NWB files from the DANDI S3 bucket with `remfile` (no full download).
# 2. Segment the maze epoch into rightward and leftward running epochs.
# 3. Pick a theta reference LFP channel by theta / background power ratio, band-pass
#    it at 6–12 Hz, and take the Hilbert phase.
# 4. Build direction-specific 1D rate maps and detect place fields.
# 5. For every (unit × direction) field, regress in-field spike phase on
#    normalized within-field travel distance; assess significance by a
#    position-phase shuffle.
# 6. Repeat for all five linear-track sessions in the dandiset and pool.
#
# All figures are written to `figures/`.

# %%
import os
import re
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.stats import norm, wilcoxon, binomtest
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

DANDISET = "000044"
CACHE = "/tmp/remfile_cache"

# asset ids for the eight sessions of DANDI:000044
ASSETS = {
    "Achilles-10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles-11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Buddy-06272013":    "185b8a36-d671-4688-ba05-9e89a902c486",
    "Cicero-09012014":   "97252767-5e90-45cf-a1b8-8cbff2f2a3a3",
    "Cicero-09102014":   "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Cicero-09172014":   "d9a1e010-af4a-4fba-812d-66e2d1a71e35",
    "Gatsby-08022013":   "93569d6c-781f-4422-938e-e935a62863de",
    "Gatsby-08282013":   "402f78e1-9e7d-486c-8822-93d538ed6ccc",
}

# Three of the eight sessions use a circular maze whose linearized coordinate wraps
# around; the analysis below assumes a bidirectional linear track, so only the five
# linear-track sessions are used.
LINEAR_SESSIONS = ["Achilles-10252013", "Buddy-06272013", "Cicero-09012014",
                   "Cicero-09172014", "Gatsby-08022013"]
PROTOTYPE = "Achilles-10252013"


# %% [markdown]
# ## Loading helpers
#
# The NWB files are 5–9 GB each, so they are opened over HTTP with `remfile` and a
# local disk cache. Only the byte ranges actually touched (one LFP channel over the
# maze epoch, the spike times, the position) are transferred.

# %%
def url_for(session):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
            f"/versions/draft/assets/{ASSETS[session]}/download/")


def open_session(session):
    """Open one NWB file over HTTP; return the raw h5py handle and a pynapple NWBFile."""
    rf = remfile.File(url_for(session), disk_cache=remfile.DiskCache(CACHE))
    h5 = h5py.File(rf, "r")
    nwbfile = NWBHDF5IO(file=h5, load_namespaces=True).read()
    return h5, nap.NWBFile(nwbfile)


def maze_epoch(nwb):
    """IntervalSet for the maze epoch (each session is PRE sleep / maze / POST sleep)."""
    ep = nwb["epochs"]
    labels = [str(l) for l in np.asarray(ep.label)]
    i = labels.index("MazeEpoch")
    return nap.IntervalSet(start=ep.start[i], end=ep.end[i])


def maze_name(nwb):
    return [k for k in nwb.keys() if "Linearized" in k][0]


def track_length(nwb):
    """Nominal track length in metres, parsed from the series name (e.g. '1.6mLinearMaze')."""
    m = re.match(r"([\d.]+)m", maze_name(nwb))
    if m is None:
        raise ValueError(f"not a linear maze: {maze_name(nwb)}")
    return float(m.group(1))


def linear_position(nwb):
    """Linearized position as a pynapple Tsd. The stored series is NaN whenever the
    animal is off the linearized part of the track (i.e. in a reward area), so
    dropping NaNs already restricts us to track traversals."""
    lin = nwb[maze_name(nwb)]
    v = np.asarray(lin.values).ravel()
    good = ~np.isnan(v)
    return nap.Tsd(t=lin.t[good], d=v[good])


# %% [markdown]
# ## Prototype session: what is in the file

# %%
h5, nwb = open_session(PROTOTYPE)
print(nwb)

# %%
units = nwb["units"]
pos = linear_position(nwb)
ep = maze_epoch(nwb)
L = track_length(nwb)
lfp_grp = h5["processing/ecephys/LFP/LFP"]
FS = float(lfp_grp["starting_time"].attrs["rate"])
N_CHAN = lfp_grp["data"].shape[1]

print(f"session          : {PROTOTYPE}")
print(f"maze epoch       : {ep.start[0]:.1f} - {ep.end[0]:.1f} s ({ep.tot_length():.0f} s)")
print(f"track            : {maze_name(nwb)}  ->  {L} m")
print(f"units            : {len(units)}  "
      f"({np.sum([str(c) == 'excitatory' for c in units.cell_type])} excitatory, "
      f"{np.sum([str(c) == 'inhibitory' for c in units.cell_type])} inhibitory)")
print(f"unit locations   : {np.unique([str(x) for x in units.location], return_counts=True)}")
print(f"LFP              : {lfp_grp['data'].shape} @ {FS} Hz")
print(f"position samples : {len(pos)} valid of "
      f"{nwb[maze_name(nwb)].shape[0]} (rest are tracking NaNs)")

# %% [markdown]
# ## Running epochs
#
# Velocity is the smoothed time derivative of the linearized position. Samples on
# either side of a tracking gap are blanked so that the jump across a gap does not
# masquerade as a fast run. Epochs faster than 10 cm/s and longer than 0.5 s are kept
# and split by sign into rightward and leftward runs.

# %%
def run_epochs(pos, speed_thresh=0.10, min_dur=0.5, smooth_s=0.25):
    t, x = pos.t, pos.values
    dt = np.median(np.diff(t))
    xs = gaussian_filter1d(x, max(smooth_s / dt, 1.0))
    v = np.gradient(xs, t)
    gap = np.r_[np.diff(t), dt] > 5 * dt
    gap |= np.r_[dt, np.diff(t)] > 5 * dt
    v[gap] = 0.0

    def _epochs(mask):
        e = nap.Tsd(t=t, d=mask.astype(float)).threshold(0.5).time_support
        return e.drop_short_intervals(min_dur)

    return _epochs(v > speed_thresh), _epochs(v < -speed_thresh), nap.Tsd(t=t, d=v)


ep_R, ep_L, velocity = run_epochs(pos)
print(f"rightward: {len(ep_R):3d} epochs, {ep_R.tot_length():.0f} s")
print(f"leftward : {len(ep_L):3d} epochs, {ep_L.tot_length():.0f} s")
print(f"median running speed: {np.median(np.abs(velocity.restrict(ep_R.union(ep_L)).values)):.2f} m/s")

# %%
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=.45, wspace=.28)

ax = fig.add_subplot(gs[0, 0])
xy = np.asarray(nwb[[k for k in nwb.keys() if "SpatialSeries" in k][0]].values)
ax.plot(xy[:, 0], xy[:, 1], ".", ms=1, alpha=.2, color="0.3")
ax.set(xlabel="x (m)", ylabel="y (m)", title="raw tracking, maze epoch")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1])
for e, c in [(ep_R, "tab:blue"), (ep_L, "tab:red")]:
    for st, en in list(zip(e.start, e.end))[:40]:
        seg = pos.restrict(nap.IntervalSet(st, en))
        ax.plot(seg.t - st, seg.values, color=c, lw=.7, alpha=.6)
ax.set(xlabel="time from run onset (s)", ylabel="position (m)",
       title="traversals (blue = rightward,\nred = leftward)")

ax = fig.add_subplot(gs[0, 2])
spd = np.abs(velocity.restrict(ep_R.union(ep_L)).values)
ax.hist(spd, bins=40, color="0.5")
ax.axvline(np.median(spd), color="tab:red", ls="--",
           label=f"median {np.median(spd):.2f} m/s")
ax.set(xlabel="running speed (m/s)", ylabel="# samples", title="speed during run epochs")
ax.legend(fontsize=7)

w = nap.IntervalSet(ep.start[0] + 200, ep.start[0] + 320)
p, v = pos.restrict(w), velocity.restrict(w)
ax = fig.add_subplot(gs[1, :])
ax.plot(p.t, p.values, "k.", ms=2)
for e, c in [(ep_R, "tab:blue"), (ep_L, "tab:red")]:
    for st, en in zip(e.start, e.end):
        if st < w.end[0] and en > w.start[0]:
            ax.axvspan(st, en, color=c, alpha=.18)
ax.set(ylabel="linearized position (m)", xlim=(w.start[0], w.end[0]),
       title="run-epoch detection over a 120 s window")
ax = fig.add_subplot(gs[2, :])
ax.plot(v.t, v.values, "k", lw=.8)
ax.axhline(0.1, color="tab:blue", ls="--", lw=.8)
ax.axhline(-0.1, color="tab:red", ls="--", lw=.8)
ax.set(xlabel="time (s)", ylabel="velocity (m/s)", xlim=(w.start[0], w.end[0]))
fig.suptitle(f"{PROTOTYPE}: behaviour on the {L} m linear track", y=.96)
plt.savefig("figures/fig01_track_and_runs.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Theta reference channel and phase
#
# All 128 channels carry a clean theta peak; the reference channel is chosen
# objectively as the one with the largest 6–12 Hz power relative to the surrounding
# band (2–5 and 13–25 Hz), measured on the first 400 s of the maze epoch. The signal
# is then band-passed at 6–12 Hz and the instantaneous phase is taken as the angle of
# the analytic signal, so that **0° / 360° is the peak of the theta-filtered LFP and
# 180° is the trough**. The recording depth of each channel is not annotated in the
# file, so the phase should be read as relative to this channel's own theta cycle
# rather than as a layer-referenced absolute phase.

# %%
def pick_theta_channel(h5, ep, channels, max_seconds=400.0):
    lfp = h5["processing/ecephys/LFP/LFP"]
    fs = float(lfp["starting_time"].attrs["rate"])
    conv = float(lfp["data"].attrs["conversion"])
    i0 = int(ep.start[0] * fs)
    i1 = min(int(ep.end[-1] * fs), i0 + int(max_seconds * fs))
    ratios, psds = [], []
    for c in tqdm(channels, desc="scanning LFP channels", leave=False):
        y = lfp["data"][i0:i1, c].astype(np.float32) * conv
        f, P = welch(y, fs=fs, nperseg=int(4 * fs))
        th = (f >= 6) & (f <= 12)
        bg = ((f >= 2) & (f < 5)) | ((f > 13) & (f < 25))
        ratios.append(P[th].mean() / P[bg].mean())
        psds.append(P)
    return int(channels[np.argmax(ratios)]), np.asarray(ratios), f, np.asarray(psds)


def theta_phase(h5, channel, ep, band=(6.0, 12.0)):
    """Band-pass one LFP channel over `ep`; return raw, filtered, phase (deg), envelope."""
    lfp = h5["processing/ecephys/LFP/LFP"]
    fs = float(lfp["starting_time"].attrs["rate"])
    conv = float(lfp["data"].attrs["conversion"])
    i0, i1 = int(ep.start[0] * fs), int(ep.end[-1] * fs)
    y = lfp["data"][i0:i1, channel].astype(np.float32) * conv
    t = i0 / fs + np.arange(len(y)) / fs
    raw = nap.Tsd(t=t, d=y)
    filt = nap.apply_bandpass_filter(raw, band, fs=fs)
    ph = nap.Tsd(t=t, d=np.degrees(np.asarray(nap.compute_hilbert_phase(filt))) % 360.0)
    env = nap.compute_hilbert_envelope(filt)
    return raw, filt, ph, env


scan_channels = np.arange(0, N_CHAN, 4)
theta_ch, theta_ratios, freqs, psds = pick_theta_channel(h5, ep, scan_channels)
raw, filt, phase, env = theta_phase(h5, theta_ch, ep)
print(f"theta reference channel: {theta_ch} (theta/background ratio "
      f"{theta_ratios.max():.1f}, range over channels {theta_ratios.min():.1f}-{theta_ratios.max():.1f})")

inst_f = np.diff(np.unwrap(np.radians(phase.values))) / np.diff(phase.t) / (2 * np.pi)
print(f"median instantaneous theta frequency: {np.median(inst_f):.2f} Hz")

# %%
# verify the phase convention empirically: where in the cycle is the filtered peak?
bin_edges = np.linspace(0, 360, 25)
idx = np.digitize(phase.values, bin_edges) - 1
mean_wave = np.array([filt.values[idx == k].mean() for k in range(24)])
centers_ph = (bin_edges[:-1] + bin_edges[1:]) / 2
print(f"filtered LFP is maximal at {centers_ph[np.argmax(mean_wave)]:.0f} deg, "
      f"minimal at {centers_ph[np.argmin(mean_wave)]:.0f} deg")

fig = plt.figure(figsize=(12, 6.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=.55, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
for k in range(len(scan_channels)):
    ax.semilogy(freqs, psds[k], lw=.6, color="0.7")
ax.semilogy(freqs, psds[np.argmax(theta_ratios)], lw=1.6, color="tab:red",
            label=f"ch {theta_ch} (selected)")
ax.set(xlim=(0, 30), xlabel="frequency (Hz)", ylabel="PSD (V$^2$/Hz)",
       title="LFP spectra, maze epoch")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.plot(scan_channels, theta_ratios, "o-", ms=3, color="0.3")
ax.plot(theta_ch, theta_ratios.max(), "o", color="tab:red")
ax.set(xlabel="channel", ylabel="theta / background power", title="reference-channel selection")

ax = fig.add_subplot(gs[0, 2])
ax.plot(centers_ph, mean_wave * 1e6, "k")
ax.axvline(centers_ph[np.argmax(mean_wave)], color="tab:red", ls="--", lw=.8)
ax.set(xlabel="Hilbert phase (deg)", ylabel="mean filtered LFP (µV)",
       title="phase convention: 0° = theta peak", xticks=[0, 90, 180, 270, 360])

t0 = ep.start[0] + 220
w = (raw.t > t0) & (raw.t < t0 + 3)
ax = fig.add_subplot(gs[1, :])
ax.plot(raw.t[w] - t0, raw.values[w] * 1e6, color="0.4", lw=.7, label="raw LFP")
ax.plot(filt.t[w] - t0, filt.values[w] * 1e6, color="tab:blue", lw=1.4, label="6-12 Hz")
ax.plot(env.t[w] - t0, np.asarray(env)[w] * 1e6, color="tab:orange", lw=1, label="envelope")
ax.set(ylabel="µV", title=f"LFP channel {theta_ch}, 3 s during running")
ax.legend(fontsize=7, ncol=3, loc="upper right")

ax = fig.add_subplot(gs[2, :])
ax.plot(phase.t[w] - t0, phase.values[w], color="tab:green", lw=.9)
ax.set(xlabel="time (s)", ylabel="theta phase (deg)", yticks=[0, 180, 360])
plt.savefig("figures/fig02_theta_lfp.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Place fields
#
# Rate maps use 2 cm position bins, smoothed with a 3 cm Gaussian, computed separately
# for rightward and leftward runs (CA1 fields on a linear track are strongly
# direction-selective). A place field is the contiguous run of bins around the peak
# that exceeds 25 % of the peak rate, with a peak above 1 Hz and a width between 12
# and 80 cm.

# %%
def rate_maps(units, pos, ep, track_len, bin_size=0.02, sigma_bins=1.5):
    nbins = int(round(track_len / bin_size))
    tc = nap.compute_tuning_curves(units, pos, bins=nbins, range=[(0, track_len)],
                                   epochs=ep, return_pandas=True)
    centers = tc.index.values
    maps = {u: gaussian_filter1d(np.nan_to_num(tc[u].values), sigma_bins, mode="nearest")
            for u in tc.columns}
    return centers, maps


def detect_field(centers, rate, peak_min=1.0, frac=0.25, wmin=0.12, wmax=0.80):
    """Contiguous region around the peak above `frac` x peak; returns (lo, hi) or None."""
    peak = rate.max()
    if peak < peak_min:
        return None
    above = rate >= frac * peak
    d = np.diff(above.astype(int))
    starts, stops = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, len(above)]
    imax = np.argmax(rate)
    k = np.where((starts <= imax) & (stops > imax))[0]
    if len(k) == 0:
        return None
    lo, hi = centers[starts[k[0]]], centers[stops[k[0]] - 1]
    return (lo, hi) if wmin <= hi - lo <= wmax else None


excitatory = units[np.array([str(c) == "excitatory" for c in units.cell_type])]
centers, maps_R = rate_maps(excitatory, pos, ep_R, L)
_, maps_L = rate_maps(excitatory, pos, ep_L, L)
n_fields = {d: sum(detect_field(centers, m[u]) is not None for u in m)
            for d, m in [("right", maps_R), ("left", maps_L)]}
print(f"place fields detected: {n_fields}  (of {len(excitatory)} excitatory units per direction)")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, m, ttl in [(axes[0], maps_R, "rightward runs"), (axes[1], maps_L, "leftward runs")]:
    M = np.array([m[u] for u in m])
    keep = M.max(1) > 1.0
    M = M[keep] / M[keep].max(1, keepdims=True)
    M = M[np.argsort(np.argmax(M, 1))]
    im = ax.imshow(M, aspect="auto", extent=[0, L, M.shape[0], 0], cmap="viridis",
                   interpolation="nearest")
    ax.set(xlabel="position (m)", title=f"{ttl} (n = {M.shape[0]} units)")
axes[0].set_ylabel("unit, sorted by field position")
cb = plt.colorbar(im, ax=axes[1]); cb.set_label("normalized rate")

ax = axes[2]
example_units = sorted(maps_L, key=lambda u: -maps_L[u].max())[:6]
for i, u in enumerate(example_units):
    r = maps_L[u]
    base = i * 1.4
    ax.plot(centers, base + r / r.max(), lw=1.2, color=f"C{i}")
    f = detect_field(centers, r)
    if f is not None:
        ax.plot([f[0], f[1]], [base - .18] * 2, lw=3, color=f"C{i}", solid_capstyle="butt")
    ax.text(0.02, base + 1.12, f"unit {u}  ({r.max():.0f} Hz peak)", fontsize=7, color=f"C{i}")
ax.set(xlabel="position (m)", yticks=[], xlim=(0, L), ylim=(-.5, 6 * 1.4 + .6),
       title="example rate maps, leftward runs\n(bar = detected field)")
plt.savefig("figures/fig03_place_fields.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Circular-linear regression
#
# For a set of spikes with within-field position $x \in [0, 1]$ and theta phase
# $\phi$, the slope $a$ (in cycles per field traversal) is the value that maximizes
# the resultant length $R(a) = \left| \langle e^{i(\phi - 2\pi a x)} \rangle \right|$,
# searched on a grid over $[-2, 2]$. The circular-linear correlation $\rho$ and its
# asymptotic normal test follow Kempter et al. (2012). Because in-field spikes from a
# single theta cycle are not independent, the asymptotic test is somewhat
# anti-conservative, so significance is instead taken from a permutation test: the
# phases are shuffled against the positions 500 times, which preserves both the
# phase distribution (theta locking) and the position distribution, and destroys only
# their pairing.
#
# Crucially, $x$ is measured in the direction the animal is *travelling*, so for
# leftward runs $x = (\text{hi} - p)/(\text{hi} - \text{lo})$. Without this the slope
# sign flips for one of the two running directions and the population median comes
# out near zero.

# %%
def _slope_basis(x, slope_range=(-2.0, 2.0), n_grid=1000):
    # an even grid over a symmetric range excludes a == 0 exactly, where the
    # circular-linear correlation below is 0/0
    a = np.linspace(*slope_range, n_grid)
    b = 2 * np.pi * a[:, None] * np.asarray(x, float)[None, :]
    return a, np.cos(b), np.sin(b)


def circ_lin_regress(x, phi_deg, basis=None, slope_range=(-2.0, 2.0), n_grid=1000):
    phi = np.radians(np.asarray(phi_deg, float))
    x = np.asarray(x, float)
    n = len(x)
    if n < 20:
        return None
    a, cosb, sinb = basis if basis is not None else _slope_basis(x, slope_range, n_grid)
    C = (cosb @ np.cos(phi) + sinb @ np.sin(phi)) / n
    S = (cosb @ np.sin(phi) - sinb @ np.cos(phi)) / n
    k = np.argmax(np.hypot(C, S))
    slope = a[k]
    phi0 = np.arctan2(S[k], C[k]) % (2 * np.pi)

    theta = (2 * np.pi * abs(slope) * x) % (2 * np.pi)
    phibar = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
    thbar = np.arctan2(np.sin(theta).sum(), np.cos(theta).sum())
    sp, st = np.sin(phi - phibar), np.sin(theta - thbar)
    denom = np.sqrt((sp ** 2).sum() * (st ** 2).sum())
    rho = 0.0 if denom == 0 else (sp * st).sum() / denom
    lam = (sp ** 2 * st ** 2).mean()
    z = 0.0 if lam == 0 else rho * np.sqrt(n * (sp ** 2).mean() * (st ** 2).mean() / lam)
    return dict(slope=slope, phi0=np.degrees(phi0), rho=rho, n=n,
                p_asym=2 * (1 - norm.cdf(abs(z))))


def shuffle_pvalue(x, phi_deg, n_shuffle=500, rng=None):
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    basis = _slope_basis(x)
    obs = circ_lin_regress(x, phi_deg, basis=basis)
    if obs is None:
        return None, None
    phi = np.asarray(phi_deg, float)
    null = np.array([abs(circ_lin_regress(x, rng.permutation(phi), basis=basis)["rho"])
                     for _ in range(n_shuffle)])
    return obs, (1 + (null >= abs(obs["rho"])).sum()) / (1 + n_shuffle)


def collect_fields(exc, pos, phase, ep_by_dir, track_len, min_spikes=50,
                   n_shuffle=500, seed=0):
    """One entry per (unit, running direction) that has a place field with enough
    in-field spikes, holding the spikes' within-field position, theta phase, and fit."""
    out = []
    rng = np.random.default_rng(seed)
    for dname, e in ep_by_dir.items():
        cent, maps = rate_maps(exc, pos, e, track_len)
        for u, r in maps.items():
            fld = detect_field(cent, r)
            if fld is None:
                continue
            lo, hi = fld
            spk = exc[u].restrict(e)
            if len(spk) == 0:
                continue
            sp_pos = pos.interpolate(spk)
            inside = (sp_pos.values >= lo) & (sp_pos.values <= hi)
            if inside.sum() < min_spikes:
                continue
            ts = nap.Ts(t=spk.t[inside])
            p_in = sp_pos.values[inside]
            x = (p_in - lo) / (hi - lo) if dname == "right" else (hi - p_in) / (hi - lo)
            phi = np.asarray(ts.value_from(phase))
            fit, p_shuf = shuffle_pvalue(x, phi, n_shuffle=n_shuffle, rng=rng)
            if fit is None:
                continue
            fit.update(p_shuffle=p_shuf, unit=int(u), direction=dname, lo=lo, hi=hi,
                       width=hi - lo, peak_rate=float(r.max()), x=x, phi=phi,
                       t=ts.t, rate_map=r, centers=cent)
            out.append(fit)
    return out


fields = collect_fields(excitatory, pos, phase, {"right": ep_R, "left": ep_L}, L)
slopes = np.array([f["slope"] for f in fields])
pvals = np.array([f["p_shuffle"] for f in fields])
sig = pvals < 0.05
print(f"{PROTOTYPE}: {len(fields)} unit x direction fields, "
      f"{sig.sum()} significant ({100 * sig.mean():.0f} %)")
print(f"median slope of significant fields: {np.median(slopes[sig]):+.2f} cycles "
      f"({360 * np.median(slopes[sig]):+.0f} deg) per traversal; "
      f"{100 * (slopes[sig] < 0).mean():.0f} % negative")

# %% [markdown]
# ## Example cells
#
# Each panel shows every in-field spike as a point: within-field position on the x
# axis, theta phase on the y axis, plotted over two theta cycles so that the
# wrap-around is visible. The red line is the fitted circular-linear regression.

# %%
# examples: significant fields with a negative (precessing) slope, strongest first
examples = sorted([f for f in fields if f["p_shuffle"] < 0.05 and f["slope"] < 0],
                  key=lambda f: f["rho"])[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), examples):
    ax.plot(f["x"], f["phi"], "k.", ms=3, alpha=.5)
    ax.plot(f["x"], f["phi"] + 360, "k.", ms=3, alpha=.5)
    xx = np.linspace(0, 1, 100)
    yy = f["phi0"] + 360 * f["slope"] * xx
    for k in range(-2, 4):
        ax.plot(xx, yy + 360 * k, "-", color="tab:red", lw=1.6)
    ax.set(xlim=(0, 1), ylim=(0, 720), yticks=[0, 180, 360, 540, 720])
    ax.set_title(f"unit {f['unit']}, {f['direction']}ward  (n = {f['n']} spikes)\n"
                 rf"$\rho$ = {f['rho']:+.2f},  slope = {360 * f['slope']:+.0f}°/field,  "
                 f"p = {f['p_shuffle']:.3f}", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("theta phase (deg)")
for ax in axes[-1, :]:
    ax.set_xlabel("normalized position in field")
fig.suptitle(f"{PROTOTYPE}: six strongest phase-precessing fields", y=1.0)
plt.tight_layout()
plt.savefig("figures/fig04_example_precession.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## A single cell, pass by pass
#
# Pooling spikes across traversals could in principle manufacture a phase-position
# relationship if, say, early and late traversals differed systematically. The next
# figure breaks one field down by individual pass: the phase-position relationship is
# present within single traversals, not only in the pooled cloud.

# %%
best = min([f for f in fields if f["n"] > 250 and f["p_shuffle"] < 0.05],
           key=lambda f: f["rho"])
ep_dir = ep_R if best["direction"] == "right" else ep_L
pass_id = np.full(len(best["t"]), -1)
for i, (st, en) in enumerate(zip(ep_dir.start, ep_dir.end)):
    pass_id[(best["t"] >= st) & (best["t"] <= en)] = i
passes = [i for i in np.unique(pass_id) if i >= 0 and (pass_id == i).sum() >= 4]
rich = sorted(passes, key=lambda i: -(pass_id == i).sum())[:4]

fig = plt.figure(figsize=(13.5, 7))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=.45, wspace=.35)

ax = fig.add_subplot(gs[0, 0])
r = best["rate_map"]
ax.plot(best["centers"], r, "k")
ax.axvspan(best["lo"], best["hi"], color="tab:orange", alpha=.25)
ax.set(xlabel="position (m)", ylabel="firing rate (Hz)",
       title=f"unit {best['unit']}, {best['direction']}ward\nrate map and detected field")

ax = fig.add_subplot(gs[0, 1])
for k, i in enumerate(passes):
    m = pass_id == i
    ax.plot(best["x"][m], np.full(m.sum(), k), "|", color="k", ms=4)
ax.set(xlabel="normalized position in field", ylabel="traversal #", xlim=(0, 1),
       title=f"in-field spikes,\n{len(passes)} traversals")

ax = fig.add_subplot(gs[0, 2:])
ax.plot(best["x"], best["phi"], "o", ms=3, color="0.35", alpha=.5)
ax.plot(best["x"], best["phi"] + 360, "o", ms=3, color="0.35", alpha=.5)
xx = np.linspace(0, 1, 100)
for k in range(-2, 4):
    ax.plot(xx, best["phi0"] + 360 * best["slope"] * xx + 360 * k, color="tab:red", lw=1.6)
ax.set(xlim=(0, 1), ylim=(0, 720), yticks=[0, 180, 360, 540, 720],
       xlabel="normalized position in field", ylabel="theta phase (deg)",
       title=rf"all traversals pooled: $\rho$ = {best['rho']:+.2f}, "
             f"slope = {360 * best['slope']:+.0f}°/field, p = {best['p_shuffle']:.3f}")

for j, i in enumerate(rich):
    ax = fig.add_subplot(gs[1, j])
    m = pass_id == i
    order = np.argsort(best["t"][m])
    xs, ps = best["x"][m][order], best["phi"][m][order]
    for k in range(-2, 4):
        ax.plot(xx, best["phi0"] + 360 * best["slope"] * xx + 360 * k,
                color="tab:red", lw=1, alpha=.5)
    ax.plot(xs, ps, "o", ms=5, color="tab:blue")
    ax.plot(xs, ps + 360, "o", ms=5, color="tab:blue")
    ax.set(xlim=(0, 1), ylim=(0, 720), yticks=[0, 180, 360, 540, 720],
           xlabel="position in field",
           title=f"traversal {i} ({m.sum()} spikes)", ylabel="phase (deg)" if j == 0 else None)
fig.suptitle("Phase precession within single field traversals "
             "(red = fit to all traversals pooled)", y=.98)
plt.savefig("figures/fig05_single_cell_passes.png", bbox_inches="tight")
plt.close()
print(f"example cell: unit {best['unit']} {best['direction']}ward, "
      f"{len(passes)} traversals, {best['n']} in-field spikes")

# %% [markdown]
# ## All five linear-track sessions
#
# The same pipeline is now run over every linear-track session in the dandiset. Each
# session takes roughly 30–60 s on a cold cache, most of it spent streaming the LFP
# channels used for reference selection.

# %%
results = {}
for s in tqdm(LINEAR_SESSIONS, desc="sessions"):
    h5_s, nwb_s = open_session(s)
    Ls = track_length(nwb_s)
    ep_s = maze_epoch(nwb_s)
    pos_s = linear_position(nwb_s)
    epR_s, epL_s, _ = run_epochs(pos_s)
    u_s = nwb_s["units"]
    exc_s = u_s[np.array([str(c) == "excitatory" for c in u_s.cell_type])]
    nch = h5_s["processing/ecephys/LFP/LFP"]["data"].shape[1]
    ch_s, ratios_s, _, _ = pick_theta_channel(h5_s, ep_s, np.arange(0, nch, 4))
    _, _, phase_s, _ = theta_phase(h5_s, ch_s, ep_s)
    res = collect_fields(exc_s, pos_s, phase_s, {"right": epR_s, "left": epL_s}, Ls)
    results[s] = dict(fields=res, channel=ch_s, track_len=Ls, n_units=len(u_s),
                      n_exc=len(exc_s), run_R=epR_s.tot_length(), run_L=epL_s.tot_length(),
                      subject=s.split("-")[0])

with open("precession_results.pkl", "wb") as fh:
    pickle.dump({s: {k: v for k, v in r.items()} for s, r in results.items()}, fh)

# %%
print(f"{'session':20s} {'track':>6s} {'exc':>4s} {'run s':>7s} {'fields':>7s} "
      f"{'sig':>10s} {'med slope':>10s} {'neg':>5s}")
for s, r in results.items():
    sl = np.array([f["slope"] for f in r["fields"]])
    pv = np.array([f["p_shuffle"] for f in r["fields"]])
    m = pv < 0.05
    print(f"{s:20s} {r['track_len']:5.1f}m {r['n_exc']:4d} "
          f"{r['run_R'] + r['run_L']:7.0f} {len(sl):7d} "
          f"{m.sum():4d} ({100 * m.mean():3.0f}%) {360 * np.median(sl[m]):+9.0f}° "
          f"{100 * (sl[m] < 0).mean():4.0f}%")

allf = [f for s in results for f in results[s]["fields"]]
sl = np.array([f["slope"] for f in allf])
pv = np.array([f["p_shuffle"] for f in allf])
rho = np.array([f["rho"] for f in allf])
nsp = np.array([f["n"] for f in allf])
sig = pv < 0.05
print(f"\nPOOLED: {len(allf)} fields from {len(results)} sessions / "
      f"{len({results[s]['subject'] for s in results})} rats")
print(f"  significant (shuffle p < 0.05): {sig.sum()} ({100 * sig.mean():.0f} %) "
      f"vs 5 % expected by chance")
print(f"  median slope, significant fields: {360 * np.median(sl[sig]):+.0f}° per traversal")
print(f"  negative slopes among significant: {100 * (sl[sig] < 0).mean():.0f} %  "
      f"(binomial p = {binomtest((sl[sig] < 0).sum(), sig.sum(), 0.5).pvalue:.1e})")
print(f"  rho over all fields: median {np.median(rho):+.3f}, "
      f"Wilcoxon vs 0 p = {wilcoxon(rho).pvalue:.1e}")

# %% [markdown]
# ## Population summary
#
# Panel A pools every in-field spike from every session: the density of spikes in the
# (within-field position, theta phase) plane forms a band running from late phase at
# field entry to early phase at field exit, repeating every 360°. Panel B restricts
# the same plot to fields whose regression is individually significant. Panel C shows
# the slope distribution, panel D the mean phase advance as a function of within-field
# position, and panels E–F the dependence on spike count and the per-session breakdown.

# %%
def phase_density(fields_subset, nx=25, nph=48):
    X = np.concatenate([f["x"] for f in fields_subset])
    P = np.concatenate([f["phi"] for f in fields_subset])
    H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360],
                               bins=[np.linspace(0, 1, nx + 1), np.linspace(0, 720, nph + 1)])
    H = H / H.sum(1, keepdims=True)
    # light smoothing for display only; circular along the phase axis
    H = gaussian_filter1d(gaussian_filter1d(H, 1.0, axis=1, mode="wrap"), 0.8, axis=0)
    return H, len(X)


def mean_phase_profile(fields_subset, nbins=10):
    """Per field, circular-mean phase in each within-field decile, unwrapped along
    position and expressed relative to the field-entry phase; then averaged."""
    edges = np.linspace(0, 1, nbins + 1)
    rows = []
    for f in fields_subset:
        idx = np.clip(np.digitize(f["x"], edges) - 1, 0, nbins - 1)
        a = np.radians(f["phi"])
        row = np.array([np.degrees(np.arctan2(np.sin(a[idx == k]).mean(),
                                              np.cos(a[idx == k]).mean()))
                        if (idx == k).sum() >= 5 else np.nan for k in range(nbins)])
        if np.isnan(row).sum() > nbins // 2:
            continue
        # unwrap sequentially: keep each step within +-180 deg of the previous bin
        out, prev = np.full(nbins, np.nan), None
        for k in range(nbins):
            if np.isnan(row[k]):
                continue
            out[k] = row[k] if prev is None else prev + (row[k] - prev + 180) % 360 - 180
            prev = out[k]
        rows.append(out - np.nanmean(out[:2]))
    return (edges[:-1] + edges[1:]) / 2, np.array(rows)


fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.42)

for k, (subset, ttl, tag) in enumerate([
        (allf, f"A  all fields (n = {len(allf)})", "all"),
        ([f for f, m in zip(allf, sig) if m], f"B  significant fields (n = {sig.sum()})", "sig")]):
    H, nspk = phase_density(subset)
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(H.T, origin="lower", aspect="auto", extent=[0, 1, 0, 720], cmap="magma")
    ax.set(xlabel="normalized position in field", ylabel="theta phase (deg)",
           yticks=[0, 180, 360, 540, 720], title=f"{ttl}, {nspk} spikes")
    plt.colorbar(im, ax=ax, label="P(phase | pos)", pad=.02)

ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-2, 2, 33)
ax.hist(sl, bins=bins, color="0.8", label=f"all fields (n = {len(sl)})")
ax.hist(sl[sig], bins=bins, color="tab:red", label=f"significant (n = {sig.sum()})")
ax.axvline(0, color="k", lw=.8)
ax.axvline(np.median(sl[sig]), color="tab:blue", ls="--", lw=1.2,
           label=f"median {np.median(sl[sig]):+.2f}")
ax.set(xlabel="slope (cycles per field traversal)", ylabel="# fields",
       title="C  slope distribution")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 0])
for subset, c, lab in [(allf, "0.5", "all fields"),
                       ([f for f, m in zip(allf, sig) if m], "tab:red", "significant")]:
    cx, rows = mean_phase_profile(subset)
    mu = np.nanmean(rows, 0)
    se = np.nanstd(rows, 0) / np.sqrt(np.sum(~np.isnan(rows), 0))
    ax.plot(cx, mu, "o-", color=c, ms=4, label=lab)
    ax.fill_between(cx, mu - se, mu + se, color=c, alpha=.25)
cx = np.linspace(0, 1, 50)
ax.plot(cx, 360 * np.median(sl[sig]) * cx, "k--", lw=1,
        label=f"median fitted slope ({360 * np.median(sl[sig]):+.0f}°)")
ax.axhline(0, color="k", lw=.8)
ax.set(xlabel="normalized position in field", ylabel="phase relative to field entry (deg)",
       title="D  mean phase advance across the field")
ax.legend(fontsize=7, loc="lower left")

ax = fig.add_subplot(gs[1, 1])
edges = [50, 100, 200, 400, 10 ** 5]
frac, lab = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (nsp >= lo) & (nsp < hi)
    frac.append(100 * sig[m].mean())
    lab.append(f"{lo}-{hi if hi < 10**4 else ''}\n(n={m.sum()})")
ax.bar(range(len(frac)), frac, color="tab:purple")
ax.axhline(5, color="k", ls="--", lw=.8)
ax.set(xticks=range(len(frac)), ylabel="% fields significant",
       xlabel="in-field spike count", title="E  detection is spike-count limited")
ax.set_xticklabels(lab, fontsize=7)

ax = fig.add_subplot(gs[1, 2])
names, fr, med = [], [], []
for s, r in results.items():
    p_s = np.array([f["p_shuffle"] for f in r["fields"]])
    s_s = np.array([f["slope"] for f in r["fields"]])
    names.append(f"{s}\n({len(p_s)} fields)")
    fr.append(100 * (p_s < 0.05).mean())
    med.append(360 * np.median(s_s[p_s < 0.05]))
ax.barh(range(len(fr)), fr, color="tab:green")
ax.axvline(5, color="k", ls="--", lw=.8)
ax.set(yticks=range(len(fr)), xlabel="% fields significant", title="F  per session")
ax.set_yticklabels(names, fontsize=7)
ax.invert_yaxis()
plt.savefig("figures/fig06_population.png", bbox_inches="tight")
plt.close()

# %% [markdown]
# In panel C the bars at the extreme ends come from fields with no real
# phase-position relationship: with nothing to fit, the maximum-resultant search
# runs to the edge of the ±2 cycle window. That is why the histogram is read
# together with the significance mask rather than on its own.

# %% [markdown]
# ## Result
#
# Across the five linear-track sessions the pooled spike density in the
# position-by-phase plane shows the diagonal band that is the signature of phase
# precession, and the fields that pass the individual permutation test have a median
# slope of roughly $-200^\circ$ per field traversal with the large majority of slopes
# negative. The mean within-field phase advance (panel D) is a monotonic decrease of
# about the same magnitude, so the effect is not carried by a few outliers.
#
# The fraction of fields that reach individual significance is strongly limited by
# spike count (panel E): fields with more than 400 in-field spikes reach roughly 50 %,
# whereas fields with 50–100 spikes are close to the noise floor. The sessions also
# differ, with `Achilles-10252013` (the session most often used with this dataset)
# giving the cleanest result. Both of these are properties of the recordings and of
# the statistical power available per field, not evidence against the phenomenon: the
# pooled density and the mean phase profile use every field and still show the effect
# clearly.

# %%
print("figures written:")
for fn in sorted(os.listdir("figures")):
    if fn.endswith(".png"):
        print("  figures/" + fn)
