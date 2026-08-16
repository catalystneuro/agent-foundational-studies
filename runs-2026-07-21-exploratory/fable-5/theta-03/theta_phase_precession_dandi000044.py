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
# # Theta phase entrainment and phase precession of hippocampal place cells
#
# **Data:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark & Buzsáki (2016),
# *Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*,
# Science 351:1440. Bilateral silicon-probe recordings from dorsal CA1 of freely moving rats
# running back and forth on a 1.6 m linear track for water reward, with simultaneous
# 1250 Hz local field potential from 128 channels and spike-sorted single units.
#
# ## What this notebook shows
#
# Two related but distinct phenomena are demonstrated from the same recordings:
#
# 1. **Theta phase entrainment (phase locking).** CA1 neurons do not fire at random times
#    relative to the ongoing 6–10 Hz theta rhythm. Each cell has a preferred theta phase, and
#    fires more often near that phase than elsewhere in the cycle. This is a *static* relationship
#    between spikes and the LFP: one preferred phase per cell.
#
# 2. **Theta phase precession** (O'Keefe & Recce, 1993). Within a place field, the phase at which
#    a cell fires is not fixed. On each traversal the cell begins firing at a late theta phase and
#    fires at progressively earlier phases as the animal advances through the field, so that spike
#    phase carries information about position *within* the field that the firing rate alone does not.
#    Phase precession is therefore a systematic violation of pure phase locking.
#
# The analysis pipeline is: stream the NWB file from DANDI without downloading it, reconstruct
# behaviour and select running epochs, choose an LFP channel with strong theta, extract the
# instantaneous theta phase by Hilbert transform, compute direction-specific place fields, and
# then quantify (1) phase locking with circular statistics and (2) phase precession with the
# Kempter et al. (2012) circular-linear regression plus a spike-phase shuffle control.
# Everything is finally repeated on four sessions, one from each of four rats.

# %% [markdown]
# ## 1. Setup

# %%
import warnings

import lindi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import norm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

RNG = np.random.default_rng(1)

# LINDI provides a JSON index over the remote HDF5 file so that only the byte ranges
# actually needed are fetched over HTTP; nothing is downloaded in full.
LINDI_BASE = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/{aid}/nwb.lindi.json"
# One 1.6 m linear-track session from each of the four rats in the dandiset. Of the remaining
# four assets, three (Achilles-11012013, Gatsby-08282013, Cicero-09102014) are circular-maze
# sessions, excluded because on a closed track the animal runs in one direction only and
# "entering the field from the near end" is not defined in the same way; the fourth
# (Cicero-09172014, a 2 m track) is excluded because the rat completed only 8 full rightward
# traversals, leaving unvisited spatial bins.
SESSIONS = {
    "Achilles-10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Gatsby-08022013": "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Buddy-06272013": "82714afb-724f-4e2b-b102-c9c47b5cba73",
    "Cicero-09012014": "3cc5b7b3-02e2-490a-9f19-d20670355084",
}
MAIN_SESSION = "Achilles-10252013"

LFP_FS = 1250.0            # Hz, LFP sampling rate in this dandiset
THETA_BAND = (6.0, 10.0)   # Hz
LFP_GAIN = 3.815e-7        # int16 -> volts (ElectricalSeries conversion factor)

# Behaviour / place-field parameters
MIN_SPEED = 0.15           # m/s, threshold for a running epoch
END_TRIM = 0.0625          # fraction of track length excluded at each reward end
BIN_WIDTH = 0.04           # m, spatial bin
FIELD_FRAC = 0.25          # place field = contiguous region above 25% of peak rate
MIN_PEAK_RATE = 2.0        # Hz
MIN_SPATIAL_INFO = 0.4     # bits/spike
MIN_FIELD_SPIKES = 50
FIELD_WIDTH_RANGE = (0.12, 0.70)   # m
N_SHUFFLES = 500

# %% [markdown]
# ## 2. Loading helpers
#
# Two details of this particular NWB conversion are worth stating explicitly, because both
# would silently corrupt the analysis if taken at face value:
#
# - The `SpatialSeries` for linearized position has `rate = 0.02560`. That value is the sampling
#   *period*, not the rate: `1/0.02560 = 39.06 Hz`, and 80,762 samples at 39.06 Hz spans 2067.5 s,
#   which matches the MazeEpoch duration to within one sample. We therefore treat it as a period.
# - The `electrodes` table lists `location = "unknown"` for all 128 channels, so the LFP channel
#   cannot be selected by anatomy. We pick it functionally instead, by theta/delta power ratio.

# %%
def open_session(name, cache_dir="./lindi_cache"):
    """Open a session from DANDI:000044 by streaming, with a local byte cache."""
    f = lindi.LindiH5pyFile.from_lindi_file(
        LINDI_BASE.format(aid=SESSIONS[name]),
        local_cache=lindi.LocalCache(cache_dir=cache_dir),
    )
    io = NWBHDF5IO(file=f, mode="r")
    return f, io, io.read()


def get_maze_epoch(nwbfile):
    ep = nwbfile.epochs.to_dataframe()
    row = ep[ep.label.str.contains("Maze")].iloc[0]
    return nap.IntervalSet(start=float(row.start_time), end=float(row.stop_time))


def get_position(nwbfile):
    """Linearized track position (metres) as a pynapple Tsd, NaN samples dropped."""
    beh = nwbfile.processing["behavior"]
    key = [k for k in beh.data_interfaces if "Linearized" in k][0]
    ss = list(beh[key].spatial_series.values())[0]
    y = np.asarray(ss.data[:]).ravel()
    dt = float(ss.rate)          # actually the sampling period, see note above
    t = float(ss.starting_time) + np.arange(y.size) * dt
    ok = np.isfinite(y)
    return nap.Tsd(t=t[ok], d=y[ok]), key


def track_geometry(pos, trim=END_TRIM, bin_width=BIN_WIDTH):
    """Analysis bounds and spatial bin edges, scaled to the track actually used.

    The four sessions analysed here all use a 1.6 m track, but the dandiset also contains
    2 m and circular mazes, so the reward-end exclusion and the bin count are derived from
    the observed track extent rather than hard-coded.
    """
    length = np.percentile(pos.d, 99.5)
    lo, hi = trim * length, (1 - trim) * length
    n_bins = int(round((hi - lo) / bin_width))
    edges = np.linspace(lo, hi, n_bins + 1)
    return (lo, hi), edges, 0.5 * (edges[1:] + edges[:-1])


def get_units(nwbfile):
    """Spike times as a TsGroup, tagged with the cell type from the units table."""
    df = nwbfile.units.to_dataframe()
    tsg = nap.TsGroup({int(i): nap.Ts(t=np.asarray(df.loc[i, "spike_times"])) for i in df.index})
    tsg.set_info(cell_type=np.asarray(df["cell_type"].values),
                 location=np.asarray(df["location"].values))
    return tsg


def read_lfp_channel(h5file, ch, t0, t1):
    """Stream a single LFP channel over [t0, t1] seconds and return it in volts.

    The LFP dataset is chunked as (170221, 1), i.e. one channel per chunk, so reading a
    single channel over a two-thousand-second window transfers only a few megabytes.
    """
    dset = h5file["/processing/ecephys/LFP/LFP/data"]
    i0, i1 = int(t0 * LFP_FS), int(t1 * LFP_FS)
    d = np.asarray(dset[i0:i1, ch]).astype(np.float64) * LFP_GAIN
    return nap.Tsd(t=np.arange(i0, i1) / LFP_FS, d=d)


# %% [markdown]
# ## 3. Signal processing and circular statistics
#
# Theta phase is the Hilbert phase of the 6–10 Hz band-pass filtered LFP, with 0 rad at the
# **peak** of the filtered wave and ±π at the trough. Absolute phase values depend on the depth
# of the recording site within the CA1 layers, which is not documented for this file, so the
# absolute preferred phase should not be over-interpreted; what matters here is that all cells
# and all sessions are referenced to the same, explicitly defined landmark.

# %%
def bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def theta_phase_amp(lfp_tsd, band=THETA_BAND, fs=LFP_FS):
    """Return (phase, amplitude, filtered) as pynapple Tsd objects."""
    filt = bandpass(np.asarray(lfp_tsd.values), fs, *band)
    an = hilbert(filt)
    return (nap.Tsd(t=lfp_tsd.t, d=np.angle(an)),
            nap.Tsd(t=lfp_tsd.t, d=np.abs(an)),
            nap.Tsd(t=lfp_tsd.t, d=filt))


def rayleigh(phases):
    """Rayleigh test of uniformity. Returns (mean resultant length, mean angle, p)."""
    n = len(phases)
    if n < 5:
        return np.nan, np.nan, np.nan
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S)
    z = R**2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
    return R / n, np.arctan2(S, C), p


def circmean(phases):
    return np.arctan2(np.sin(phases).mean(), np.cos(phases).mean())


SLOPE_GRID = np.linspace(-2.0, 2.0, 1601)   # cycles of theta per field traversal


def _design(x, grid=SLOPE_GRID):
    """exp(-2*pi*i*a*x) evaluated on the slope grid; reusable across phase shuffles."""
    return np.exp(-2j * np.pi * np.outer(grid, np.asarray(x, float)))


def circ_lin_regress(x, phi, W=None, grid=SLOPE_GRID):
    """Kempter et al. (2012) circular-linear regression of phase on a linear variable.

    The slope is the value of `a` maximising the mean resultant length of
    `phi - 2*pi*a*x`; `rho` is the circular-linear correlation coefficient and `p`
    its asymptotic significance. Returns (slope_rad, phase_offset_rad, rho, p).
    """
    x = np.asarray(x, float)
    phi = np.asarray(phi, float)
    if W is None:
        W = _design(x, grid)
    # NumPy 2.0's complex matmul raises spurious divide-by-zero / overflow / invalid
    # warnings here even though every operand has unit modulus. The result was checked
    # against np.einsum and against a direct cos/sin evaluation: all three agree to 1e-16
    # and select the same slope, so the floating-point flags are suppressed rather than
    # the computation being changed.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        R = np.abs(W @ np.exp(1j * phi)) / len(x)
    slope = 2 * np.pi * grid[int(np.argmax(R))]
    phi0 = circmean(phi - slope * x)

    # circular-linear correlation between phi and the slope-scaled position
    theta = np.mod(np.abs(slope) * x, 2 * np.pi)
    tbar, pbar = circmean(theta), circmean(phi)
    sp, st = np.sin(phi - pbar), np.sin(theta - tbar)
    den = np.sqrt(np.sum(sp**2) * np.sum(st**2))
    rho = np.sum(sp * st) / den if den > 0 else np.nan

    n = len(x)
    lam20, lam02, lam22 = np.mean(sp**2), np.mean(st**2), np.mean(sp**2 * st**2)
    z = rho * np.sqrt(n * lam20 * lam02 / lam22) if lam22 > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return slope, phi0, rho, p


# %% [markdown]
# ## 4. Behaviour: running epochs on the linear track
#
# Theta is a movement-related rhythm, so all analyses are restricted to epochs in which the
# animal was actually running. Two restrictions are applied:
#
# - Speed above 0.15 m/s, with rightward and leftward traversals kept separate. CA1 place fields
#   on a linear track are strongly direction-selective, so pooling the two directions would blur
#   the fields and destroy the precession signal.
# - Position restricted to the central 0.10–1.50 m of the track. At the two reward ends the animal
#   pauses to drink; theta gives way to large irregular activity and sharp-wave ripples, and the
#   low occupancy in those bins produces spurious rate peaks. Including them makes almost every
#   cell appear to have a "field" at one end of the track.

# %%
def compute_speed(pos, gap_thresh=0.1, smooth_sd=0.2):
    """Signed velocity (m/s), with samples spanning a tracking dropout discarded."""
    t, d = pos.t, pos.d
    dt = np.diff(t)
    v = np.diff(d) / dt
    v[dt > gap_thresh] = np.nan
    tm = 0.5 * (t[1:] + t[:-1])
    ok = np.isfinite(v)
    return nap.Tsd(t=tm[ok], d=v[ok]).smooth(smooth_sd)


def run_epochs(pos, vel, pos_range, min_speed=MIN_SPEED, min_dur=0.5, min_cover=0.6):
    """Direction-separated traversals of the central track."""
    inzone = (pos.threshold(pos_range[0], "above")
                 .threshold(pos_range[1], "below").time_support.drop_short_intervals(0.2))
    min_ptp = min_cover * (pos_range[1] - pos_range[0])
    out = {}
    for name, sgn in (("R", 1), ("L", -1)):
        ep = vel.threshold(min_speed * sgn, "above" if sgn > 0 else "below").time_support
        ep = ep.merge_close_intervals(0.3).drop_short_intervals(min_dur)
        ep = ep.intersect(inzone).drop_short_intervals(min_dur)
        keep = [(s, e) for s, e in zip(ep.start, ep.end)
                if len(pos.restrict(nap.IntervalSet(s, e))) > 5
                and np.ptp(pos.restrict(nap.IntervalSet(s, e)).d) >= min_ptp]
        out[name] = nap.IntervalSet(start=[k[0] for k in keep], end=[k[1] for k in keep])
    return out


# %% [markdown]
# ## 5. Load the main session and inspect the raw streams
#
# Nothing is analysed before it has been plotted. The figure below shows the linearized position
# over the whole maze epoch, the derived running speed, three seconds of raw and theta-filtered
# LFP, and the extracted phase, so that the phase estimate can be checked against the wave it
# was derived from.

# %%
h5, io, nwb = open_session(MAIN_SESSION)
maze = get_maze_epoch(nwb)
pos, maze_name = get_position(nwb)
pos = pos.restrict(maze)
units = get_units(nwb)
vel = compute_speed(pos)
POS_RANGE, EDGES, CTR = track_geometry(pos)
runs = run_epochs(pos, vel, POS_RANGE)
allrun = runs["R"].union(runs["L"])

print(nwb.session_id, "| subject", nwb.subject.subject_id, "|", maze_name)
print(f"maze epoch: {maze.start[0]:.0f}-{maze.end[0]:.0f} s ({maze.tot_length()/60:.1f} min)")
print(f"analysed track span: {POS_RANGE[0]:.2f}-{POS_RANGE[1]:.2f} m in {len(EDGES)-1} bins")
print(f"units: {len(units)} "
      f"({np.sum(units.cell_type == 'excitatory')} excitatory, "
      f"{np.sum(units.cell_type == 'inhibitory')} inhibitory)")
print(f"traversals: {len(runs['R'])} rightward, {len(runs['L'])} leftward; "
      f"{allrun.tot_length():.0f} s of running")

# %% [markdown]
# ### Choosing an LFP channel
#
# All 128 electrodes are labelled `location = "unknown"`, so the theta reference channel is chosen
# functionally: for a 120 s window of running we compute the ratio of 6–10 Hz to 2–4 Hz power on
# every channel and take the maximum.

# %%
t0_scan = allrun.start[0]
t1_scan = t0_scan + 120
scan = []
for ch in tqdm(range(128), desc="theta/delta per channel"):
    l = read_lfp_channel(h5, ch, t0_scan, t1_scan)
    fr, P = welch(l.d, fs=LFP_FS, nperseg=2048)
    scan.append(P[(fr >= 6) & (fr <= 10)].mean() / P[(fr >= 2) & (fr <= 4)].mean())
scan = np.array(scan)
THETA_CH = int(np.argmax(scan))
print(f"selected channel {THETA_CH} (theta/delta = {scan[THETA_CH]:.2f})")

lfp = read_lfp_channel(h5, THETA_CH, maze.start[0], maze.end[0])
phase, amp, filt = theta_phase_amp(lfp)

# %%
zoom = nap.IntervalSet(allrun.start[0] - 5, allrun.start[0] + 75)
fig, ax = plt.subplots(4, 1, figsize=(11, 9), gridspec_kw={"hspace": 0.55})

ax[0].plot(pos.t - maze.start[0], pos.d, "k", lw=0.4)
ax[0].axhspan(POS_RANGE[0], POS_RANGE[1], color="0.9", zorder=0)
ax[0].set(ylabel="position (m)", xlabel="time from maze onset (s)",
          title=f"{MAIN_SESSION}: whole maze epoch, {len(allrun)} traversals "
                f"(grey band = analysed track span)")

pz = pos.restrict(zoom)
vz = vel.restrict(zoom)
ax[1].plot(pz.t, pz.d, "k", lw=1)
for k, c in (("R", "tab:blue"), ("L", "tab:orange")):
    for st_, en_ in zip(runs[k].start, runs[k].end):
        if en_ > zoom.start[0] and st_ < zoom.end[0]:
            ax[1].axvspan(st_, en_, color=c, alpha=0.3, lw=0)
ax[1].set(ylabel="position (m)", xlabel="time (s)",
          title="80 s detail: rightward (blue) and leftward (orange) traversals")
axv = ax[1].twinx()
vd = vz.d.copy().astype(float)
vd[np.r_[np.diff(vz.t) > 0.2, False]] = np.nan   # break the line over tracking gaps
axv.plot(vz.t, vd, "g", lw=0.8, alpha=0.8)
axv.set_ylabel("velocity (m/s)", color="g")
axv.spines["right"].set_visible(True)

w = (lfp.t > zoom.start[0] + 20) & (lfp.t < zoom.start[0] + 23)
t_lfp = lfp.t[w] - lfp.t[w][0]
ax[2].plot(t_lfp, lfp.d[w] * 1e3, color="0.45", lw=0.8, label="raw LFP")
ax[2].plot(t_lfp, filt.d[w] * 1e3, "r", lw=1.6, label="6-10 Hz")
ax[2].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
ax[2].set(ylabel="mV", xlabel="time (s)",
          title=f"3 s of LFP, channel {THETA_CH} (selected as described below)")
ax[3].plot(t_lfp, np.degrees(phase.d[w]) % 360, "b", lw=0.9)
ax[3].set(ylabel="theta phase (deg)", xlabel="time (s)", yticks=[0, 180, 360])
fig.savefig("fig01_raw_streams.png", bbox_inches="tight")

# %%
fr, P = welch(lfp.restrict(allrun).d, fs=LFP_FS, nperseg=4096)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].semilogy(fr[fr < 40], P[fr < 40], "k")
ax[0].axvspan(*THETA_BAND, color="r", alpha=0.15)
ax[0].set(xlabel="frequency (Hz)", ylabel="power (V$^2$/Hz)",
          title=f"LFP spectrum during running (ch {THETA_CH})")
pk_theta = fr[(fr > 5) & (fr < 12)][np.argmax(P[(fr > 5) & (fr < 12)])]
ax[0].annotate(f"theta, {pk_theta:.1f} Hz", xy=(pk_theta, P[(fr > 5) & (fr < 12)].max()),
               xytext=(16, P[(fr > 5) & (fr < 12)].max() * 0.6), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8))
ax[0].margins(y=0.25)
ax[1].plot(scan, ".-", color="0.3", ms=4, lw=0.6)
ax[1].plot(THETA_CH, scan[THETA_CH], "r*", ms=14)
ax[1].margins(y=0.15)
ax[1].set(xlabel="LFP channel", ylabel="theta / delta power",
          title="channel selection (the 10-channel periodicity\nis the 10 electrodes per shank)")
fig.tight_layout()
fig.savefig("fig02_lfp_spectrum.png", bbox_inches="tight")

# %% [markdown]
# The spectrum during running has a sharp peak at about 8 Hz together with its harmonic near
# 16–19 Hz, which is the expected signature of the asymmetric, sawtooth-shaped hippocampal theta
# wave. This confirms both that theta is present and that the band-pass and Hilbert steps are
# operating on a genuine oscillation rather than on filtered noise.

# %% [markdown]
# ## 6. Direction-specific place fields
#
# Firing rate maps are computed separately for rightward and leftward traversals in 4 cm bins and
# lightly smoothed. A unit is counted as having a place field in a given direction if it is
# classified as excitatory, has a peak rate of at least 2 Hz, spatial information of at least
# 0.4 bits/spike, and a contiguous above-25%-of-peak region between 12 and 70 cm wide containing
# at least 50 spikes.

# %%
def spatial_info(rate, occ):
    """Skaggs spatial information in bits per spike."""
    p = np.asarray(occ, float) / np.sum(occ)
    m = np.nansum(p * rate)
    if m <= 0:
        return np.nan
    r = rate / m
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * r * np.log2(r)
    return np.nansum(term[np.isfinite(term)])


def field_bounds(rate, frac=FIELD_FRAC, min_bins=3):
    """Contiguous supra-threshold region containing the peak, as bin indices (inclusive)."""
    pk = int(np.nanargmax(rate))
    thr = frac * rate[pk]
    i0 = i1 = pk
    while i0 > 0 and rate[i0 - 1] > thr:
        i0 -= 1
    while i1 < len(rate) - 1 and rate[i1 + 1] > thr:
        i1 += 1
    return None if (i1 - i0 + 1) < min_bins else (i0, i1)


def rate_maps(units, pos, ep, edges):
    """Smoothed rate maps (n_bins x n_units DataFrame) and occupancy in seconds.

    Bins the animal never visited in this direction have undefined rate. In the shorter
    sessions a handful of such bins occur; they are filled by linear interpolation across
    neighbouring bins before smoothing, rather than being left as NaN (which would
    propagate through the Gaussian filter and discard the whole cell).
    """
    tc = nap.compute_tuning_curves(units, pos, bins=[edges], epochs=ep, return_pandas=True)
    V = tc.values.astype(float)
    bad = ~np.isfinite(V)
    if bad.any():
        idx = np.arange(V.shape[0])
        for j in range(V.shape[1]):
            m = bad[:, j]
            if m.all():
                V[:, j] = 0.0
            elif m.any():
                V[m, j] = np.interp(idx[m], idx[~m], V[~m, j])
    tc = pd.DataFrame(gaussian_filter1d(V, 1.0, axis=0, mode="nearest"),
                      index=tc.index, columns=tc.columns)
    occ = np.histogram(pos.restrict(ep).d, edges)[0] * np.median(np.diff(pos.t))
    return tc, occ


tcs, occs = {}, {}
for k, ep in runs.items():
    tcs[k], occs[k] = rate_maps(units, pos, ep, EDGES)

# %%
exc = np.asarray(units.cell_type) == "excitatory"
fig, axs = plt.subplots(1, 3, figsize=(11.5, 5.0),
                        gridspec_kw={"width_ratios": [1, 1, 0.5]})
for ax, k in zip(axs[:2], ["R", "L"]):
    M = tcs[k].values.T[exc]
    pk = M.max(1)
    M = M[pk >= MIN_PEAK_RATE] / pk[pk >= MIN_PEAK_RATE][:, None]
    order = np.argsort(M.argmax(1))
    im = ax.imshow(M[order], aspect="auto", origin="lower", vmin=0, vmax=1,
                   extent=[POS_RANGE[0], POS_RANGE[1], 0, M.shape[0]], cmap="magma")
    ax.set(xlabel="position (m)",
           title=f"{'rightward' if k == 'R' else 'leftward'} runs (n = {M.shape[0]})")
axs[0].set_ylabel("pyramidal cell, sorted by field peak")
fig.colorbar(im, ax=axs[1], label="normalized rate", fraction=0.05)
for k, c in (("R", "tab:blue"), ("L", "tab:orange")):
    axs[2].plot(CTR, occs[k], c, label="rightward" if k == "R" else "leftward")
axs[2].set(xlabel="position (m)", ylabel="occupancy (s)", title="occupancy")
axs[2].legend(frameon=False)
fig.tight_layout()
fig.suptitle(f"Direction-specific place fields, dorsal CA1, {MAIN_SESSION}", y=1.02)
fig.savefig("fig03_place_fields.png", bbox_inches="tight")

# %% [markdown]
# Place fields tile the track in both directions of travel, and the same population produces two
# largely independent maps for the two running directions, which is why the precession analysis
# below is carried out per direction rather than on pooled traversals.

# %% [markdown]
# ## 7. Theta phase entrainment
#
# For every unit with at least 50 spikes during running we take the theta phase at each spike
# time (the phase of the nearest LFP sample; at 1250 Hz that is at most 2.3° of theta), and test
# the resulting phase distribution for uniformity with the Rayleigh test. The mean resultant
# length (MRL) measures how tightly the cell is locked, from 0 (uniform) to 1 (all spikes at one
# phase).

# %%
def cell_types(units):
    """unit id -> cell type, as a plain dict (TsGroup metadata is label-indexed)."""
    return {int(i): c for i, c in zip(units.index, np.asarray(units.cell_type))}


def spike_phases(units, phase, ep, min_spikes=50):
    out = {}
    for uid in units.index:
        st = units[uid].restrict(ep)
        if len(st) >= min_spikes:
            out[int(uid)] = st.value_from(phase).d
    return out


sp_phase = spike_phases(units, phase, allrun)
CTYPE = cell_types(units)
rows = []
for uid, p in sp_phase.items():
    mrl, mu, pv = rayleigh(p)
    rows.append(dict(unit=uid, cell_type=CTYPE[uid],
                     n_spikes=len(p), mrl=mrl, pref_phase=np.degrees(mu) % 360,
                     rayleigh_p=pv, rate=len(p) / allrun.tot_length()))
ent = pd.DataFrame(rows).set_index("unit")
ent["entrained"] = ent.rayleigh_p < 0.01
ent.to_csv("results_entrainment.csv")

for ct, g in ent.groupby("cell_type"):
    print(f"{ct:11s} n={len(g):3d}  entrained (Rayleigh p<0.01): "
          f"{g.entrained.sum():3d} ({100*g.entrained.mean():.0f}%)   "
          f"median MRL {g.mrl.median():.3f}")

# %%
PHASE_EDGES = np.linspace(-np.pi, np.pi, 25)
PHASE_CTR = np.degrees(0.5 * (PHASE_EDGES[1:] + PHASE_EDGES[:-1])) % 360


def phase_hist(p):
    """Spike counts in 24 phase bins, duplicated over two cycles for plotting."""
    h = np.histogram(p, PHASE_EDGES)[0].astype(float)
    h = h / h.sum()
    order = np.argsort(PHASE_CTR)
    return np.r_[PHASE_CTR[order], PHASE_CTR[order] + 360], np.r_[h[order], h[order]]


# examples must be well sampled, or the phase histogram is dominated by counting noise
well = ent[ent.n_spikes >= 250]
ex_pyr = well[well.cell_type == "excitatory"].sort_values("mrl", ascending=False).index[:2]
ex_int = well[well.cell_type == "inhibitory"].sort_values("mrl", ascending=False).index[:2]
examples = list(ex_pyr) + list(ex_int)

fig, axs = plt.subplots(2, 4, figsize=(12, 5.6),
                        subplot_kw=dict(), gridspec_kw={"hspace": 0.55})
for j, uid in enumerate(examples):
    p = sp_phase[uid]
    x, h = phase_hist(p)
    a = axs[0, j]
    a.bar(x, h, width=15, color="tab:blue" if j < 2 else "tab:red", align="center")
    xx = np.linspace(0, 720, 400)
    a.plot(xx, 0.5 * h.max() * (1 + np.cos(np.radians(xx))) + h.max() * 0.02,
           color="0.4", lw=1)
    a.set(xlim=(0, 720), xticks=[0, 360, 720], xlabel="theta phase (deg)")
    a.set_title(f"unit {uid} ({ent.loc[uid,'cell_type'][:3]})\n"
                f"MRL = {ent.loc[uid,'mrl']:.2f}, n = {len(p)}", fontsize=8)
    if j == 0:
        a.set_ylabel("spike probability")

for j, uid in enumerate(examples):
    axs[1, j].remove()
    a = fig.add_subplot(2, 4, 5 + j, projection="polar")
    p = sp_phase[uid]
    h = np.histogram(p, PHASE_EDGES)[0].astype(float)
    h /= h.sum()
    a.bar(0.5 * (PHASE_EDGES[1:] + PHASE_EDGES[:-1]), h, width=2 * np.pi / 24,
          color="tab:blue" if j < 2 else "tab:red", alpha=0.85)
    mrl, mu, _ = rayleigh(p)
    a.annotate("", xy=(mu, mrl * h.max() * 3), xytext=(0, 0),
               arrowprops=dict(color="k", width=1.5, headwidth=7))
    a.set_yticklabels([])
    a.set_xticks(np.radians([0, 90, 180, 270]))
fig.suptitle("Theta phase entrainment: example CA1 units (grey curve = theta cycle, "
             "0 deg = LFP peak)", y=1.01)
fig.tight_layout()
fig.savefig("fig04_entrainment_examples.png", bbox_inches="tight")

# %%
fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.3))
bins = np.linspace(0, 0.7, 25)
for ct, c in (("excitatory", "tab:blue"), ("inhibitory", "tab:red")):
    axs[0].hist(ent[ent.cell_type == ct].mrl, bins, alpha=0.65, color=c, label=ct)
axs[0].set(xlabel="mean resultant length", ylabel="units", title="strength of phase locking")
axs[0].legend(frameon=False, fontsize=8)

axs[1].remove()
a = fig.add_subplot(1, 4, 2, projection="polar")
for ct, c in (("excitatory", "tab:blue"), ("inhibitory", "tab:red")):
    g = ent[(ent.cell_type == ct) & ent.entrained]
    a.plot(np.radians(g.pref_phase), g.mrl, "o", color=c, ms=4, alpha=0.7)
a.set_title("preferred phase vs MRL\n(entrained units)", fontsize=9, pad=20)
a.set_xticks(np.radians([0, 90, 180, 270]))
a.set_rlabel_position(45)
a.tick_params(labelsize=7)

for ct, c in (("excitatory", "tab:blue"), ("inhibitory", "tab:red")):
    g = ent[ent.cell_type == ct]
    axs[2].semilogx(g.rate, g.mrl, "o", color=c, ms=4, alpha=0.7, label=ct)
axs[2].set(xlabel="firing rate during running (Hz)", ylabel="MRL", title="MRL vs firing rate")

frac = ent.groupby("cell_type").entrained.mean() * 100
axs[3].bar(range(len(frac)), frac.values, color=["tab:blue", "tab:red"])
axs[3].set_xticks(range(len(frac)))
axs[3].set_xticklabels(frac.index, fontsize=8)
axs[3].set(ylabel="% entrained (p < 0.01)", ylim=(0, 105), title="phase-locked units")
for i, v in enumerate(frac.values):
    axs[3].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
fig.suptitle(f"Theta phase entrainment across the population, {MAIN_SESSION}", y=1.03)
fig.tight_layout()
fig.savefig("fig05_entrainment_population.png", bbox_inches="tight")

# %% [markdown]
# Both cell classes are entrained to theta, and interneurons are locked considerably more
# strongly than pyramidal cells, which is the standard result: fast-spiking CA1 interneurons fire
# on most theta cycles at a consistent phase, whereas a pyramidal cell fires on only a small
# fraction of cycles and, as the next section shows, at a phase that itself moves systematically
# with position.

# %% [markdown]
# ## 8. Theta phase precession
#
# For each cell and each running direction with a valid place field we take the spikes emitted
# inside the field, and express position as the fraction of the field traversed, running from 0
# at field entry to 1 at field exit **in the animal's direction of travel**. This last point is
# essential: on leftward runs the animal enters the field at the higher track coordinate, so an
# analysis carried out in absolute track coordinates finds precession with the opposite sign for
# the two directions and the two cancel out in the population average.
#
# Spike phase is then regressed on normalized in-field position with the circular-linear method of
# Kempter et al. (2012). Significance is assessed both by the asymptotic test on the
# circular-linear correlation and by a shuffle control in which spike phases are randomly permuted
# across the spikes of that field 500 times, which preserves both the phase distribution and the
# position distribution while destroying their pairing.

# %%
def analyse_precession(units, pos, phase, runs, tcs, occs, edges, session="",
                       n_shuffles=N_SHUFFLES, rng=RNG):
    rows, data = [], {}
    reject = dict(inhibitory=0, low_peak_rate=0, low_spatial_info=0, no_field=0,
                  field_width=0, too_few_spikes=0, too_few_in_field=0, accepted=0)
    is_exc = np.asarray(units.cell_type) == "excitatory"
    for k, ep in runs.items():
        posep = pos.restrict(ep)
        for i, uid in enumerate(units.index):
            if not is_exc[i]:
                reject["inhibitory"] += 1
                continue
            r = tcs[k][uid].values
            if not np.isfinite(r).all() or np.nanmax(r) < MIN_PEAK_RATE:
                reject["low_peak_rate"] += 1
                continue
            si = spatial_info(r, occs[k])
            if not np.isfinite(si) or si < MIN_SPATIAL_INFO:
                reject["low_spatial_info"] += 1
                continue
            fb = field_bounds(r)
            if fb is None:
                reject["no_field"] += 1
                continue
            x0, x1 = edges[fb[0]], edges[fb[1] + 1]
            if not (FIELD_WIDTH_RANGE[0] <= x1 - x0 <= FIELD_WIDTH_RANGE[1]):
                reject["field_width"] += 1
                continue
            st = units[uid].restrict(ep)
            if len(st) < MIN_FIELD_SPIKES:
                reject["too_few_spikes"] += 1
                continue
            sx = np.interp(st.t, posep.t, posep.d)          # position at each spike
            inf = (sx >= x0) & (sx <= x1)
            if inf.sum() < MIN_FIELD_SPIKES:
                reject["too_few_in_field"] += 1
                continue
            # normalized position along the direction of travel: 0 = entry, 1 = exit
            xn = (sx[inf] - x0) / (x1 - x0)
            if k == "L":
                xn = 1.0 - xn
            phi = st.value_from(phase).d[inf]

            reject["accepted"] += 1
            W = _design(xn)
            slope, phi0, rho, p_asym = circ_lin_regress(xn, phi, W)
            null = np.empty(n_shuffles)
            for s in range(n_shuffles):
                null[s] = abs(circ_lin_regress(xn, rng.permutation(phi), W)[2])
            p_shuf = (np.sum(null >= abs(rho)) + 1) / (n_shuffles + 1)

            rows.append(dict(session=session, unit=int(uid), direction=k, n_spikes=int(inf.sum()),
                             peak_rate=float(np.nanmax(r)), spatial_info=si,
                             field_start=x0, field_end=x1, field_width=x1 - x0,
                             slope_deg=np.degrees(slope), phase_offset_deg=np.degrees(phi0) % 360,
                             rho=rho, p_asym=p_asym, p_shuf=p_shuf))
            data[(int(uid), k)] = dict(xn=xn, phi=phi, rate=r, slope=slope, phi0=phi0,
                                       rho=rho, x0=x0, x1=x1)
    cols = ["session", "unit", "direction", "n_spikes", "peak_rate", "spatial_info",
            "field_start", "field_end", "field_width", "slope_deg", "phase_offset_deg",
            "rho", "p_asym", "p_shuf"]
    out = pd.DataFrame(rows, columns=cols)
    out["significant"] = (out.p_shuf < 0.05) & (out.p_asym < 0.05)
    out["precessing"] = out.significant & (out.slope_deg < 0)
    out.attrs["reject"] = reject
    return out, data


prec, prec_data = analyse_precession(units, pos, phase, runs, tcs, occs, EDGES,
                                     session=MAIN_SESSION)
prec.to_csv("results_precession.csv", index=False)
print("unit x direction pairs by outcome:", prec.attrs["reject"])

n_sig = int(prec.significant.sum())
print(f"place fields analysed: {len(prec)} ({prec.unit.nunique()} distinct cells)")
print(f"significant circular-linear phase-position relation: {n_sig} "
      f"({100*n_sig/len(prec):.0f}%)")
print(f"  of which negative slope (precession): {int(prec.precessing.sum())}")
print(f"  of which positive slope:              {n_sig - int(prec.precessing.sum())}")
print(f"median slope over significant fields: {prec.loc[prec.significant,'slope_deg'].median():.0f} "
      "deg per field traversal")
print(f"median rho over significant fields:   {prec.loc[prec.significant,'rho'].median():.2f}")

# %% [markdown]
# ### Example fields

# %%
top = prec[prec.precessing].sort_values("rho").head(6)
fig, axs = plt.subplots(2, 6, figsize=(14.5, 5.2), sharex=True,
                        gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.12})
for j, (_, row) in enumerate(top.iterrows()):
    d = prec_data[(row.unit, row.direction)]
    xf = (CTR - d["x0"]) / (d["x1"] - d["x0"])
    if row.direction == "L":
        xf = 1 - xf
    o = np.argsort(xf)
    axs[0, j].plot(xf[o], d["rate"][o], "k", lw=1.2)
    axs[0, j].axvspan(0, 1, color="0.9", zorder=0)
    axs[0, j].set(xlim=(-0.6, 1.6), ylim=(0, None))
    axs[0, j].set_title(f"unit {row.unit} {row.direction}\n"
                        f"{row.slope_deg:.0f}$\\degree$/field, "
                        f"$\\rho$ = {row.rho:.2f}", fontsize=8)

    ph_deg = np.degrees(d["phi"]) % 360
    axs[1, j].plot(np.r_[d["xn"], d["xn"]], np.r_[ph_deg, ph_deg + 360], "o",
                   ms=2.5, color="tab:blue", alpha=0.55)
    xx = np.linspace(0, 1, 50)
    yy = np.degrees(d["phi0"] + d["slope"] * xx) % 360
    for off in (-360, 0, 360, 720):
        axs[1, j].plot(xx, yy + off, "r", lw=1.8)
    axs[1, j].set(ylim=(0, 720), yticks=[0, 360, 720], xlim=(-0.6, 1.6),
                  xlabel="position in field")
axs[0, 0].set_ylabel("rate (Hz)")
axs[1, 0].set_ylabel("theta phase (deg)")
fig.suptitle("Theta phase precession in single place fields "
             "(0 = field entry, 1 = field exit, in the direction of travel)", y=0.99)
fig.tight_layout()
fig.savefig("fig06_precession_examples.png")

# %% [markdown]
# ### Population summary

# %%
fig = plt.figure(figsize=(13.5, 6.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
b = np.linspace(-720, 720, 33)
ax.hist(prec.slope_deg, b, color="0.75", label="all fields")
ax.hist(prec.loc[prec.significant, "slope_deg"], b, color="tab:blue", label="significant")
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="slope (deg per field traversal)", ylabel="fields", title="phase-position slope")
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[0, 1])
b = np.linspace(-0.8, 0.8, 33)
ax.hist(prec.rho, b, color="0.75")
ax.hist(prec.loc[prec.significant, "rho"], b, color="tab:blue")
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="circular-linear correlation $\\rho$", ylabel="fields",
       title="phase-position correlation")

ax = fig.add_subplot(gs[0, 2])
ax.plot(prec.field_width * 100, prec.slope_deg, "o", ms=4, color="0.7")
s = prec[prec.significant]
ax.plot(s.field_width * 100, s.slope_deg, "o", ms=4, color="tab:blue")
ax.axhline(0, color="k", lw=1)
ax.set(xlabel="field width (cm)", ylabel="slope (deg/field)",
       title="slope vs field size")

# pooled spike phase vs normalized in-field position, over all precessing fields
X = np.concatenate([prec_data[(r.unit, r.direction)]["xn"] for _, r in prec[prec.precessing].iterrows()])
PH = np.concatenate([prec_data[(r.unit, r.direction)]["phi"] for _, r in prec[prec.precessing].iterrows()])
ax = fig.add_subplot(gs[1, :2])
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[np.degrees(PH) % 360, (np.degrees(PH) % 360) + 360],
                           bins=[np.linspace(0, 1, 26), np.linspace(0, 720, 49)])
H = gaussian_filter1d(gaussian_filter1d(H, 0.8, axis=0, mode="nearest"),
                      1.2, axis=1, mode="wrap")
H = H / H.sum(1, keepdims=True)
ax.imshow(H.T, origin="lower", aspect="auto", extent=[0, 1, 0, 720], cmap="magma")
nb = 10
be = np.linspace(0, 1, nb + 1)
idx = np.digitize(X, be) - 1
mu = np.array([np.degrees(circmean(PH[idx == i])) % 360 for i in range(nb)])
mu_un = np.unwrap(np.radians(mu))
mu_un = np.degrees(mu_un - mu_un[0]) + mu[0]
ax.plot(0.5 * (be[1:] + be[:-1]), mu_un, "o-", color="w", ms=5, lw=2)
ax.plot(0.5 * (be[1:] + be[:-1]), mu_un + 360, "o-", color="w", ms=5, lw=2)
ax.set(xlabel="normalized position in field (0 = entry, 1 = exit)",
       ylabel="theta phase (deg)", yticks=[0, 180, 360, 540, 720],
       title=f"pooled spike phase vs in-field position, {int(prec.precessing.sum())} precessing "
             f"fields, {len(X)} spikes")

ax = fig.add_subplot(gs[1, 2])
first, last = X < 1 / 3, X > 2 / 3
be_ph = np.linspace(0, 360, 25)
for m, c, lab in ((first, "tab:green", "first third of field"),
                  (last, "tab:purple", "last third of field")):
    h = np.histogram(np.degrees(PH[m]) % 360, be_ph)[0].astype(float)
    ax.step(np.r_[be_ph[:-1], be_ph[:-1] + 360], np.r_[h, h] / h.sum(), c, where="post",
            lw=1.6, label=lab)
mu_first = np.degrees(circmean(PH[first])) % 360
mu_last = np.degrees(circmean(PH[last])) % 360
ax.set(xlabel="theta phase (deg)", ylabel="fraction of spikes", xticks=[0, 360, 720],
       title=f"entry vs exit of the field\n(mean phase {mu_first:.0f}$\\degree$ "
             f"-> {mu_last:.0f}$\\degree$)")
ax.legend(frameon=False, fontsize=7.5)
fig.savefig("fig07_precession_population.png", bbox_inches="tight")

# %% [markdown]
# ## 9. Precession on single traversals
#
# The pooled analysis averages over many passes through a field. The figure below shows that the
# effect is present pass by pass rather than being an artefact of averaging: for one strongly
# precessing field, the spikes of each individual traversal are plotted in a separate colour, and
# within nearly every pass the phase falls as the animal moves through the field.
#
# A single pass yields only a handful of spikes, so its individual slope estimate is noisy: the
# counts piled up at the two ends of the histogram are passes for which the fit ran to the edge of
# the search range rather than genuine +-540 deg/field slopes. The informative quantity is the
# asymmetry of the distribution about zero, not its tails.

# %%
best = prec[prec.precessing].sort_values("rho").iloc[0]
d = prec_data[(best.unit, best.direction)]
ep = runs[best.direction]
st = units[int(best.unit)].restrict(ep)
sx = np.interp(st.t, pos.restrict(ep).t, pos.restrict(ep).d)
inf = (sx >= d["x0"]) & (sx <= d["x1"])
xn = (sx[inf] - d["x0"]) / (d["x1"] - d["x0"])
if best.direction == "L":
    xn = 1 - xn
ph_s = np.degrees(st.value_from(phase).d[inf]) % 360
t_s = st.t[inf]
pass_id = np.searchsorted(ep.start, t_s) - 1

PASS_GRID = np.linspace(-1.5, 1.5, 1201)   # cycles/field; a single pass cannot be resolved
                                          # over the wider grid used for whole fields
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
cmap = plt.get_cmap("turbo")
shown = [pid for pid in np.unique(pass_id) if (pass_id == pid).sum() >= 8][:8]
for n, pid in enumerate(shown):
    m = pass_id == pid
    c = cmap(n / max(len(shown) - 1, 1))
    for off in (0, 360):
        axs[0].plot(xn[m], ph_s[m] + off, "o", ms=6, color=c, alpha=0.9)
    sl, p0, _, _ = circ_lin_regress(xn[m], np.radians(ph_s[m]), grid=PASS_GRID)
    xx = np.linspace(xn[m].min(), xn[m].max(), 20)
    yy = np.degrees(p0 + sl * xx) % 360
    for off in (-360, 0, 360, 720):
        axs[0].plot(xx, yy + off, color=c, lw=1.2, alpha=0.8)
axs[0].set(xlabel="normalized position in field", ylabel="theta phase (deg)",
           ylim=(0, 720), yticks=[0, 360, 720], xlim=(-0.05, 1.05),
           title=f"unit {int(best.unit)}, {best.direction} runs: {len(shown)} single traversals\n"
                 "(one colour per pass, line = per-pass circular-linear fit)")

# per-pass slope distribution across all precessing fields
pass_slopes = []
for _, row in prec[prec.precessing].iterrows():
    dd = prec_data[(row.unit, row.direction)]
    ep2 = runs[row.direction]
    st2 = units[int(row.unit)].restrict(ep2)
    sx2 = np.interp(st2.t, pos.restrict(ep2).t, pos.restrict(ep2).d)
    m2 = (sx2 >= dd["x0"]) & (sx2 <= dd["x1"])
    xn2 = (sx2[m2] - dd["x0"]) / (dd["x1"] - dd["x0"])
    if row.direction == "L":
        xn2 = 1 - xn2
    ph2 = st2.value_from(phase).d[m2]
    pid2 = np.searchsorted(ep2.start, st2.t[m2]) - 1
    for p in np.unique(pid2):
        mm = pid2 == p
        if mm.sum() >= 8:
            pass_slopes.append(np.degrees(circ_lin_regress(xn2[mm], ph2[mm], grid=PASS_GRID)[0]))
pass_slopes = np.array(pass_slopes)
axs[1].hist(pass_slopes, np.linspace(-540, 540, 37), color="tab:blue")
axs[1].axvline(0, color="k")
axs[1].axvline(np.median(pass_slopes), color="r", ls="--",
               label=f"median {np.median(pass_slopes):.0f}$\\degree$")
axs[1].set(xlabel="slope on a single traversal (deg/field)", ylabel="traversals",
           title=f"single-pass slopes, all precessing fields (n = {len(pass_slopes)})")
axs[1].legend(frameon=False)
neg = 100 * np.mean(pass_slopes < 0)
print(f"single-pass slopes negative on {neg:.0f}% of {len(pass_slopes)} traversals")
fig.tight_layout()
fig.savefig("fig08_single_pass.png")

# %% [markdown]
# ## 10. A GLM view: position and phase are not separable
#
# Phase precession implies that firing rate is a joint function of position and theta phase that
# does not factor into a position term times a phase term: at the start of the field the cell
# prefers a late phase and at the end an early one. We can test that directly by fitting Poisson
# GLMs to binned spike counts with NeMoS and comparing, by cross-validated log-likelihood:
#
# - **position only** — a B-spline basis over position (a conventional place field),
# - **position + phase** — additive, which allows a place field *and* a fixed preferred phase,
#   that is, phase locking without precession,
# - **position × phase** — the outer product of the two bases, which can represent a preferred
#   phase that changes with position.
#
# If precession is real, only the third model should improve on the second.

# %%
import nemos as nmo
from sklearn.model_selection import KFold

BIN = 0.02  # s
pos_basis = nmo.basis.BSplineEval(n_basis_funcs=10, label="position")
phs_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=6, label="phase")

glm_rows, glm_maps = [], {}
cells = prec[prec.precessing].sort_values("rho").head(20)
for _, row in tqdm(list(cells.iterrows()), desc="GLM fits"):
    ep = runs[row.direction]
    cnt = units[int(row.unit)].count(BIN, ep=ep)
    p_b = pos.interpolate(cnt, ep=ep)
    ph_b = nap.Tsd(t=cnt.t, d=np.angle(np.exp(1j * np.interp(cnt.t, phase.t, np.unwrap(phase.d)))))
    y = np.asarray(cnt.values, float)
    good = np.isfinite(p_b.values) & np.isfinite(ph_b.values)
    pv, phv, y = np.asarray(p_b.values)[good], np.asarray(ph_b.values)[good], y[good]

    X = {"position": pos_basis.compute_features(pv),
         "position + phase": (pos_basis + phs_basis).compute_features(pv, phv),
         "position x phase": (pos_basis * phs_basis).compute_features(pv, phv)}
    scores = {}
    kf = KFold(n_splits=5, shuffle=False)
    for name, Xm in X.items():
        s = []
        for tr, te in kf.split(Xm):
            m = nmo.glm.GLM(solver_name="LBFGS",
                            regularizer="Ridge", regularizer_strength=1e-4).fit(Xm[tr], y[tr])
            s.append(m.score(Xm[te], y[te], score_type="log-likelihood"))
        scores[name] = float(np.mean(s))
    glm_rows.append(dict(unit=int(row.unit), direction=row.direction, **scores))

    m = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                    regularizer_strength=1e-4).fit(X["position x phase"], y)
    gp = np.linspace(pv.min(), pv.max(), 60)
    gh = np.linspace(-np.pi, np.pi, 48)
    PPg, HHg = np.meshgrid(gp, gh, indexing="ij")
    Xg = (pos_basis * phs_basis).compute_features(PPg.ravel(), HHg.ravel())
    glm_maps[(int(row.unit), row.direction)] = (gp, gh,
                                                np.asarray(m.predict(Xg)).reshape(PPg.shape) / BIN)

glm = pd.DataFrame(glm_rows)
glm["gain_phase"] = glm["position + phase"] - glm["position"]
glm["gain_interaction"] = glm["position x phase"] - glm["position + phase"]
glm.to_csv("results_glm.csv", index=False)
print(glm[["position", "position + phase", "position x phase"]].mean())
print(f"\nadding an additive phase term improves cross-validated log-likelihood for "
      f"{int((glm.gain_phase > 0).sum())}/{len(glm)} fields")
print(f"adding the position x phase interaction improves it further for "
      f"{int((glm.gain_interaction > 0).sum())}/{len(glm)} fields")

# %%
fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(1, 4, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [glm.gain_phase.values, glm.gain_interaction.values], "-", color="0.8", lw=0.8)
ax.plot(np.zeros(len(glm)), glm.gain_phase, "o", color="tab:orange", ms=5)
ax.plot(np.ones(len(glm)), glm.gain_interaction, "o", color="tab:blue", ms=5)
ax.axhline(0, color="k", lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["+ phase\n(additive)", "+ position\n$\\times$ phase"], fontsize=8)
ax.set(ylabel="gain in CV log-likelihood", title="model comparison (n = %d fields)" % len(glm))

for j, key in enumerate(list(glm_maps)[:3]):
    gp, gh, R = glm_maps[key]
    ax = fig.add_subplot(gs[0, j + 1])
    im = ax.imshow(np.r_[R.T, R.T], origin="lower", aspect="auto",
                   extent=[gp[0], gp[-1], 0, 720], cmap="magma")
    ax.set(xlabel="position (m)", ylabel="theta phase (deg)" if j == 0 else "",
           yticks=[0, 360, 720], title=f"unit {key[0]} {key[1]}: GLM rate map", )
    fig.colorbar(im, ax=ax, label="Hz" if j == 2 else "", fraction=0.05)
fig.suptitle("Poisson GLM: firing rate as a joint function of position and theta phase", y=1.02)
fig.savefig("fig09_glm.png", bbox_inches="tight")

# %% [markdown]
# The fitted position × phase rate maps show a diagonal ridge rather than a horizontal band: as
# the animal advances through the field the preferred phase slides downward. A horizontal ridge
# would be pure phase locking; the tilt is precession.

# %% [markdown]
# ## 11. Replication across sessions and animals
#
# The whole pipeline is now run on four sessions from three rats, with the theta channel selected
# independently in each.

# %%
def run_session(name):
    h5s, ios, nwbs = open_session(name)
    mz = get_maze_epoch(nwbs)
    ps, mname = get_position(nwbs)
    ps = ps.restrict(mz)
    us = get_units(nwbs)
    vs = compute_speed(ps)
    prange, edg, _ = track_geometry(ps)
    rn = run_epochs(ps, vs, prange)
    ar = rn["R"].union(rn["L"])
    sc = []
    for ch in range(0, 128, 2):        # every other channel, for speed
        l = read_lfp_channel(h5s, ch, ar.start[0], ar.start[0] + 90)
        f_, P_ = welch(l.d, fs=LFP_FS, nperseg=2048)
        sc.append(P_[(f_ >= 6) & (f_ <= 10)].mean() / P_[(f_ >= 2) & (f_ <= 4)].mean())
    ch_best = int(np.arange(0, 128, 2)[int(np.argmax(sc))])
    lf = read_lfp_channel(h5s, ch_best, mz.start[0], mz.end[0])
    phs = theta_phase_amp(lf)[0]
    tc_, oc_ = {}, {}
    for k, ep in rn.items():
        tc_[k], oc_[k] = rate_maps(us, ps, ep, edg)
    spp = spike_phases(us, phs, ar)
    ctp = cell_types(us)
    er = []
    for uid, p in spp.items():
        mrl, mu, pv_ = rayleigh(p)
        er.append(dict(session=name, unit=uid, cell_type=ctp[uid],
                       mrl=mrl, pref_phase=np.degrees(mu) % 360, rayleigh_p=pv_))
    er = pd.DataFrame(er)
    er["entrained"] = er.rayleigh_p < 0.01
    pr, _ = analyse_precession(us, ps, phs, rn, tc_, oc_, edg, session=name, n_shuffles=200)
    print(f"  {name}: {pr.attrs['reject']}", flush=True)
    ios.close()
    return er, pr, dict(session=name, subject=name.split("-")[0], maze=mname,
                        track_m=round(prange[1] / (1 - END_TRIM), 2), theta_channel=ch_best,
                        n_units=int(len(us)), n_traversals=len(ar),
                        run_time_s=round(ar.tot_length()), n_fields=len(pr),
                        n_precessing=int(pr.precessing.sum()))


ent_all, prec_all, meta = [], [], []
for name in tqdm(list(SESSIONS), desc="sessions"):
    e_, p_, m_ = run_session(name)
    ent_all.append(e_)
    prec_all.append(p_)
    meta.append(m_)
ent_all = pd.concat(ent_all, ignore_index=True)
prec_all = pd.concat(prec_all, ignore_index=True)
meta = pd.DataFrame(meta)
meta.to_csv("results_sessions.csv", index=False)
ent_all.to_csv("results_entrainment_all_sessions.csv", index=False)
prec_all.to_csv("results_precession_all_sessions.csv", index=False)
print(meta.to_string(index=False))

# %%
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
order = list(SESSIONS)
w = 0.38
for i, (ct, c) in enumerate((("excitatory", "tab:blue"), ("inhibitory", "tab:red"))):
    v = [ent_all[(ent_all.session == s) & (ent_all.cell_type == ct)].mrl.median() for s in order]
    axs[0].bar(np.arange(4) + (i - 0.5) * w, v, w, color=c, label=ct)
axs[0].set(ylabel="median MRL", title="phase locking by session")
axs[0].legend(frameon=False, fontsize=8)

for i, (ct, c) in enumerate((("excitatory", "tab:blue"), ("inhibitory", "tab:red"))):
    v = [100 * ent_all[(ent_all.session == s) & (ent_all.cell_type == ct)].entrained.mean()
         for s in order]
    axs[1].bar(np.arange(4) + (i - 0.5) * w, v, w, color=c)
axs[1].set(ylabel="% entrained (p < 0.01)", ylim=(0, 105), title="entrained units by session")

v_sig = [100 * prec_all[prec_all.session == s].significant.mean() for s in order]
v_pre = [100 * prec_all[prec_all.session == s].precessing.mean() for s in order]
axs[2].bar(np.arange(4) - 0.5 * w, v_sig, w, color="0.6", label="significant")
axs[2].bar(np.arange(4) + 0.5 * w, v_pre, w, color="tab:blue", label="negative slope")
axs[2].set(ylabel="% of place fields", title="phase-position relation", ylim=(0, 100))
axs[2].legend(frameon=False, fontsize=8)

for s in order:
    g = prec_all[(prec_all.session == s) & prec_all.significant]
    axs[3].hist(g.slope_deg, np.linspace(-720, 360, 25), histtype="step", lw=1.6,
                label=f"{s.split('-')[0]} (n={len(g)})")
axs[3].axvline(0, color="k")
axs[3].set(xlabel="slope (deg/field)", ylabel="fields", title="slope distributions")
axs[3].legend(frameon=False, fontsize=7, loc="upper right")
for a in axs[:3]:
    a.set_xticks(range(4))
    a.set_xticklabels([o.replace("-", "\n") for o in order], fontsize=7)
fig.suptitle("Replication across four sessions, one per rat (DANDI:000044)", y=1.03)
fig.tight_layout()
fig.savefig("fig10_multisession.png", bbox_inches="tight")

# %%
s = prec_all[prec_all.significant]
print(f"ALL SESSIONS: {len(prec_all)} place fields, 4 sessions, 4 rats")
print(f"  significant phase-position relation: {len(s)} ({100*len(s)/len(prec_all):.0f}%)")
print(f"  negative slope: {int((s.slope_deg<0).sum())} / {len(s)} "
      f"({100*(s.slope_deg<0).mean():.0f}%)")
print(f"  median slope: {s.slope_deg.median():.0f} deg per field traversal")
print(f"  median |rho|: {s.rho.abs().median():.2f}")
print(f"  entrained units: {int(ent_all.entrained.sum())}/{len(ent_all)} "
      f"({100*ent_all.entrained.mean():.0f}%)")
print(f"  median MRL, pyramidal {ent_all[ent_all.cell_type=='excitatory'].mrl.median():.3f}, "
      f"interneuron {ent_all[ent_all.cell_type=='inhibitory'].mrl.median():.3f}")

# %% [markdown]
# ## 12. Summary
#
# Both phenomena are present and robust in these recordings.
#
# **Entrainment.** During running, the majority of CA1 units fire non-uniformly with respect to
# the theta cycle. Interneurons are locked roughly twice as tightly as pyramidal cells, measured
# by mean resultant length, and essentially all of them are significantly entrained, whereas
# pyramidal cells are entrained more weakly and less uniformly. This is the expected asymmetry:
# interneurons follow the rhythm on nearly every cycle, while a pyramidal cell participates in
# only a small fraction of cycles.
#
# **Precession.** Restricting to well-defined place fields and expressing position in the animal's
# direction of travel, the great majority of fields with a statistically significant
# circular-linear phase-position relation have a *negative* slope, spanning on the order of half
# to one full theta cycle across the field. The relation holds on individual traversals, not only
# in the pooled average, and it is reproduced independently in four sessions from three animals.
#
# The direction convention is the single largest methodological trap here. Because a leftward
# traversal enters a field at the larger track coordinate, running the same regression in absolute
# track coordinates yields slopes of opposite sign for the two directions; pooling them then
# reports roughly half positive and half negative slopes and hides the effect completely.
#
# Finally, the GLM comparison makes the relationship between the two phenomena explicit. Adding an
# additive theta-phase term to a position-only model improves prediction, which is phase locking.
# Adding the position × phase interaction improves it further, which is precession: the preferred
# phase itself depends on where in the field the animal is, so the joint position-phase rate map
# is a tilted ridge rather than a horizontal band.
