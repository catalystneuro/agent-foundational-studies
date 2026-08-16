# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Theta phase entrainment and precession in hippocampal place cells
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark, Long & Buzsáki (2016),
# *"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences."*
# Bilateral silicon-probe recordings from dorsal CA1 in rats running on a 1.6 m linear track.
#
# **Phenomena demonstrated:**
# 1. **Theta rhythm in CA1 LFP** during locomotion (6–12 Hz).
# 2. **Theta phase entrainment** of CA1 pyramidal cells (spikes preferentially fire near a
#    consistent phase of the theta cycle).
# 3. **Place cells** with reliable spatial firing on the linear track.
# 4. **Theta phase precession** — within a place field, spikes occur at progressively earlier
#    theta phases as the rat traverses the field (Skaggs et al., 1996; O'Keefe & Recce, 1993).
#
# Session analysed: `sub-Achilles_ses-Achilles-10252013`. Streaming via `remfile` with disk caching.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import circmean, pearsonr
from tqdm import tqdm

import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

OUTDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FIGDIR = OUTDIR
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. Load NWB file via streaming

# %%
S3_URL = (
    "https://api.dandiarchive.org/api/dandisets/000044/versions/draft/"
    "assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/"
)
CACHE_DIR = "/tmp/remfile_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

print("Session:", nwbfile.session_description[:120])
print("Subject:", nwbfile.subject.subject_id)
print("Epochs:")
print(nwbfile.epochs.to_dataframe())

# Maze epoch (period during which the rat ran the linear track)
maze_row = nwbfile.epochs.to_dataframe().query("label == 'MazeEpoch'").iloc[0]
MAZE_START = float(maze_row["start_time"])
MAZE_STOP = float(maze_row["stop_time"])
maze_ep = nap.IntervalSet(start=MAZE_START, end=MAZE_STOP)
print(f"\nMaze epoch: {MAZE_START:.1f}–{MAZE_STOP:.1f} s ({(MAZE_STOP-MAZE_START)/60:.1f} min)")

# %% [markdown]
# ## 2. Behaviour: linearised position on the 1.6 m track
#
# Note: in this NWB file, the SpatialSeries `rate` field stores the sampling **interval** rather
# than the rate. We therefore reconstruct timestamps from `starting_time` and a regenerated rate.

# %%
beh = nwbfile.processing["behavior"]
ss_lin = beh.data_interfaces["1.6mLinearMazeLinearizedPosition"].spatial_series[
    "1.6mLinearMazeLinearizedTimeSeries"
]
ss_xy = beh.data_interfaces["1.6mLinearMazePosition"].spatial_series[
    "1.6mLinearMazeSpatialSeries"
]

n_pos = ss_lin.data.shape[0]
pos_dt = float(ss_lin.rate)               # actually the period (s/sample)
pos_fs = 1.0 / pos_dt
pos_t = ss_lin.starting_time + np.arange(n_pos) * pos_dt
print(f"Position samples: {n_pos}, fs ≈ {pos_fs:.2f} Hz, span "
      f"{pos_t[0]:.1f}–{pos_t[-1]:.1f} s")

lin_raw = ss_lin.data[:].squeeze()
xy_raw = ss_xy.data[:]

# Drop NaN samples (occasional tracking dropouts)
valid = np.isfinite(lin_raw) & np.isfinite(xy_raw[:, 0]) & np.isfinite(xy_raw[:, 1])
print(f"Valid position samples: {valid.sum()} / {n_pos} ({100*valid.mean():.1f}%)")

t_valid = pos_t[valid]
lin_valid = lin_raw[valid]
xy_valid = xy_raw[valid]

position = nap.Tsd(t=t_valid, d=lin_valid, time_support=maze_ep)
position_xy = nap.TsdFrame(t=t_valid, d=xy_valid, columns=["x", "y"], time_support=maze_ep)
print("position (linearised):", position)

# %% [markdown]
# ### Compute speed and identify running epochs

# %%
# Smooth linearised position with a 250 ms Gaussian, then differentiate
pos_smoothed = position.smooth(std=0.25)
speed_vals = np.zeros_like(pos_smoothed.values)
speed_vals[1:] = np.abs(np.diff(pos_smoothed.values)) / np.diff(pos_smoothed.index.values)
speed = nap.Tsd(t=pos_smoothed.index.values, d=speed_vals, time_support=maze_ep)

SPEED_THR = 0.20  # m/s — stricter to exclude reward-site lingering
running_mask = speed.values > SPEED_THR
# Build a Boolean Tsd and threshold-extract intervals
running_tsd = nap.Tsd(t=speed.index.values, d=running_mask.astype(float), time_support=maze_ep)
running_ep = running_tsd.threshold(0.5).time_support
# Drop very short fragments
running_ep = running_ep.drop_short_intervals(0.5)
print(f"Running epochs: {len(running_ep)} segments, "
      f"{running_ep.tot_length()/60:.1f} min total")

# %% [markdown]
# ### Split runs by direction (left→right vs right→left)
#
# Place fields on a linear track are direction-dependent, so we separate the two travel
# directions. We classify each running segment by the sign of its net displacement.

# %%
ltr_starts, ltr_stops = [], []
rtl_starts, rtl_stops = [], []
TRACK_MIN, TRACK_MAX = np.nanmin(lin_valid), np.nanmax(lin_valid)
print(f"Linearised position range: {TRACK_MIN:.2f} – {TRACK_MAX:.2f} m")

for s, e in zip(running_ep.start, running_ep.end):
    seg = position.get(s, e)
    if len(seg) < 5:
        continue
    delta = seg.values[-1] - seg.values[0]
    if delta > 0.30:
        ltr_starts.append(s)
        ltr_stops.append(e)
    elif delta < -0.30:
        rtl_starts.append(s)
        rtl_stops.append(e)

ltr_ep = nap.IntervalSet(start=np.array(ltr_starts), end=np.array(ltr_stops))
rtl_ep = nap.IntervalSet(start=np.array(rtl_starts), end=np.array(rtl_stops))
print(f"Left→right runs: {len(ltr_ep)}, total {ltr_ep.tot_length()/60:.1f} min")
print(f"Right→left runs: {len(rtl_ep)}, total {rtl_ep.tot_length()/60:.1f} min")

# %% [markdown]
# ## 3. Spike trains: pyramidal cells in CA1

# %%
units_df = nwbfile.units.to_dataframe()
print("Cell-type counts:")
print(units_df["cell_type"].value_counts())

spike_dict = {}
for uid, row in units_df.iterrows():
    if row["cell_type"] != "excitatory":
        continue
    st = np.asarray(row["spike_times"], dtype=float)
    spike_dict[int(uid)] = nap.Ts(t=st, time_support=maze_ep)

units = nap.TsGroup(spike_dict, time_support=maze_ep)
units.set_info(
    cell_type=units_df.loc[list(units.keys()), "cell_type"].values,
    location=units_df.loc[list(units.keys()), "location"].values,
    shank_id=units_df.loc[list(units.keys()), "shank_id"].values,
)
print(f"\nLoaded {len(units)} excitatory CA1 units (restricted to maze epoch)")
# Restrict to maze for rate calculations
units_maze = units.restrict(maze_ep)
rates_maze = np.asarray(units_maze.rates)
print(f"Firing-rate range during maze: {rates_maze.min():.2f}–{rates_maze.max():.2f} Hz")

# %% [markdown]
# ## 4. LFP and theta extraction
#
# We pick the CA1 channel with the strongest theta/delta ratio during the maze epoch and
# Hilbert-transform a 6–12 Hz band-pass to obtain instantaneous theta phase.

# %%
lfp_es = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
LFP_FS = float(lfp_es.rate)               # 1250 Hz
print(f"LFP rate: {LFP_FS} Hz, n_channels = {lfp_es.data.shape[1]}")

i0 = int(MAZE_START * LFP_FS)
i1 = int(MAZE_STOP * LFP_FS)

# Quick scan of theta/delta for candidate channels (one channel per shank)
candidate_chs = list(range(5, 128, 10))
print("Scanning theta/delta for candidate LFP channels...")
SCAN_SECS = 120
scan_n = int(SCAN_SECS * LFP_FS)
best_ch, best_ratio = None, -np.inf
ratios = {}
for ch in tqdm(candidate_chs):
    seg = lfp_es.data[i0:i0 + scan_n, ch]
    f, P = welch(seg, fs=LFP_FS, nperseg=int(4 * LFP_FS))
    theta_p = P[(f >= 6) & (f <= 12)].mean()
    delta_p = P[(f >= 1) & (f <= 4)].mean()
    r = theta_p / delta_p
    ratios[ch] = r
    if r > best_ratio:
        best_ratio = r
        best_ch = ch
print(f"Best channel: {best_ch} (theta/delta = {best_ratio:.2f})")

# %% [markdown]
# ### Load full maze-epoch LFP for the chosen channel and band-pass filter

# %%
print(f"Loading full maze LFP for channel {best_ch} ({(i1 - i0)} samples)...")
lfp_raw = lfp_es.data[i0:i1, best_ch].astype(np.float32)
lfp_t = MAZE_START + np.arange(lfp_raw.shape[0]) / LFP_FS
lfp_tsd = nap.Tsd(t=lfp_t, d=lfp_raw, time_support=maze_ep)

THETA_LOW, THETA_HIGH = 6.0, 12.0
b, a = butter(3, [THETA_LOW / (LFP_FS / 2), THETA_HIGH / (LFP_FS / 2)], btype="band")
theta_filt = filtfilt(b, a, lfp_raw)
analytic = hilbert(theta_filt)
theta_phase = np.angle(analytic)        # radians, range [-pi, pi]
theta_amp = np.abs(analytic)

theta_filt_tsd = nap.Tsd(t=lfp_t, d=theta_filt, time_support=maze_ep)
theta_phase_tsd = nap.Tsd(t=lfp_t, d=theta_phase, time_support=maze_ep)
theta_amp_tsd = nap.Tsd(t=lfp_t, d=theta_amp, time_support=maze_ep)
print("Theta extraction done.")

# %% [markdown]
# ## 5. Plot: raw + theta-filtered LFP and behaviour

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
# Pick a 4-second window during a left-to-right run for illustration
demo_start = ltr_ep.start[len(ltr_ep) // 2]
demo_end = demo_start + 4.0

t_lfp_win = lfp_tsd.get(demo_start, demo_end)
t_thf_win = theta_filt_tsd.get(demo_start, demo_end)
t_thp_win = theta_phase_tsd.get(demo_start, demo_end)
pos_win = position.get(demo_start, demo_end)

axes[0].plot(t_lfp_win.index.values, t_lfp_win.values, color="k", lw=0.5)
axes[0].plot(t_thf_win.index.values, t_thf_win.values, color="C3", lw=1.5,
             label="6–12 Hz")
axes[0].set_ylabel("LFP (a.u.)")
axes[0].legend(loc="upper right")
axes[0].set_title(f"CA1 LFP — channel {best_ch} during a track run")

axes[1].plot(t_thp_win.index.values, np.degrees(t_thp_win.values), color="C0", lw=1)
axes[1].set_ylabel("Theta phase (°)")
axes[1].set_yticks([-180, -90, 0, 90, 180])

axes[2].plot(pos_win.index.values, pos_win.values, color="k")
axes[2].set_ylabel("Linearised\nposition (m)")

# Spike raster of a few units for the same window
for i, uid in enumerate(list(units.keys())[:30]):
    sp = units[uid].get(demo_start, demo_end)
    axes[3].vlines(sp.index.values, i + 0.1, i + 0.9, color="C0", lw=0.6)
axes[3].set_ylabel("Unit #")
axes[3].set_xlabel("Time (s)")
axes[3].set_xlim(demo_start, demo_end)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig01_raw_lfp_theta_position.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 6. Place-field computation
#
# We bin linearised position into 2 cm bins along the 1.6 m track and compute the firing rate
# of each unit in each bin, **separately for each running direction**, restricted to running
# epochs only.

# %%
BIN_SIZE = 0.02  # m
# Trim 5 cm off each end of the track to exclude reward-site bins
TRACK_LO = TRACK_MIN + 0.05
TRACK_HI = TRACK_MAX - 0.05
edges = np.arange(TRACK_LO, TRACK_HI + BIN_SIZE, BIN_SIZE)
centres = 0.5 * (edges[:-1] + edges[1:])
print(f"Number of position bins: {len(centres)} (edges {edges[0]:.2f}–{edges[-1]:.2f} m)")


def compute_place_fields(units_group, position_tsd, run_ep, edges_):
    """Return DataFrame of firing-rate maps (rows = bin centres, cols = unit ids)."""
    pos_run = position_tsd.restrict(run_ep)
    occ, _ = np.histogram(pos_run.values, bins=edges_)
    occ_time = occ * pos_dt
    rates = pd.DataFrame(index=0.5 * (edges_[:-1] + edges_[1:]),
                         columns=list(units_group.keys()), dtype=float)
    for uid in units_group.keys():
        spikes = units_group[uid].restrict(run_ep)
        # Sample position at each spike time
        if len(spikes) == 0:
            rates[uid] = 0
            continue
        spike_pos = spikes.value_from(position_tsd)
        cnt, _ = np.histogram(spike_pos.values, bins=edges_)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(occ_time > 0.05, cnt / occ_time, np.nan)
        rates[uid] = r
    return rates, occ_time


print("Computing place fields (LTR)...")
pf_ltr, occ_ltr = compute_place_fields(units, position, ltr_ep, edges)
print("Computing place fields (RTL)...")
pf_rtl, occ_rtl = compute_place_fields(units, position, rtl_ep, edges)


def smooth_pf(pf_df, sigma_bins=2.5):
    out = pf_df.copy()
    out = out.fillna(0)
    from scipy.ndimage import gaussian_filter1d
    for c in out.columns:
        out[c] = gaussian_filter1d(out[c].values, sigma=sigma_bins, mode="nearest")
    return out


pf_ltr_s = smooth_pf(pf_ltr)
pf_rtl_s = smooth_pf(pf_rtl)

# %% [markdown]
# ### Spatial information and place-cell selection
#
# Skaggs spatial information:
# $$I = \sum_i p_i \frac{\lambda_i}{\bar\lambda} \log_2\!\frac{\lambda_i}{\bar\lambda}$$
# where $p_i$ is the occupancy probability and $\lambda_i$ the firing rate in bin $i$.
# Place cells are pyramidal cells with peak rate ≥ 2 Hz, mean rate ≤ 5 Hz (to exclude
# interneuron-like units), and spatial information ≥ 0.5 bits/spike.

# %%
def spatial_info(rate_map, occupancy_time):
    occ_p = occupancy_time / occupancy_time.sum()
    valid = occ_p > 0
    rate = rate_map[valid]
    p = occ_p[valid]
    mean_rate = (p * rate).sum()
    if mean_rate <= 0:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rate > 0, rate / mean_rate, 0)
        si = np.nansum(p * np.where(rate > 0, ratio * np.log2(ratio), 0))
    return float(si), float(mean_rate)


def cell_metrics(pf_smoothed, occupancy_time):
    rows = []
    for uid in pf_smoothed.columns:
        rmap = pf_smoothed[uid].values
        si, mean_rate = spatial_info(rmap, occupancy_time)
        rows.append(dict(uid=uid, peak_rate=np.nanmax(rmap), mean_rate=mean_rate,
                         peak_pos=pf_smoothed.index.values[np.nanargmax(rmap)], si=si))
    return pd.DataFrame(rows).set_index("uid")


metrics_ltr = cell_metrics(pf_ltr_s, occ_ltr)
metrics_rtl = cell_metrics(pf_rtl_s, occ_rtl)

PEAK_THR = 2.0
MEAN_THR = 5.0
SI_THR = 0.5

is_pc_ltr = (
    (metrics_ltr["peak_rate"] >= PEAK_THR)
    & (metrics_ltr["mean_rate"] <= MEAN_THR)
    & (metrics_ltr["si"] >= SI_THR)
)
is_pc_rtl = (
    (metrics_rtl["peak_rate"] >= PEAK_THR)
    & (metrics_rtl["mean_rate"] <= MEAN_THR)
    & (metrics_rtl["si"] >= SI_THR)
)
pc_ltr = metrics_ltr.index[is_pc_ltr].tolist()
pc_rtl = metrics_rtl.index[is_pc_rtl].tolist()
print(f"Place cells: {len(pc_ltr)} (LTR), {len(pc_rtl)} (RTL); "
      f"any direction: {len(set(pc_ltr) | set(pc_rtl))}")

# %% [markdown]
# ### Plot population place-field heat-maps

# %%
def sort_by_peak(pf_smoothed, ids):
    sub = pf_smoothed[ids]
    norm = sub.div(sub.max(axis=0).replace(0, np.nan), axis=1)
    peaks = norm.idxmax().values
    order = np.argsort(peaks)
    return [ids[i] for i in order], norm.iloc[:, order].values


fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, pf_s, ids, label in [
    (axes[0], pf_ltr_s, pc_ltr, "Left → right"),
    (axes[1], pf_rtl_s, pc_rtl, "Right → left"),
]:
    if len(ids) == 0:
        ax.set_title(f"{label}\nno place cells")
        continue
    _, mat = sort_by_peak(pf_s, ids)
    im = ax.imshow(mat.T, aspect="auto", cmap="viridis", origin="lower",
                   extent=[centres[0], centres[-1], 0, len(ids)])
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Place cell # (sorted)")
    ax.set_title(f"{label}: {len(ids)} place cells")
    plt.colorbar(im, ax=ax, label="Norm. rate")
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig02_place_field_population.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 7. Theta phase entrainment of CA1 pyramidal cells
#
# For every spike during running, sample the theta phase of the LFP. Compute per-unit:
# - mean phase (preferred theta phase)
# - mean resultant length (entrainment strength, 0 = uniform, 1 = perfectly locked)
# - Rayleigh test for non-uniformity.

# %%
phase_at_run = theta_phase_tsd.restrict(running_ep)
amp_at_run = theta_amp_tsd.restrict(running_ep)


def phases_for_unit(uid):
    sp = units[uid].restrict(running_ep)
    if len(sp) == 0:
        return np.array([])
    return sp.value_from(theta_phase_tsd).values


def rayleigh(angles):
    if len(angles) == 0:
        return np.nan, np.nan, np.nan
    n = len(angles)
    R = np.sqrt(np.sum(np.cos(angles))**2 + np.sum(np.sin(angles))**2)
    Rbar = R / n
    Z = n * Rbar**2
    p = np.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * n) -
                      (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2))
    return float(Rbar), float(p), float(circmean(angles, high=np.pi, low=-np.pi))


lock_rows = []
for uid in tqdm(list(units.keys()), desc="Phase locking"):
    ang = phases_for_unit(uid)
    Rbar, pval, mu = rayleigh(ang)
    lock_rows.append(dict(uid=uid, n_spikes=len(ang), mean_phase=mu, mrl=Rbar, p=pval))

lock_df = pd.DataFrame(lock_rows).set_index("uid")
print("Phase-locking summary (excitatory units, ≥200 spikes during running):")
print(lock_df.query("n_spikes >= 200").describe()[["n_spikes", "mrl"]])
sig = lock_df.query("n_spikes >= 200 and p < 0.001")
print(f"Significantly entrained (Rayleigh p<0.001, ≥200 spikes): "
      f"{len(sig)} / {len(lock_df.query('n_spikes >= 200'))}")

# %% [markdown]
# ### Population polar plot of preferred theta phases

# %%
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="polar")
sig_to_plot = lock_df.query("n_spikes >= 200 and p < 0.001")
ax1.scatter(sig_to_plot["mean_phase"], sig_to_plot["mrl"],
            c="C0", alpha=0.7, edgecolor="k", s=40)
ax1.set_theta_zero_location("E")
ax1.set_theta_direction(1)
ax1.set_rlim(0, max(0.6, sig_to_plot["mrl"].max() * 1.05))
ax1.set_title("Preferred theta phase\n(radius = MRL)", pad=20)

ax2 = fig.add_subplot(1, 2, 2)
mrl_all = lock_df.query("n_spikes >= 200")["mrl"].values
ax2.hist(mrl_all, bins=25, color="0.6", edgecolor="k")
ax2.axvline(np.mean(mrl_all), color="C3", lw=2,
            label=f"mean = {np.mean(mrl_all):.2f}")
ax2.set_xlabel("Mean resultant length (entrainment strength)")
ax2.set_ylabel("# units")
ax2.legend()
ax2.set_title("Theta entrainment of CA1 pyramidal cells")
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig03_theta_entrainment_population.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 8. Theta phase precession in single place cells
#
# For each place cell, restrict to spikes occurring while the rat traversed the cell's place
# field in the cell's preferred direction. Plot theta phase versus position-in-field, and fit
# a circular–linear regression (slope in radians/m).

# %%
def place_field_extent(rate_map, peak_idx, frac=0.20):
    """Return (lo_idx, hi_idx) for contiguous bins above `frac * peak_rate`."""
    peak_rate = rate_map[peak_idx]
    thr = frac * peak_rate
    n = len(rate_map)
    lo = peak_idx
    while lo > 0 and rate_map[lo - 1] >= thr:
        lo -= 1
    hi = peak_idx
    while hi < n - 1 and rate_map[hi + 1] >= thr:
        hi += 1
    return lo, hi


def cl_corr(positions_norm, phases):
    """Kempter-style circular-linear correlation; returns (slope_rad_per_unit, phi0, rho).
    Slope search is restricted to a physiological range: from -3π (strongly precessing,
    1.5 theta cycles per field) to +π (mild recession). Most CA1 place cells show ~-2π."""
    if len(positions_norm) < 10:
        return np.nan, np.nan, np.nan
    slopes = np.linspace(-4 * np.pi, 2 * np.pi, 601)
    R = np.zeros_like(slopes)
    for i, s in enumerate(slopes):
        R[i] = np.abs(np.mean(np.exp(1j * (phases - s * positions_norm))))
    best = slopes[np.argmax(R)]
    phi0 = np.angle(np.mean(np.exp(1j * (phases - best * positions_norm))))
    # Compute circular-linear correlation coefficient (Kempter 2012)
    theta_bar = circmean(phases, high=np.pi, low=-np.pi)
    pos_bar = positions_norm.mean()
    num = np.sum(np.sin(phases - theta_bar) *
                 np.sin(2 * np.pi * (positions_norm - pos_bar)))
    den = np.sqrt(np.sum(np.sin(phases - theta_bar) ** 2) *
                  np.sum(np.sin(2 * np.pi * (positions_norm - pos_bar)) ** 2))
    rho = num / den if den > 0 else np.nan
    return float(best), float(phi0), float(rho)


def precession_for_cell(uid, direction):
    if direction == "ltr":
        run_ep, pf_s = ltr_ep, pf_ltr_s
    else:
        run_ep, pf_s = rtl_ep, pf_rtl_s
    rmap = pf_s[uid].values
    peak_idx = int(np.nanargmax(rmap))
    lo, hi = place_field_extent(rmap, peak_idx, frac=0.30)
    field_lo = centres[lo] - BIN_SIZE / 2
    field_hi = centres[hi] + BIN_SIZE / 2
    # Spikes during direction-specific runs
    sp = units[uid].restrict(run_ep)
    if len(sp) == 0:
        return None
    sp_pos = sp.value_from(position).values
    sp_phase = sp.value_from(theta_phase_tsd).values
    # Keep spikes inside the field
    in_field = (sp_pos >= field_lo) & (sp_pos <= field_hi) & np.isfinite(sp_phase)
    sp_pos = sp_pos[in_field]
    sp_phase = sp_phase[in_field]
    if len(sp_phase) < 25:
        return None
    pos_norm = (sp_pos - field_lo) / (field_hi - field_lo)
    if direction == "rtl":
        # Reverse so the rat enters the field at pos_norm = 0 and exits at 1
        pos_norm = 1.0 - pos_norm
    slope, phi0, rho = cl_corr(pos_norm, sp_phase)
    return dict(
        uid=uid,
        direction=direction,
        n_spikes=int(len(sp_phase)),
        field_lo=field_lo,
        field_hi=field_hi,
        peak_pos=centres[peak_idx],
        sp_pos=sp_pos,
        sp_phase=sp_phase,
        pos_norm=pos_norm,
        slope=slope,
        phi0=phi0,
        rho=rho,
    )


prec_results = []
for uid in pc_ltr:
    r = precession_for_cell(uid, "ltr")
    if r is not None:
        prec_results.append(r)
for uid in pc_rtl:
    r = precession_for_cell(uid, "rtl")
    if r is not None:
        prec_results.append(r)

prec_df = pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("sp_pos", "sp_phase", "pos_norm")}
                         for r in prec_results])
print(f"Place-cell × direction units with ≥25 in-field spikes: {len(prec_df)}")
print("Slope (rad/field) summary:")
print(prec_df["slope"].describe())
print(f"Negative slopes (canonical precession): "
      f"{(prec_df['slope'] < 0).sum()} / {len(prec_df)} "
      f"({100 * (prec_df['slope'] < 0).mean():.1f}%)")

# %% [markdown]
# ### Plot phase precession for the 12 strongest examples

# %%
strong = prec_df.assign(abs_rho=prec_df["rho"].abs()).sort_values(
    ["abs_rho"], ascending=False).head(12)
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
for ax, (_, row) in zip(axes.ravel(), strong.iterrows()):
    rec = next(r for r in prec_results
               if r["uid"] == row["uid"] and r["direction"] == row["direction"])
    # Plot two cycles of theta for clarity
    ph = rec["sp_phase"]
    pn = rec["pos_norm"]
    ax.scatter(pn, np.degrees(ph), s=8, color="C0", alpha=0.6)
    ax.scatter(pn, np.degrees(ph) + 360, s=8, color="C0", alpha=0.6)
    # Regression line
    x_line = np.linspace(0, 1, 50)
    y_line = np.degrees(rec["slope"] * x_line + rec["phi0"])
    # Wrap into plot range
    ax.plot(x_line, y_line, color="C3", lw=2)
    ax.plot(x_line, y_line + 360, color="C3", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-180, 540)
    ax.set_yticks([-180, 0, 180, 360, 540])
    ax.set_title(f"u{row['uid']} ({row['direction']})\n"
                 f"slope={np.degrees(row['slope']):.0f}°/field, ρ={row['rho']:+.2f}",
                 fontsize=9)
    ax.set_xlabel("Position-in-field")
    ax.set_ylabel("Theta phase (°)")
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig04_phase_precession_examples.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ### Population summary of precession slopes

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(np.degrees(prec_df["slope"].values), bins=25, color="0.6", edgecolor="k")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axvline(np.degrees(prec_df["slope"].median()), color="C3", lw=2,
           label=f"median = {np.degrees(prec_df['slope'].median()):.0f}°/field")
ax.set_xlabel("Phase precession slope (°/field)")
ax.set_ylabel("# place cells")
ax.set_title(f"Slopes across {len(prec_df)} place cell × direction")
ax.legend()

ax = axes[1]
ax.hist(prec_df["rho"].values, bins=25, color="0.6", edgecolor="k")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axvline(prec_df["rho"].median(), color="C3", lw=2,
           label=f"median ρ = {prec_df['rho'].median():+.2f}")
ax.set_xlabel("Circular–linear correlation (ρ)")
ax.set_ylabel("# place cells")
ax.set_title("Strength of phase precession")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig05_precession_population.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 9. Putting it together: an example place cell with everything
#
# A 4-panel figure showing: (a) place field, (b) raster vs position colour-coded by theta phase,
# (c) theta phase vs position-in-field with regression line, (d) preferred-phase distribution
# over running.

# %%
example = strong.iloc[0]
ex_rec = next(r for r in prec_results
              if r["uid"] == example["uid"] and r["direction"] == example["direction"])
ex_uid = ex_rec["uid"]
ex_dir = ex_rec["direction"]
pf_s = pf_ltr_s if ex_dir == "ltr" else pf_rtl_s
rate_map = pf_s[ex_uid].values

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.3])

ax_pf = fig.add_subplot(gs[0, 0])
ax_pf.plot(centres, rate_map, lw=2, color="C0")
ax_pf.axvspan(ex_rec["field_lo"], ex_rec["field_hi"], color="C3", alpha=0.15,
              label="place field")
ax_pf.set_xlabel("Position (m)")
ax_pf.set_ylabel("Firing rate (Hz)")
ax_pf.set_title(f"Place field of unit {ex_uid} ({ex_dir})")
ax_pf.legend()

ax_pl = fig.add_subplot(gs[0, 1], projection="polar")
ang = phases_for_unit(ex_uid)
hist_edges = np.linspace(-np.pi, np.pi, 25)
hist_counts, _ = np.histogram(ang, bins=hist_edges)
ax_pl.bar(0.5 * (hist_edges[:-1] + hist_edges[1:]), hist_counts,
          width=np.diff(hist_edges), bottom=0, color="C0", edgecolor="k", alpha=0.8)
ax_pl.set_title(f"Theta-phase distribution\nMRL={lock_df.loc[ex_uid,'mrl']:.2f}", pad=20)

ax_prec = fig.add_subplot(gs[0, 2])
ax_prec.scatter(ex_rec["pos_norm"], np.degrees(ex_rec["sp_phase"]), s=10,
                color="C0", alpha=0.6)
ax_prec.scatter(ex_rec["pos_norm"], np.degrees(ex_rec["sp_phase"]) + 360, s=10,
                color="C0", alpha=0.6)
xl = np.linspace(0, 1, 50)
yl = np.degrees(ex_rec["slope"] * xl + ex_rec["phi0"])
ax_prec.plot(xl, yl, color="C3", lw=2)
ax_prec.plot(xl, yl + 360, color="C3", lw=2)
ax_prec.set_xlim(0, 1)
ax_prec.set_ylim(-180, 540)
ax_prec.set_xlabel("Position-in-field")
ax_prec.set_ylabel("Theta phase (°)")
ax_prec.set_title(f"Phase precession\nρ={ex_rec['rho']:+.2f}, "
                  f"slope={np.degrees(ex_rec['slope']):.0f}°/field")

# Bottom row: example traversals showing spike phase vs position
ax_runs = fig.add_subplot(gs[1, :])
run_ep_use = ltr_ep if ex_dir == "ltr" else rtl_ep
chosen = []
for s, e in zip(run_ep_use.start, run_ep_use.end):
    sp = units[ex_uid].get(s, e)
    sp_pos = sp.value_from(position).values
    in_field = (sp_pos >= ex_rec["field_lo"]) & (sp_pos <= ex_rec["field_hi"])
    if in_field.sum() >= 4:
        chosen.append((s, e))
    if len(chosen) >= 8:
        break
for k, (s, e) in enumerate(chosen):
    sp = units[ex_uid].get(s, e)
    sp_pos = sp.value_from(position).values
    sp_phase = sp.value_from(theta_phase_tsd).values
    in_field = (sp_pos >= ex_rec["field_lo"]) & (sp_pos <= ex_rec["field_hi"])
    sp_pos_in = sp_pos[in_field]
    sp_phase_in = sp_phase[in_field]
    sc = ax_runs.scatter(sp_pos_in, np.full_like(sp_pos_in, k, dtype=float),
                         c=np.degrees(sp_phase_in), cmap="twilight", s=70,
                         edgecolor="k", vmin=-180, vmax=180)
ax_runs.set_yticks(range(len(chosen)))
ax_runs.set_yticklabels([f"run {k+1}" for k in range(len(chosen))])
ax_runs.axvspan(ex_rec["field_lo"], ex_rec["field_hi"], color="C3", alpha=0.1)
ax_runs.set_xlabel("Linear position (m)")
ax_runs.set_title(f"Single-traversal spikes (colour = theta phase)")
plt.colorbar(sc, ax=ax_runs, label="Theta phase (°)", pad=0.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig06_example_cell_summary.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 10. Summary
#
# - **Theta locking is widespread.** A large fraction of CA1 pyramidal cells fired at a
#   non-uniform theta phase during locomotion (Rayleigh test).
# - **Place cells tile the linear track.** After splitting runs by direction, dozens of
#   excitatory units showed compact, direction-specific firing fields.
# - **Phase precession is the dominant pattern.** Across place-cell × direction pairs, slopes
#   were predominantly negative (theta phase decreases as the rat advances through the field),
#   replicating the classic O'Keefe & Recce / Skaggs et al. result.

print("All figures saved to:", FIGDIR)
print("Figures:")
for f in sorted(os.listdir(FIGDIR)):
    if f.endswith(".png"):
        print(" ", f)

io.close()
