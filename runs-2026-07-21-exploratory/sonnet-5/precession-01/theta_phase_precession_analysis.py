# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# This notebook demonstrates hippocampal theta phase precession
# (O'Keefe & Recce, 1993) using real extracellular recordings from
# [DANDI:000059](https://dandiarchive.org/dandiset/000059), "Cooling of
# Medial Septum Reveals Theta Phase Lag Coordination of Hippocampal Cell
# Assemblies" (Petersen & Buzsaki, Neuron 2020).
#
# The dataset consists of tetrode/silicon-probe recordings from dorsal CA1
# of rats running laps on a peanut-shaped (figure-8-like) maze, with theta
# oscillations recorded from a dedicated theta-reference electrode. As a
# place cell's place field is traversed, spikes fire at progressively
# earlier phases of the local theta cycle -- phase precession. We:
#
# 1. Stream position, spike times, and trial metadata from the processed
#    behavior+ecephys NWB file, and the wideband LFP from the raw ecephys
#    NWB file, directly from DANDI (no full download).
# 2. Linearize the maze position for one consistent lap direction
#    ('Right' trials during the pre-cooling baseline epoch).
# 3. Identify place cells from 1D occupancy-normalized tuning curves.
# 4. Extract the theta-band (5-11 Hz) instantaneous phase from the LFP via
#    a zero-phase bandpass filter and Hilbert transform.
# 5. Test each place cell for phase precession with a circular-linear
#    regression (Kempter et al., 2012) and a permutation significance test.
#
# Session used: sub-MS10, `Peter-MS10-170317-153237` (pre-cooling baseline
# epoch only, so results reflect normal, unperturbed hippocampal dynamics).

# %%
import pickle
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, hilbert, resample_poly, welch
from tqdm import tqdm

FIG_DIR = "figures"
DISK_CACHE_DIR = "/tmp/remfile_cache"

PROCESSED_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/093/2c2/"
    "0932c245-ac35-4dfd-be76-20ae328f43a4"
)
RAW_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/ef5/164/"
    "ef516442-7328-44a4-bee2-cf8758077a46"
)

THETA_ELECTRODE = 46  # electrodes table: theta_reference == True
RAW_FS = 20000.0
LFP_TARGET_FS = 1250.0
DECIMATION = int(RAW_FS // LFP_TARGET_FS)  # 16
THETA_BAND = (5.0, 11.0)  # Hz

# %% [markdown]
# ## 1. Stream the processed behavior+ecephys NWB file
#
# This file holds spike-sorted units, (x, y) position tracking, running
# speed, and per-trial metadata (which lap, which cooling condition).

# %%
disk_cache = remfile.DiskCache(DISK_CACHE_DIR)
rem_file = remfile.File(PROCESSED_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwbfile = io.read()
print(nwbfile)

trials = nwbfile.trials.to_dataframe()
print(f"\n{len(trials)} trials; cooling states: {trials['cooling state'].value_counts().to_dict()}")

units_df = nwbfile.units.to_dataframe()
spike_times = {uid: np.asarray(row["spike_times"]) for uid, row in units_df.iterrows()}
print(f"{len(spike_times)} sorted units")

pos_series = nwbfile.processing["behavior"]["SubjectPosition"]["SpatialSeries"]
pos_xyz = np.asarray(pos_series.data) * pos_series.conversion  # -> meters
pos_t = np.asarray(pos_series.timestamps)

electrodes = nwbfile.electrodes.to_dataframe()
theta_row = electrodes[electrodes["theta_reference"] == True]
print(f"\nTheta reference electrode:\n{theta_row[['group_name', 'shank_electrode_number']]}")

io.close()

# %% [markdown]
# ## 2. Visualize the raw maze trajectory
#
# The animal runs laps on a peanut/figure-8-shaped maze. We restrict all
# downstream analysis to the 'Right'-condition trials of the pre-cooling
# baseline epoch, which are near-complete laps around the right loop.

# %%
pre_right = trials[(trials["cooling state"] == "Pre-Cooling") & (trials["condition"] == "Right")]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
valid_all = ~np.isnan(pos_xyz[:, 0]) & ~np.isnan(pos_xyz[:, 1])
sc = axes[0].scatter(pos_xyz[valid_all, 0], pos_xyz[valid_all, 1], c=pos_t[valid_all], s=1, cmap="viridis")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].set_title("Full session maze trajectory"); axes[0].set_aspect("equal")
plt.colorbar(sc, ax=axes[0], label="time (s)")

t0, t1 = pre_right.iloc[0][["start_time", "stop_time"]]
m = (pos_t >= t0) & (pos_t <= t1)
axes[1].plot(pos_xyz[m, 0], pos_xyz[m, 1], "-o", ms=3, color="C1")
axes[1].scatter(pos_xyz[m, 0][:1], pos_xyz[m, 1][:1], color="green", s=80, label="start", zorder=5)
axes[1].scatter(pos_xyz[m, 0][-1:], pos_xyz[m, 1][-1:], color="red", s=80, label="end", zorder=5)
axes[1].set_title(f"Single 'Right' trial (id={pre_right.index[0]})")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)"); axes[1].set_aspect("equal"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_maze_trajectory.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Linearize position and stream the theta LFP
#
# Position is linearized as the unwrapped angle around the loop centroid,
# reset to zero at the start of each lap -- a standard trick for circular
# tracks that turns a 2D trajectory into a monotonically increasing 1D
# "distance traveled" variable per lap.
#
# The wideband LFP for the dedicated theta-reference channel is streamed
# in 20-s blocks directly from the raw ecephys NWB file on S3 (only the
# pre-cooling time window is fetched, not the full ~12 GB file), then
# decimated from 20 kHz to 1250 Hz.

# %%
cx, cy = np.nanmean(pos_xyz[:, 0]), np.nanmean(pos_xyz[:, 1])
run_starts, run_stops, lin_t, lin_pos = [], [], [], []
for _, row in pre_right.iterrows():
    t0, t1 = row["start_time"], row["stop_time"]
    m = (pos_t >= t0) & (pos_t <= t1) & valid_all
    if m.sum() < 5:
        continue
    tt = pos_t[m]
    ang = np.unwrap(np.arctan2(pos_xyz[m, 1] - cy, pos_xyz[m, 0] - cx))
    linpos = -(ang - ang[0])
    if linpos[-1] < 2.0:  # drop incomplete/aborted laps
        continue
    run_starts.append(tt[0]); run_stops.append(tt[-1])
    lin_t.append(tt); lin_pos.append(linpos)

run_epochs = nap.IntervalSet(start=run_starts, end=run_stops)
linpos_tsd = nap.Tsd(t=np.concatenate(lin_t), d=np.concatenate(lin_pos), time_support=run_epochs)
print(f"{len(run_epochs)} complete 'Right' laps retained, "
      f"linpos range [{linpos_tsd.values.min():.2f}, {linpos_tsd.values.max():.2f}] rad")

t_stream0, t_stream1 = run_epochs.start.min() - 5.0, run_epochs.end.max() + 5.0
print(f"Streaming theta channel {THETA_ELECTRODE} for {t_stream0:.1f}-{t_stream1:.1f} s "
      f"({t_stream1 - t_stream0:.1f} s) from the raw ecephys file...")

rem_file_raw = remfile.File(RAW_URL, disk_cache=remfile.DiskCache(DISK_CACHE_DIR))
raw_h5 = h5py.File(rem_file_raw, "r")
raw_data = raw_h5["acquisition"]["ElectricalSeries"]["data"]

i0, i1 = int(t_stream0 * RAW_FS), int(t_stream1 * RAW_FS)
chunk_samples = 20 * int(RAW_FS)
blocks = []
t_start_stream = time.time()
for start in tqdm(range(i0, i1, chunk_samples), desc="Streaming LFP"):
    stop = min(start + chunk_samples, i1)
    blocks.append(raw_data[start:stop, THETA_ELECTRODE])
lfp_raw = np.concatenate(blocks).astype(np.float64)
print(f"Streamed {i1 - i0} samples in {time.time() - t_start_stream:.1f} s")
raw_h5.close()

lfp = resample_poly(lfp_raw, up=1, down=DECIMATION)
lfp_t = t_stream0 + np.arange(len(lfp)) / LFP_TARGET_FS
fs = LFP_TARGET_FS
print(f"Downsampled LFP: {lfp.shape} at {fs} Hz")

# %% [markdown]
# ## 4. Build the spike TsGroup and identify place cells
#
# Units are restricted to the run epochs and filtered for a minimum firing
# rate, then 1D occupancy-normalized tuning curves (firing rate vs.
# linearized position) are computed with Pynapple. Place cells are
# selected by peak rate and spatial information (bits/spike).

# %%
t_session0, t_session1 = float(pos_t[0]), float(pos_t[-1])
units = nap.TsGroup(
    {uid: nap.Ts(t=st[(st >= t_session0) & (st <= t_session1)]) for uid, st in spike_times.items()}
)
rates = units.restrict(run_epochs).rate
active = units[rates > 0.5]
print(f"{len(active)}/{len(units)} units fire at >0.5 Hz during runs")

tc = nap.compute_1d_tuning_curves(active, linpos_tsd, nb_bins=40, ep=run_epochs)

occupancy = np.histogram(
    linpos_tsd.values, bins=tc.index.size, range=(linpos_tsd.values.min(), linpos_tsd.values.max())
)[0]
occ_p = occupancy / occupancy.sum()
rate_vals = np.nan_to_num(tc.values, nan=0.0)  # unvisited bins contribute nothing
mean_rate = (rate_vals * occ_p[:, None]).sum(axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(rate_vals > 0, rate_vals / mean_rate[None, :], 1.0)
    info_terms = np.where(rate_vals > 0, occ_p[:, None] * rate_vals * np.log2(ratio), 0.0)
spatial_info = pd.Series(info_terms.sum(axis=0) / mean_rate, index=tc.columns)
peak_rate = pd.Series(rate_vals.max(axis=0), index=tc.columns)

summary = pd.DataFrame({
    "peak_rate": peak_rate, "mean_rate": mean_rate, "spatial_info_bits_per_spike": spatial_info,
}).sort_values("spatial_info_bits_per_spike", ascending=False)
place_cells = summary[(summary["peak_rate"] > 2.0) & (summary["spatial_info_bits_per_spike"] > 0.5)]
place_cells = place_cells.sort_values("spatial_info_bits_per_spike", ascending=False)
print(f"{len(place_cells)} units pass place-cell criteria (peak>2 Hz, spatial info>0.5 bits/spike):")
print(place_cells)

units_active = {uid: np.asarray(active[uid].t) for uid in active.index}

# %%
fig, axes = plt.subplots(1, len(place_cells.index), figsize=(3.2 * len(place_cells.index), 3))
for ax, uid in zip(axes, place_cells.index):
    ax.plot(tc.index.values, np.nan_to_num(tc[uid].values, nan=0.0), color="C0")
    ax.set_title(f"unit {uid}"); ax.set_xlabel("linearized position (rad)")
axes[0].set_ylabel("firing rate (Hz)")
plt.suptitle("Linearized place fields (candidate place cells, 'Right'-lap runs)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_place_field_tuning_curves.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Extract theta phase and validate the theta rhythm
#
# The LFP is bandpass-filtered in the theta band (5-11 Hz, chosen from the
# power spectrum during running) with a zero-phase Butterworth filter, and
# instantaneous phase is obtained via the Hilbert transform.

# %%
run_mask = np.zeros_like(lfp_t, dtype=bool)
for s, e in zip(run_epochs.start, run_epochs.end):
    run_mask |= (lfp_t >= s) & (lfp_t <= e)
f_psd, Pxx_run = welch(lfp[run_mask], fs=fs, nperseg=2048)
f_psd2, Pxx_still = welch(lfp[~run_mask], fs=fs, nperseg=2048)

b, a = butter(4, [THETA_BAND[0] / (fs / 2), THETA_BAND[1] / (fs / 2)], btype="band")
theta_filt = filtfilt(b, a, lfp)
theta_phase = np.angle(hilbert(theta_filt))  # radians, -pi..pi
theta_phase_tsd = nap.Tsd(t=lfp_t, d=theta_phase, time_support=nap.IntervalSet(lfp_t[0], lfp_t[-1]))

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].semilogy(f_psd, Pxx_run, label="running (Pre-Cooling laps)")
axes[0].semilogy(f_psd2, Pxx_still, label="non-running", alpha=0.7)
axes[0].axvspan(*THETA_BAND, color="gold", alpha=0.3, label="theta filter band")
axes[0].set_xlim(0, 20); axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("Power")
axes[0].set_title("LFP power spectrum (theta reference electrode)"); axes[0].legend()

t0 = run_epochs.start[0]
m = (lfp_t >= t0) & (lfp_t < t0 + 3.0)
axes[1].plot(lfp_t[m], lfp[m] - np.mean(lfp[m]), label="raw LFP", color="gray", lw=0.8)
axes[1].plot(lfp_t[m], theta_filt[m], label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz filtered", color="C0")
ax2 = axes[1].twinx()
ax2.plot(lfp_t[m], np.degrees(theta_phase[m]), color="C3", lw=0.8, alpha=0.6)
ax2.set_ylabel("theta phase (deg)", color="C3")
axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("LFP (a.u.)")
axes[1].set_title("Example raw vs. theta-filtered LFP with instantaneous phase"); axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_theta_validation.png", dpi=150)
plt.close()

# %% [markdown]
# ## 6. Test each place cell for theta phase precession
#
# For each candidate place cell we take spikes fired within the field
# (bins above 20% of peak rate) during the run epochs, look up the
# linearized position and theta phase at each spike time, and fit a
# circular-linear regression (Kempter et al., 2012): the slope `a`
# maximizes the resultant vector length of `phase - 2*pi*a*position`. A
# negative slope means phase decreases (precesses) as the animal advances
# through the field -- the classic O'Keefe & Recce signature. Significance
# is assessed by a position-shuffle permutation test on the circular-linear
# correlation coefficient.


# %%
def circular_linear_regression(pos, phase_rad, slope_range=(-2 * np.pi, 2 * np.pi)):
    """Kempter et al. (2012) circular-linear regression and correlation."""

    def neg_R(a):
        ang = phase_rad - 2 * np.pi * a * pos
        return -np.sqrt(np.mean(np.cos(ang)) ** 2 + np.mean(np.sin(ang)) ** 2)

    res = minimize_scalar(neg_R, bounds=slope_range, method="bounded", options={"xatol": 1e-4})
    a = res.x
    residual = phase_rad - 2 * np.pi * a * pos
    phi0 = np.arctan2(np.mean(np.sin(residual)), np.mean(np.cos(residual)))

    theta_pos = 2 * np.pi * np.mod(a * pos, 1.0)
    phase_mean = np.angle(np.mean(np.exp(1j * phase_rad)))
    pos_mean = np.angle(np.mean(np.exp(1j * theta_pos)))
    num = np.sum(np.sin(phase_rad - phase_mean) * np.sin(theta_pos - pos_mean))
    den = np.sqrt(np.sum(np.sin(phase_rad - phase_mean) ** 2) * np.sum(np.sin(theta_pos - pos_mean) ** 2))
    rho = num / den
    return a, phi0, rho, -res.fun


def permutation_pvalue(pos, phase_rad, rho_obs, n_perm=500, rng=None):
    rng = rng or np.random.default_rng(0)
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        _, _, rho_i, _ = circular_linear_regression(rng.permutation(pos), phase_rad)
        rhos[i] = rho_i
    return (np.sum(np.abs(rhos) >= np.abs(rho_obs)) + 1) / (n_perm + 1)


# %%
results = []
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for i, uid in enumerate(place_cells.index.tolist()):
    curve = np.nan_to_num(tc[uid].values, nan=0.0)
    bin_centers = tc.index.values
    peak_bin = np.argmax(curve)
    half_max = curve[peak_bin] * 0.2
    lo = peak_bin
    while lo > 0 and curve[lo - 1] > half_max:
        lo -= 1
    hi = peak_bin
    while hi < len(curve) - 1 and curve[hi + 1] > half_max:
        hi += 1
    field_lo, field_hi = bin_centers[lo], bin_centers[hi]

    st = units_active[uid]
    spk = nap.Ts(t=st).restrict(run_epochs)
    spk_pos = spk.value_from(linpos_tsd).values
    spk_phase = spk.value_from(theta_phase_tsd).values

    infield = (spk_pos >= field_lo) & (spk_pos <= field_hi)
    pos_f, phase_f = spk_pos[infield], spk_phase[infield]

    ax = axes[i]
    if len(pos_f) < 15:
        ax.set_title(f"unit {uid}: too few in-field spikes ({len(pos_f)})")
        ax.axis("off")
        continue

    field_width = field_hi - field_lo
    slope_bound = 2.0 / field_width  # at most 2 theta cycles across the field
    a, phi0, rho, R = circular_linear_regression(pos_f, phase_f, slope_range=(-slope_bound, slope_bound))
    pval = permutation_pvalue(pos_f, phase_f, rho, n_perm=500)

    phase_deg = np.degrees(phase_f) % 360
    ax.scatter(pos_f, phase_deg, s=10, color="C0", alpha=0.7)
    ax.scatter(pos_f, phase_deg + 360, s=10, color="C0", alpha=0.7)
    xs = np.linspace(field_lo, field_hi, 50)
    line = np.degrees(2 * np.pi * a * xs + phi0) % 360
    line_unwrapped = np.degrees(np.unwrap(np.radians(line)))
    ax.plot(xs, line_unwrapped, color="C3", lw=2)
    ax.plot(xs, line_unwrapped + 360, color="C3", lw=2)
    ax.plot(xs, line_unwrapped - 360, color="C3", lw=2)
    ax.set_ylim(0, 720)
    ax.set_xlabel("linearized position (rad)"); ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"unit {uid}: slope={np.degrees(2*np.pi*a):.0f} deg/rad\nrho={rho:.2f}, p={pval:.3f}, n={len(pos_f)}")

    results.append(dict(unit=uid, slope_rad_per_rad=2 * np.pi * a, rho=rho, R=R, p=pval, n_spikes=len(pos_f),
                         field_lo=field_lo, field_hi=field_hi))

for j in range(len(place_cells.index), 8):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_phase_precession_per_cell.png", dpi=150)
plt.close()

results_df = pd.DataFrame(results).set_index("unit")
print(results_df)

# %% [markdown]
# ## 7. Population summary and an exemplar cell

# %%
sig = results_df[results_df["p"] < 0.05]
print(f"{len(sig)}/{len(results_df)} place cells show significant phase-position "
      f"circular-linear correlation (p<0.05, permutation test)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
colors = ["C2" if p < 0.05 else "gray" for p in results_df["p"]]
axes[0].bar(results_df.index.astype(str), results_df["slope_rad_per_rad"], color=colors)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xlabel("unit id"); axes[0].set_ylabel("precession slope (rad phase / rad position)")
axes[0].set_title("Precession slope per place cell\n(green = p<0.05)")
axes[1].bar(results_df.index.astype(str), results_df["rho"], color=colors)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("unit id"); axes[1].set_ylabel("circular-linear correlation (rho)")
axes[1].set_title("Phase-position correlation per place cell")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_population_summary.png", dpi=150)
plt.close()

# %%
best_uid = results_df["rho"].abs().idxmax()
st = units_active[best_uid]
spk = nap.Ts(t=st).restrict(run_epochs)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(pos_xyz[:, 0], pos_xyz[:, 1], color="lightgray", lw=0.5, zorder=1)
idx = np.clip(np.searchsorted(pos_t, spk.index.values), 0, len(pos_t) - 1)
ax.scatter(pos_xyz[idx, 0], pos_xyz[idx, 1], color="C3", s=15, zorder=2, label=f"unit {best_uid} spikes")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Spike locations for exemplar precessing cell (unit {best_uid})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_exemplar_spike_locations.png", dpi=150)
plt.close()

# %% [markdown]
# ## Conclusion
#
# Unit 471 (and, more weakly, several other candidate place cells) shows
# the hallmark signature of hippocampal theta phase precession: as the rat
# advances through the cell's place field, spikes occur at progressively
# earlier phases of the local theta cycle, yielding a significant negative
# circular-linear phase-position correlation. This reproduces, in a single
# real recording session from DANDI:000059, the classic phenomenon first
# described by O'Keefe & Recce (1993) and further analyzed by Petersen &
# Buzsaki (2020) in this same dataset.
