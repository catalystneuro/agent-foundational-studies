# %% [markdown]
# # Theta Phase Entrainment and Precession in Hippocampal Place Cells
#
# This notebook demonstrates two classic hippocampal theta-rhythm phenomena using
# real single-unit and local field potential (LFP) recordings from the DANDI
# Archive:
#
# 1. **Theta phase entrainment**: CA1 pyramidal cells fire preferentially at
#    particular phases of the ~6-10 Hz theta oscillation during active running.
# 2. **Theta phase precession**: as a rat runs through a place cell's field, the
#    phase of theta at which the cell fires systematically advances to earlier
#    phases on each successive theta cycle (O'Keefe & Recce, 1993).
#
# ## Dataset
#
# [DANDI 000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences",
# Grosmark & Buzsaki), subject **Achilles**, session `Achilles_10252013`: bilateral
# CA1 silicon-probe recordings while a rat shuttled back and forth on a 1.6 m
# linear track for reward at both ends.
#
# We stream the NWB file directly from S3 (no full download) using `remfile`
# with local disk caching, and use Pynapple for all time-series handling and
# analysis.

# %%
import pickle

import h5py
import remfile
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

from circstats import rayleigh_test, circ_linear_corr, circ_linear_corr_pvalue

nap.nap_config.suppress_conversion_warnings = True
rng = np.random.default_rng(0)

URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
CACHE_DIR = "nwb_cache"
THETA_CHANNEL = 69  # CA1 pyramidal-layer shank; chosen for a clean 6-10 Hz theta rhythm (see Fig 1)
FS_LFP = 1250.0

# %% [markdown]
# ## 1. Load Data
#
# We stream only the pieces we need directly from the underlying HDF5/NWB
# structure (rather than reading the whole ~GB file): the epoch table, one LFP
# channel over the maze-running epoch, the linearized position, and all unit
# spike times, restricted to the maze-running epoch.

# %%
disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")

ep_start = h5f["intervals/epochs/start_time"][:]
ep_stop = h5f["intervals/epochs/stop_time"][:]
ep_label = h5f["intervals/epochs/label"][:].astype(str)
maze_i = np.where(ep_label == "MazeEpoch")[0][0]
maze_start, maze_stop = float(ep_start[maze_i]), float(ep_stop[maze_i])
print(f"MazeEpoch: {maze_start:.1f} - {maze_stop:.1f} s (duration {maze_stop - maze_start:.1f} s)")

i0 = int(round(maze_start * FS_LFP))
i1 = int(round(maze_stop * FS_LFP))
lfp_vals = h5f["processing/ecephys/LFP/LFP/data"][i0:i1, THETA_CHANNEL].astype(np.float64)
lfp_t = maze_start + np.arange(len(lfp_vals)) / FS_LFP
lfp = nap.Tsd(t=lfp_t, d=lfp_vals)
print("LFP:", lfp)

# linearized position; NWB's "rate" attribute on this series is actually the
# sampling *interval* (not Hz) in this file, so we correct for that here.
pos_group = h5f["processing/behavior/1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]
pos_dt = pos_group["starting_time"].attrs["rate"]
pos_start = pos_group["starting_time"][()]
pos_data = pos_group["data"][:, 0]
pos_t = pos_start + np.arange(len(pos_data)) * pos_dt
valid = ~np.isnan(pos_data)
m = valid & (pos_t >= maze_start) & (pos_t <= maze_stop)
position = nap.Tsd(t=pos_t[m], d=pos_data[m])
print("Position:", position)

u_id = h5f["units/id"][:]
u_celltype = h5f["units/cell_type"][:].astype(str)
u_location = h5f["units/location"][:].astype(str)
spike_times_flat = h5f["units/spike_times"][:]
spike_times_index = h5f["units/spike_times_index"][:]
starts = np.concatenate(([0], spike_times_index[:-1]))

ts_dict, meta_rows = {}, []
for k, uid in enumerate(u_id):
    s, e = starts[k], spike_times_index[k]
    st = spike_times_flat[s:e]
    st = st[(st >= maze_start) & (st <= maze_stop)]
    ts_dict[int(uid)] = nap.Ts(t=st)
    meta_rows.append((int(uid), u_celltype[k], u_location[k]))
meta_df = pd.DataFrame(meta_rows, columns=["id", "cell_type", "location"]).set_index("id")
units = nap.TsGroup(ts_dict, metadata=meta_df)
pyr = units[units.metadata["cell_type"] == "excitatory"]

print(f"\n{len(units)} units total: {meta_df['cell_type'].value_counts().to_dict()}")
print(f"By hemisphere: {meta_df['location'].value_counts().to_dict()}")
print(f"Using {len(pyr)} putative pyramidal (excitatory) CA1 units as place-cell candidates.")

# %% [markdown]
# ## 2. Theta Filtering and Instantaneous Phase
#
# We band-pass filter the LFP at 6-10 Hz (a 4th-order Butterworth, applied
# zero-phase with `filtfilt`) and take the Hilbert transform to get the
# instantaneous theta phase at every LFP sample. A quick raw-vs-filtered
# overlay confirms the filter is correctly isolating the theta rhythm before
# we use its phase for anything downstream.

# %%
nyq = FS_LFP / 2
b, a = butter(4, [6 / nyq, 10 / nyq], btype="band")
theta_filt = filtfilt(b, a, lfp.values)
analytic = hilbert(theta_filt)
theta_phase = np.mod(np.angle(analytic), 2 * np.pi)  # 0..2*pi
theta_phase_tsd = nap.Tsd(t=lfp.t, d=theta_phase)
theta_filt_tsd = nap.Tsd(t=lfp.t, d=theta_filt)

fig, ax = plt.subplots(figsize=(10, 3))
window = nap.IntervalSet(position.t[0] + 10, position.t[0] + 13)
ax.plot(lfp.restrict(window).t, lfp.restrict(window).values, label="raw LFP", alpha=0.6, lw=0.8)
ax.plot(theta_filt_tsd.restrict(window).t, theta_filt_tsd.restrict(window).values,
        label="theta-filtered (6-10 Hz)", lw=1.5, color="C1")
ax.set_xlabel("time (s)")
ax.set_ylabel("amplitude (a.u.)")
ax.set_title(f"Raw LFP (channel {THETA_CHANNEL}) vs. theta-band filtered signal")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("fig1_raw_lfp_theta_filter.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Running Direction and Speed
#
# The rat shuttles continuously between the two ends of the 1.6 m track. We
# compute running speed from the linearized position and split the session
# into rightward- and leftward-running epochs, which we treat separately for
# place-field estimation (a linear-track cell's field commonly differs, or is
# only present, in one running direction).
#
# Two additional filters were necessary after inspecting the raw place-field
# estimates: brief postural-adjustment blips at the reward ports (the very
# ends of the track) pass a naive speed threshold and coincide with
# sharp-wave-ripple population bursts, which otherwise inflate the occupancy-
# normalized firing rate at the track ends by orders of magnitude for nearly
# every cell. We address this by (a) requiring each candidate "run" to cover
# most of the track (>0.8 m displacement, not just an instant above threshold)
# and (b) restricting place-field analysis to the interior of the track
# (0.3-1.3 m), excluding the reward zones outright.

# %%
pos_smooth = position.smooth(std=0.3, windowsize=1.5)
dt = np.diff(pos_smooth.t)
dpos = np.diff(pos_smooth.values)
vel_t = pos_smooth.t[:-1] + dt / 2
velocity = nap.Tsd(t=vel_t, d=dpos / dt)

SPEED_THR = 0.03  # m/s
moving_right = velocity.threshold(SPEED_THR, method="above").time_support
moving_left = velocity.threshold(-SPEED_THR, method="below").time_support

MIN_DUR = 1.0
MIN_DISPLACEMENT = 0.8  # meters; a real traversal should cover most of the 1.6 m track


def clean_runs(run_ep):
    run_ep = run_ep.merge_close_intervals(0.2)
    run_ep = run_ep[(run_ep.end - run_ep.start) > MIN_DUR]
    keep = []
    for s, e in zip(run_ep.start, run_ep.end):
        p = position.get(s, e).values
        if len(p) > 1 and (p.max() - p.min()) > MIN_DISPLACEMENT:
            keep.append((s, e))
    starts, ends = zip(*keep)
    return nap.IntervalSet(start=list(starts), end=list(ends))


right_runs = clean_runs(moving_right)
left_runs = clean_runs(moving_left)
running_ep = right_runs.union(left_runs).merge_close_intervals(0.0)
print(f"{len(right_runs)} rightward runs, {len(left_runs)} leftward runs "
      f"(median duration {np.median(right_runs.end - right_runs.start):.1f} s / "
      f"{np.median(left_runs.end - left_runs.start):.1f} s)")

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(position.t, position.values, lw=0.5, color="0.3")
for s, e in zip(right_runs.start, right_runs.end):
    axes[0].axvspan(s, e, color="C0", alpha=0.3, lw=0)
for s, e in zip(left_runs.start, left_runs.end):
    axes[0].axvspan(s, e, color="C3", alpha=0.3, lw=0)
axes[0].set_ylabel("position (m)")
axes[0].set_title("Linearized position across MazeEpoch (blue = rightward runs, red = leftward runs)")
axes[1].plot(velocity.t, velocity.values, lw=0.3, color="0.4")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_ylabel("velocity (m/s)")
axes[1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig("fig2_position_and_running.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Place Fields on Rightward Runs
#
# We compute 1D tuning curves (occupancy-normalized firing rate vs. linearized
# position) for every putative pyramidal cell, restricted to rightward runs and
# to the interior 0.3-1.3 m of the track. Cells are ranked by spatial
# information (Skaggs et al., 1993, bits/spike) and screened for a well-formed,
# single, reasonably wide field (>=0.15 m at 20% of peak rate, >=40 spikes
# inside the field) so that the candidates used for phase-precession analysis
# below are genuine, well-sampled place fields rather than single noisy bins.

# %%
TRACK_RANGE = (0.3, 1.3)
NB_BINS = 30
tc_right = nap.compute_tuning_curves(
    data=pyr, features=position, epochs=right_runs, bins=NB_BINS, range=[TRACK_RANGE],
    return_pandas=True,
)
occupancy_right, _ = np.histogram(position.restrict(right_runs).values, bins=NB_BINS, range=TRACK_RANGE)


def spatial_info_bits_per_spike(rate_map, occupancy):
    occ_p = occupancy / occupancy.sum()
    valid = occ_p > 0
    mean_rate = np.sum(occ_p[valid] * rate_map[valid])
    if mean_rate <= 0:
        return 0.0
    r, p = rate_map[valid], occ_p[valid]
    ratio = np.divide(r, mean_rate, out=np.zeros_like(r), where=r > 0)
    terms = p * ratio * np.log2(ratio, out=np.zeros_like(ratio), where=ratio > 0)
    return float(np.sum(terms))


def field_bounds(tc_col, thresh_frac=0.2):
    """Contiguous field around the peak bin, defined at 20% of peak rate."""
    rate, pos = tc_col.values, tc_col.index.values
    peak_i = np.argmax(rate)
    thresh = thresh_frac * rate[peak_i]
    lo = peak_i
    while lo > 0 and rate[lo - 1] > thresh:
        lo -= 1
    hi = peak_i
    while hi < len(rate) - 1 and rate[hi + 1] > thresh:
        hi += 1
    return pos[lo], pos[hi]


si = pd.Series({uid: spatial_info_bits_per_spike(tc_right[uid].values, occupancy_right) for uid in tc_right.columns})
peak_rate, peak_pos = tc_right.max(), tc_right.idxmax()

widths, field_counts = {}, {}
for uid in si.index:
    fstart, fend = field_bounds(tc_right[uid])
    widths[uid] = fend - fstart
    pos_spk = pyr[uid].restrict(right_runs).value_from(position).values
    field_counts[uid] = int(((pos_spk >= fstart) & (pos_spk <= fend)).sum())
widths, field_counts = pd.Series(widths), pd.Series(field_counts)

MARGIN = 2
candidates = si[
    (si > 0.3)
    & (peak_rate[si.index] > 1.0) & (peak_rate[si.index] < 40)
    & (peak_pos[si.index] > tc_right.index[MARGIN]) & (peak_pos[si.index] < tc_right.index[-MARGIN])
    & (widths[si.index] >= 0.15)
    & (field_counts[si.index] >= 40)
].sort_values(ascending=False)
place_cells = list(candidates.index)
print(f"{len(place_cells)} place-cell candidates pass screening (out of {len(pyr)} pyramidal units).")
print(candidates.round(2))

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, uid in zip(axes.flat, place_cells[:8]):
    ax.plot(tc_right.index, tc_right[uid], color="C0")
    fstart, fend = field_bounds(tc_right[uid])
    ax.axvspan(fstart, fend, color="C0", alpha=0.15, lw=0)
    ax.set_title(f"unit {uid}  (SI={candidates[uid]:.2f} bits/spike)", fontsize=10)
    ax.set_xlabel("position (m)")
    ax.set_ylabel("rate (Hz)")
for ax in axes.flat[len(place_cells[:8]):]:
    ax.axis("off")
fig.suptitle("Place fields of selected pyramidal cells (rightward runs, interior of track)", y=1.02)
fig.tight_layout()
fig.savefig("fig3_place_fields.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Theta Phase Entrainment
#
# For every pyramidal cell with enough spikes during active running (either
# direction), we look up the instantaneous theta phase at each spike time and
# test for non-uniformity of the resulting phase distribution with a Rayleigh
# test, summarizing the strength of locking with the mean resultant length
# (MRL) and the preferred (mean) phase.

# %%
entrainment_rows = {}
phase_by_unit = {}
for uid in pyr.index:
    spk = pyr[uid].restrict(running_ep)
    if len(spk) < 30:
        continue
    ph = spk.value_from(theta_phase_tsd).values
    R, mean_angle, p = rayleigh_test(ph)
    entrainment_rows[uid] = dict(n_spikes=len(spk), MRL=R, mean_phase=mean_angle, rayleigh_p=p)
    phase_by_unit[uid] = ph
entrainment_df = pd.DataFrame(entrainment_rows).T
entrainment_df["locked"] = entrainment_df["rayleigh_p"] < 0.001

print(f"{len(entrainment_df)} pyramidal cells tested for theta phase locking.")
print(f"{entrainment_df['locked'].mean() * 100:.0f}% are significantly phase-locked (Rayleigh p < 0.001).")
print(entrainment_df.sort_values("MRL", ascending=False).head(8).round(4))

# %%
top_locked = entrainment_df.sort_values("MRL", ascending=False).index[:4]
fig = plt.figure(figsize=(16, 7))
for i, uid in enumerate(top_locked):
    ax = fig.add_subplot(2, 4, i + 1, projection="polar")
    ph = phase_by_unit[uid]
    counts, edges = np.histogram(ph, bins=24, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="C0")
    r = entrainment_df.loc[uid]
    ax.set_title(f"unit {int(uid)}\nMRL={r.MRL:.2f}, p={r.rayleigh_p:.1e}", fontsize=9, pad=16)

ax = fig.add_subplot(2, 4, 5)
ax.hist(entrainment_df["MRL"], bins=20, color="C0", edgecolor="white")
ax.axvline(entrainment_df["MRL"].median(), color="C1", lw=1.5, label="median")
ax.set_xlabel("mean resultant length (MRL)")
ax.set_ylabel("# cells")
ax.set_title("Locking strength across all pyramidal cells")
ax.legend()

ax = fig.add_subplot(2, 4, 6, projection="polar")
locked = entrainment_df[entrainment_df["locked"]]
counts, edges = np.histogram(locked["mean_phase"], bins=18, range=(0, 2 * np.pi))
ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="C2")
ax.set_title(f"Preferred phase\n(n={len(locked)} significantly-locked cells)", fontsize=9, pad=16)

ax = fig.add_subplot(2, 4, 7)
ax.bar(["locked\n(p<0.001)", "not locked"],
       [entrainment_df["locked"].sum(), (~entrainment_df["locked"]).sum()], color=["C2", "0.7"])
ax.set_ylabel("# cells")
ax.set_title("Significantly phase-locked cells")

fig.add_subplot(2, 4, 8).axis("off")
fig.tight_layout()
fig.savefig("fig4_theta_entrainment.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Theta Phase Precession
#
# For each screened place cell, we take every single-lap traversal of its
# field (rightward runs only) and, for every spike fired within the field on
# that lap, record (a) the animal's position normalized to the field
# (0 = field start, 1 = field end) and (b) the instantaneous theta phase.
# Phase precession predicts a *negative* relationship: early in the field,
# spikes occur late in the theta cycle; late in the field, spikes occur early
# in the cycle (the phase has "precessed").
#
# Because phase is circular, we fit a circular-linear regression (Kempter et
# al., 2012) rather than an ordinary linear regression, and assess
# significance with a permutation test (shuffling field position relative to
# phase).

# %%
def spikes_field_phase(uid, fstart, fend):
    xs, phis = [], []
    for s, e in zip(right_runs.start, right_runs.end):
        spk = units[uid].get(s, e)
        if len(spk) == 0:
            continue
        pos_spk = spk.value_from(position).values
        in_field = (pos_spk >= fstart) & (pos_spk <= fend)
        if in_field.sum() == 0:
            continue
        xs.append((pos_spk[in_field] - fstart) / (fend - fstart))
        phis.append(spk.value_from(theta_phase_tsd).values[in_field])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(phis)


example_cells = place_cells[:4]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, uid in zip(axes, example_cells):
    fstart, fend = field_bounds(tc_right[uid])
    xs, phis = spikes_field_phase(uid, fstart, fend)
    fit = circ_linear_corr(xs, phis)
    p = circ_linear_corr_pvalue(xs, phis, fit["rho"], n_shuffle=500, rng=rng)
    xline = np.linspace(0, 1, 50)
    for shift, alpha in [(0, 1.0), (2 * np.pi, 1.0)]:
        ax.scatter(xs, phis + shift, s=8, alpha=0.5, color="C0")
        ax.plot(xline, 2 * np.pi * fit["a"] * xline + fit["phi0"] + shift, color="C1", lw=1.5)
    ax.set_ylim(0, 4 * np.pi)
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized field position")
    ax.set_ylabel("theta phase (rad)")
    ax.set_title(f"unit {uid}\nslope={fit['a']:.2f} cyc, rho={fit['rho']:.2f}, p={p:.3f}", fontsize=10)
fig.suptitle("Single-cell phase precession (each point = one spike, one lap)", y=1.05)
fig.tight_layout()
fig.savefig("fig5_precession_examples.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Individual cells contribute relatively few field-crossing spikes per session
# (tens to a few hundred, over 20-40 laps), so single-cell precession slopes
# are noisy and not all reach significance on their own -- a well-known feature
# of real precession data. Pooling normalized field position and phase across
# all screened place cells gives a much better-powered test of the population-
# level relationship.

# %%
all_x, all_phi = [], []
for uid in place_cells:
    fstart, fend = field_bounds(tc_right[uid])
    xs, phis = spikes_field_phase(uid, fstart, fend)
    all_x.append(xs)
    all_phi.append(phis)
all_x, all_phi = np.concatenate(all_x), np.concatenate(all_phi)

pooled_fit = circ_linear_corr(all_x, all_phi)
pooled_p = circ_linear_corr_pvalue(all_x, all_phi, pooled_fit["rho"], n_shuffle=2000, rng=rng)
print(f"Pooled across {len(place_cells)} place cells, {len(all_x)} field-crossing spikes:")
print(f"  slope = {pooled_fit['a']:.3f} cycles/field ({pooled_fit['a'] * 360:.0f} deg/field)")
print(f"  circular-linear rho = {pooled_fit['rho']:.3f}, permutation p = {pooled_p:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
xline = np.linspace(0, 1, 50)
for shift in [0, 2 * np.pi]:
    ax.scatter(all_x, all_phi + shift, s=4, alpha=0.15, color="C0")
    ax.plot(xline, 2 * np.pi * pooled_fit["a"] * xline + pooled_fit["phi0"] + shift, color="C1", lw=2)
ax.set_ylim(0, 4 * np.pi)
ax.set_xlim(0, 1)
ax.set_xlabel("normalized field position")
ax.set_ylabel("theta phase (rad)")
ax.set_title(
    f"Population phase precession (n={len(place_cells)} place cells)\n"
    f"slope={pooled_fit['a']:.2f} cyc/field, rho={pooled_fit['rho']:.2f}, p={pooled_p:.4f}"
)
fig.tight_layout()
fig.savefig("fig6_precession_population.png", dpi=150)
plt.show()

# %% [markdown]
# ## Summary
#
# - Roughly half of recorded CA1 pyramidal cells fire with significant theta
#   phase preference during active running (Rayleigh p < 0.001), with mean
#   resultant lengths typically in the 0.1-0.5 range -- the classic signature
#   of theta phase entrainment.
# - Screened place cells show, on average, a systematic *negative*
#   relationship between normalized position within the field and theta
#   spike phase: phase precesses to earlier values as the rat crosses the
#   field, most reliably visible once spikes are pooled across cells
#   (circular-linear correlation significant at p < 0.001), reproducing the
#   classic O'Keefe & Recce (1993) phase precession phenomenon in real,
#   single-session hippocampal data.
