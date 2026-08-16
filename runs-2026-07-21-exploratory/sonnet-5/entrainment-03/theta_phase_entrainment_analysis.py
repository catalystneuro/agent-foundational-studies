# %% [markdown]
# # Theta Phase Entrainment of Hippocampal Neurons
#
# This notebook demonstrates theta phase entrainment: the tendency of
# hippocampal neurons to fire preferentially at a particular phase of the
# ~6-10 Hz theta oscillation that dominates the local field potential (LFP)
# during active locomotion (Buzsaki, 2002; O'Keefe & Recce, 1993). We test
# this directly using real single-unit and LFP recordings from CA1, and
# compare entrainment strength between putative pyramidal (excitatory) cells
# and interneurons (inhibitory), since hippocampal interneurons are
# classically reported to be more strongly and consistently theta-locked
# than principal cells (Csicsvari et al., 1999; Klausberger & Somogyi, 2008).
#
# ## Dataset
#
# [DANDI 000044](https://dandiarchive.org/dandiset/000044) ("Diversity in
# neural firing dynamics supports both rigid and learned hippocampal
# sequences", Grosmark & Buzsaki, NYU), subject **Gatsby**, session
# `Gatsby_08022013`: bilateral CA1 silicon-probe recordings (80 sorted
# units: 25 in left CA1, 55 in right CA1, split into putative excitatory and
# inhibitory cells) while the rat shuttled back and forth on a 1.6 m linear
# track for reward. The NWB file is streamed directly from S3 with
# `remfile` (disk-cached locally, no full download), and Pynapple is used
# for all time-series handling and analysis.
#
# ## Approach
#
# 1. Extract instantaneous theta phase from one CA1 LFP channel, chosen
#    objectively as the channel with the largest fraction of spectral power
#    in the 6-10 Hz band (6-10 Hz bandpass + Hilbert transform).
# 2. Identify locomotion epochs from linearized position (theta is weak or
#    absent during immobility, so entrainment is tested only while the
#    animal is actively running).
# 3. For every unit, assign each spike the instantaneous theta phase at
#    which it occurred and test for non-uniformity of the spike-phase
#    distribution with a Rayleigh test, using the mean resultant length
#    (MRL) as a continuous measure of entrainment strength.
# 4. Cross-check the phase-based result with two phase-free signatures of
#    theta entrainment: the spike-triggered average LFP (does the raw LFP
#    oscillate around a cell's spikes?) and the spike-time autocorrelogram
#    (do a cell's own spikes show ~125-150 ms rhythmicity?).
# 5. Compare entrainment strength between excitatory and inhibitory units at
#    the population level, and relate theta-band LFP amplitude to running
#    speed as a sanity check that the effect tracks genuine locomotion-
#    driven theta.

# %%
import h5py
import remfile
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, welch
from tqdm import tqdm

nap.nap_config.suppress_conversion_warnings = True

URL = "https://dandiarchive.s3.amazonaws.com/blobs/0a1/72f/0a172fd9-8a8d-403f-bacc-2c72488ea259"
CACHE_DIR = "nwb_cache"
FS_LFP = 1250.0

# %% [markdown]
# ## 1. Load Data
#
# We stream only the pieces needed directly from the underlying HDF5/NWB
# structure (rather than the whole ~7.9 GB file): the epoch table, LFP
# channels over the maze-running epoch, the linearized position, and all
# unit spike times restricted to the maze epoch.

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

# %% [markdown]
# ### 1a. Select a theta-band LFP channel objectively
#
# We compute the fraction of spectral power falling in the 6-10 Hz theta
# band (relative to 1-50 Hz total power) for a set of candidate channels
# spanning the probe, using a 120 s snippet, and pick the channel with the
# strongest theta signature.

# %%
candidate_channels = [0, 4, 8, 16, 24, 40, 56, 72, 88, 97, 99, 100, 104, 120]
n_snip = int(round(120 * FS_LFP))
theta_fracs = {}
for ch in tqdm(candidate_channels, desc="Scanning channels for theta"):
    seg = h5f["processing/ecephys/LFP/LFP/data"][i0:i0 + n_snip, ch].astype(np.float64)
    f, pxx = welch(seg, fs=FS_LFP, nperseg=4096)
    band = (f >= 6) & (f <= 10)
    total = (f >= 1) & (f <= 50)
    theta_fracs[ch] = pxx[band].sum() / pxx[total].sum()

THETA_CHANNEL = max(theta_fracs, key=theta_fracs.get)
print("Theta power fraction by channel:", {k: round(v, 3) for k, v in theta_fracs.items()})
print(f"Selected THETA_CHANNEL = {THETA_CHANNEL} (theta fraction = {theta_fracs[THETA_CHANNEL]:.3f})")

# %%
lfp_vals = h5f["processing/ecephys/LFP/LFP/data"][i0:i1, THETA_CHANNEL].astype(np.float64)
lfp_t = maze_start + np.arange(len(lfp_vals)) / FS_LFP
lfp = nap.Tsd(t=lfp_t, d=lfp_vals)
print("LFP:", lfp)

# Linearized position (already confirmed to be sampled at `rate` Hz in this
# file: n_samples / rate matches the MazeEpoch duration exactly). The
# tracker loses the animal intermittently (occlusion), leaving NaN gaps of
# up to ~80 s; we build an explicit time_support of only the contiguous
# tracked bouts (consecutive valid samples <= 4x the nominal sample
# interval apart) so that downstream speed/running-epoch computations do
# not bridge across untracked gaps.
pos_group = h5f["processing/behavior/1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]
pos_rate = pos_group["starting_time"].attrs["rate"]
pos_start = pos_group["starting_time"][()]
pos_data = pos_group["data"][:, 0]
pos_t = pos_start + np.arange(len(pos_data)) / pos_rate
valid = ~np.isnan(pos_data)
m = valid & (pos_t >= maze_start) & (pos_t <= maze_stop)
vt, vd = pos_t[m], pos_data[m]

expected_dt = 1.0 / pos_rate
gap = np.diff(vt)
bout_break = np.where(gap > 4 * expected_dt)[0]
bout_starts = np.concatenate([[0], bout_break + 1])
bout_ends = np.concatenate([bout_break, [len(vt) - 1]])
# drop degenerate bouts (<5 samples) that are too short to smooth/differentiate
long_enough = (bout_ends - bout_starts + 1) >= 5
bout_starts, bout_ends = bout_starts[long_enough], bout_ends[long_enough]
tracked_ep = nap.IntervalSet(start=vt[bout_starts] - expected_dt / 2, end=vt[bout_ends] + expected_dt / 2)
print(f"Tracked bouts: {len(tracked_ep)}, total tracked time = {tracked_ep.tot_length():.1f} s "
      f"of {maze_stop - maze_start:.1f} s MazeEpoch")

position = nap.Tsd(t=vt, d=vd, time_support=tracked_ep)
print("Position:", position)

# Units, restricted to the maze epoch, with cell-type / location metadata.
spike_times_flat = h5f["units/spike_times"][:]
spike_times_index = h5f["units/spike_times_index"][:]
cell_type = h5f["units/cell_type"][:].astype(str)
location = h5f["units/location"][:].astype(str)
shank_id = h5f["units/shank_id"][:]

bounds = np.concatenate([[0], spike_times_index])
units_dict = {}
meta_rows = []
for u in range(len(spike_times_index)):
    st = spike_times_flat[bounds[u]:bounds[u + 1]]
    st_maze = st[(st >= maze_start) & (st <= maze_stop)]
    units_dict[u] = st_maze
    meta_rows.append({"cell_type": cell_type[u], "location": location[u], "shank_id": shank_id[u]})

units = nap.TsGroup(
    {u: nap.Ts(t=t) for u, t in units_dict.items()},
    time_support=nap.IntervalSet(maze_start, maze_stop),
)
meta_df = pd.DataFrame(meta_rows)
units.set_info(cell_type=meta_df["cell_type"].values)
units.set_info(location=meta_df["location"].values)
units.set_info(shank_id=meta_df["shank_id"].values)
print(units)
print(units.get_info("cell_type").value_counts())

# %% [markdown]
# ## 2. Raw LFP and Theta-Band Filtering
#
# We band-pass filter the selected channel at 6-10 Hz with a 4th-order
# Butterworth filter (zero-phase, `filtfilt`) and extract the analytic
# signal via the Hilbert transform to get instantaneous theta phase and
# amplitude.

# %%
def bandpass_filter(data, fs, low, high, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


lfp_theta_vals = bandpass_filter(lfp.values, FS_LFP, 6.0, 10.0)
analytic = hilbert(lfp_theta_vals)
theta_phase_vals = np.angle(analytic)  # -pi to pi
theta_amp_vals = np.abs(analytic)

lfp_theta = nap.Tsd(t=lfp.index.values, d=lfp_theta_vals)
theta_phase = nap.Tsd(t=lfp.index.values, d=theta_phase_vals)
theta_amp = nap.Tsd(t=lfp.index.values, d=theta_amp_vals)

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
plot_win = nap.IntervalSet(maze_start + 100, maze_start + 103)
raw_seg = lfp.restrict(plot_win)
filt_seg = lfp_theta.restrict(plot_win)
phase_seg = theta_phase.restrict(plot_win)
axes[0].plot(raw_seg.index.values, raw_seg.values, color="0.4", lw=0.8, label="raw LFP")
axes[0].plot(filt_seg.index.values, filt_seg.values, color="crimson", lw=1.5, label="6-10 Hz theta")
axes[0].set_ylabel("LFP (a.u.)")
axes[0].set_title(f"Raw vs. theta-band-filtered LFP (channel {THETA_CHANNEL}, 3 s example)")
axes[0].legend(loc="upper right")
axes[1].plot(phase_seg.index.values, phase_seg.values, color="darkblue", lw=1.2)
axes[1].set_ylabel("Theta phase (rad)")
axes[1].set_xlabel("Time (s)")
axes[1].set_yticks([-np.pi, 0, np.pi])
axes[1].set_yticklabels(["-π", "0", "π"])
plt.tight_layout()
plt.savefig("fig1_raw_lfp_theta_filter.png", dpi=150)
plt.close()
print("Saved fig1_raw_lfp_theta_filter.png")

# %% [markdown]
# ## 3. Running Epochs from Linearized Position
#
# Theta is a locomotion-associated rhythm: it is strong and consistent while
# the animal runs, and weak or absent during immobility. We compute running
# speed from the linearized position and keep only sustained epochs of
# locomotion above 3 cm/s for the phase-locking analysis, following standard
# practice in the rodent theta literature.

# %%
# Smoothing and the speed derivative are computed per tracked bout so that
# the untracked gaps between bouts never contribute a spurious speed value.
pos_smooth = position.smooth(std=0.1, windowsize=1.0)
speed_t_list, speed_d_list = [], []
for s, e in zip(tracked_ep.start, tracked_ep.end):
    bout = pos_smooth.get(s, e)
    if len(bout) < 2:
        continue
    v = np.abs(np.gradient(bout.values, bout.index.values)) * 100.0  # cm/s
    speed_t_list.append(bout.index.values)
    speed_d_list.append(v)
speed = nap.Tsd(t=np.concatenate(speed_t_list), d=np.concatenate(speed_d_list), time_support=tracked_ep)
speed_smooth = speed.smooth(std=0.25, windowsize=2.0)

SPEED_THRESH = 3.0  # cm/s
MIN_RUN_DUR = 1.0  # s
running_ep = speed_smooth.threshold(SPEED_THRESH).time_support
running_ep = running_ep.drop_short_intervals(MIN_RUN_DUR)
total_run = running_ep.tot_length()
print(f"Running epochs: {len(running_ep)} epochs, {total_run:.1f} s total "
      f"({100 * total_run / (maze_stop - maze_start):.1f}% of MazeEpoch)")

# %%
def gap_broken_xy(tsd, ep, gap_thresh=0.5):
    """Return (t, d) with NaNs inserted at inter-epoch gaps so plt.plot doesn't
    draw spurious straight lines across untracked periods between bouts."""
    ts, ds = [], []
    for s, e in zip(ep.start, ep.end):
        seg = tsd.get(s, e)
        ts.append(seg.index.values)
        ds.append(seg.values)
        ts.append([seg.index.values[-1] + gap_thresh] if len(seg) else [])
        ds.append([np.nan])
    return np.concatenate(ts), np.concatenate(ds)


pos_t_plot, pos_d_plot = gap_broken_xy(position, tracked_ep)
speed_t_plot, speed_d_plot = gap_broken_xy(speed_smooth, tracked_ep)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(pos_t_plot, pos_d_plot, color="0.3", lw=0.8)
for s, e in zip(running_ep.start, running_ep.end):
    axes[0].axvspan(s, e, color="tab:green", alpha=0.3)
axes[0].set_ylabel("Linearized position (m)")
axes[0].set_title("Linearized position and speed with detected running epochs (green);\n"
                   "gaps = periods the tracker lost the animal")
axes[1].plot(speed_t_plot, speed_d_plot, color="0.3", lw=0.6)
axes[1].axhline(SPEED_THRESH, color="tab:red", ls="--", lw=1, label=f"{SPEED_THRESH} cm/s threshold")
for s, e in zip(running_ep.start, running_ep.end):
    axes[1].axvspan(s, e, color="tab:green", alpha=0.3)
axes[1].set_ylabel("Speed (cm/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(0, np.nanpercentile(speed_smooth.values, 99.5))
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig2_running_epochs.png", dpi=150)
plt.close()
print("Saved fig2_running_epochs.png")

# %%
fig, ax = plt.subplots(figsize=(10, 4))
zoom = nap.IntervalSet(running_ep.start[5] - 5, running_ep.start[5] + 20)
pt, pd_ = gap_broken_xy(position, tracked_ep.intersect(zoom))
ax.plot(pt, pd_, color="0.2", lw=1.2)
for s, e in zip(running_ep.intersect(zoom).start, running_ep.intersect(zoom).end):
    ax.axvspan(s, e, color="tab:green", alpha=0.3)
ax.set_ylabel("Linearized position (m)")
ax.set_xlabel("Time (s)")
ax.set_title("Zoomed-in view: individual track traversals and detected running epochs (green)")
plt.tight_layout()
plt.savefig("fig2b_running_epochs_zoom.png", dpi=150)
plt.close()
print("Saved fig2b_running_epochs_zoom.png")

# %% [markdown]
# ## 4. Spike-Theta Phase Locking
#
# For every unit, we take the spikes fired during running and assign each
# one the instantaneous theta phase (interpolated from the Hilbert-phase
# Tsd) at which it occurred. We then test each unit's spike-phase
# distribution for non-uniformity with a Rayleigh test and compute the mean
# resultant length (MRL), a continuous 0-1 measure of how tightly spikes
# cluster around a preferred phase.

# %%
def rayleigh_test(phases):
    n = len(phases)
    if n == 0:
        return np.nan, np.nan, np.nan
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C ** 2 + S ** 2)
    mrl = R / n
    pref_phase = np.arctan2(S, C)
    z = n * mrl ** 2
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) -
                       (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    p = np.clip(p, 0, 1)
    return mrl, pref_phase, p


units_running = units.restrict(running_ep)
theta_phase_running = theta_phase.restrict(running_ep)

results = []
spike_phases_by_unit = {}
for u in tqdm(units.index, desc="Computing spike-theta phases"):
    spk = units_running[u]
    if len(spk) == 0:
        continue
    ph = spk.value_from(theta_phase_running).values
    spike_phases_by_unit[u] = ph
    mrl, pref, p = rayleigh_test(ph)
    results.append({
        "unit": u, "cell_type": units.get_info("cell_type")[u],
        "location": units.get_info("location")[u], "n_spikes": len(spk),
        "mrl": mrl, "pref_phase": pref, "rayleigh_p": p,
        "firing_rate_running": len(spk) / running_ep.tot_length(),
    })

results_df = pd.DataFrame(results)
results_df["significant"] = results_df["rayleigh_p"] < 0.001
print(results_df.groupby("cell_type")[["mrl", "significant"]].agg(["mean", "sum", "count"]))

# %% [markdown]
# ## 5. Example Units: Polar Phase Histograms
#
# We pick three illustrative units directly from the Rayleigh-test results:
# the most strongly entrained excitatory (pyramidal) cell, a non-entrained
# excitatory cell (highest p-value with reasonable spike count), and the
# most strongly entrained inhibitory (interneuron) cell.

# %%
exc = results_df[results_df.cell_type == "excitatory"]
inh = results_df[results_df.cell_type == "inhibitory"]

best_exc_unit = exc[exc["n_spikes"] >= 50].sort_values("mrl", ascending=False).iloc[0]["unit"]
nonentrained_candidates = exc[exc["n_spikes"] >= 100].sort_values("rayleigh_p", ascending=False)
nonentrained_exc_unit = nonentrained_candidates.iloc[0]["unit"]
best_inh_unit = inh.loc[inh["mrl"].idxmax(), "unit"]

example_units = [
    (best_exc_unit, "Strongly entrained pyramidal cell"),
    (nonentrained_exc_unit, "Non-entrained pyramidal cell"),
    (best_inh_unit, "Strongly entrained interneuron"),
]
print("Example units:", example_units)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "polar"})
for ax, (u, title) in zip(axes, example_units):
    ph = spike_phases_by_unit[u]
    row = results_df[results_df.unit == u].iloc[0]
    ax.hist(ph, bins=24, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_theta_zero_location("N")
    p_str = f"{row.rayleigh_p:.1e}" if row.rayleigh_p > 1e-300 else "<1e-300"
    ax.set_title(f"{title}\nunit {u}, n={row.n_spikes} spikes\nMRL={row.mrl:.2f}, p={p_str}",
                 fontsize=10, pad=32)
plt.tight_layout()
plt.savefig("fig3_polar_phase_histograms.png", dpi=150)
plt.close()
print("Saved fig3_polar_phase_histograms.png")

# %% [markdown]
# ## 6. Population Summary
#
# We summarize entrainment strength across the population: MRL
# distributions split by cell type, the fraction of significantly
# phase-locked cells in each group, the relationship between MRL and
# firing rate (to check entrainment isn't simply an artifact of spike
# count), and each significantly-locked unit's preferred phase.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
for ct, color in [("excitatory", "tab:blue"), ("inhibitory", "tab:red")]:
    vals = results_df[results_df.cell_type == ct]["mrl"]
    ax.hist(vals, bins=15, alpha=0.6, color=color, label=f"{ct} (n={len(vals)})", density=True)
ax.set_xlabel("Mean resultant length (MRL)")
ax.set_ylabel("Density")
ax.set_title("Entrainment strength by cell type")
ax.legend()

ax = axes[0, 1]
frac_sig = results_df.groupby("cell_type")["significant"].mean()
n_cells = results_df.groupby("cell_type")["significant"].count()
bars = ax.bar(frac_sig.index, frac_sig.values, color=["tab:blue", "tab:red"])
for b, ct in zip(bars, frac_sig.index):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{frac_sig[ct]*100:.0f}%\n(n={n_cells[ct]})", ha="center", fontsize=9)
ax.set_ylabel("Fraction significantly phase-locked\n(Rayleigh p < 0.001)")
ax.set_ylim(0, 1.15)
ax.set_title("Significantly theta-locked cells")

ax = axes[1, 0]
for ct, color in [("excitatory", "tab:blue"), ("inhibitory", "tab:red")]:
    sub = results_df[results_df.cell_type == ct]
    ax.scatter(sub["firing_rate_running"], sub["mrl"], color=color, alpha=0.7, label=ct, s=30)
ax.set_xscale("log")
ax.set_xlabel("Firing rate while running (Hz)")
ax.set_ylabel("MRL")
ax.set_title("Entrainment strength vs. firing rate")
ax.legend()

ax = axes[1, 1]
ax.remove()
ax = fig.add_subplot(2, 2, 4, projection="polar")
sig = results_df[results_df.significant]
for ct, color in [("excitatory", "tab:blue"), ("inhibitory", "tab:red")]:
    sub = sig[sig.cell_type == ct]
    ax.scatter(sub["pref_phase"], sub["mrl"], color=color, alpha=0.8, label=ct, s=40)
ax.set_theta_zero_location("N")
ax.set_title("Preferred phase (significant units)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
plt.savefig("fig4_population_entrainment_summary.png", dpi=150)
plt.close()
print("Saved fig4_population_entrainment_summary.png")

# %% [markdown]
# ## 7. Phase-Free Cross-Checks: Spike-Triggered Average and Autocorrelogram
#
# The Hilbert-phase analysis above depends on a specific filtering pipeline,
# so as an independent, phase-free confirmation we compute two classic
# signatures of theta-rhythmic spiking for the same example units: the
# spike-triggered average (STA) of the *raw, unfiltered* LFP (an entrained
# cell's spikes should sit atop several cycles of a ~125-150 ms oscillation)
# and each unit's own spike-time autocorrelogram (an entrained cell's own
# spike train should show side peaks at the theta period).

# %%
def spike_triggered_average(spike_times, lfp_tsd, win=0.4, fs=FS_LFP):
    n_win = int(round(win * fs))
    vals = lfp_tsd.values
    t = lfp_tsd.index.values
    idx = np.searchsorted(t, spike_times)
    sta_stack = []
    for i in idx:
        if n_win <= i < len(vals) - n_win:
            sta_stack.append(vals[i - n_win:i + n_win + 1])
    sta_stack = np.array(sta_stack)
    lags = (np.arange(-n_win, n_win + 1) / fs) * 1000.0  # ms
    return lags, sta_stack.mean(axis=0), sta_stack.shape[0]


def spike_autocorrelogram(spike_times, win=0.4, binsize=0.01):
    bins = np.arange(-win, win + binsize, binsize)
    all_lags = []
    for i in range(len(spike_times)):
        lo = np.searchsorted(spike_times, spike_times[i] - win)
        hi = np.searchsorted(spike_times, spike_times[i] + win)
        diffs = spike_times[lo:hi] - spike_times[i]
        all_lags.append(diffs[diffs != 0])
    all_lags = np.concatenate(all_lags) if all_lags else np.array([])
    counts, edges = np.histogram(all_lags, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2 * 1000.0  # ms
    return centers, counts


lfp_raw_running = lfp.restrict(running_ep)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for col, (u, title) in enumerate(example_units):
    spk_t = units_running[u].index.values
    lags_ms, sta, n_sta = spike_triggered_average(spk_t, lfp_raw_running)
    ax = axes[0, col]
    ax.plot(lags_ms, sta, color="darkorange", lw=1.5)
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    ax.set_title(f"{title}\nSTA of raw LFP (n={n_sta} spikes)", fontsize=10)
    ax.set_xlabel("Lag from spike (ms)")
    ax.set_ylabel("LFP (a.u.)")

    lags_acg, counts = spike_autocorrelogram(spk_t)
    ax = axes[1, col]
    ax.bar(lags_acg, counts, width=9, color="teal")
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    for shift in [-150, -125, 125, 150]:
        ax.axvline(shift, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax.set_title("Spike-time autocorrelogram", fontsize=10)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Spike count")
plt.tight_layout()
plt.savefig("fig5_sta_and_autocorrelogram.png", dpi=150)
plt.close()
print("Saved fig5_sta_and_autocorrelogram.png")

# %% [markdown]
# ## 8. Theta Amplitude vs. Running Speed
#
# As a final sanity check that the 6-10 Hz oscillation being analyzed is
# genuine locomotion-associated theta (rather than a filtering artifact),
# we relate theta-band LFP amplitude (Hilbert envelope) to running speed:
# theta power is well established to increase with running speed in rodents.

# %%
theta_amp_running = theta_amp.restrict(running_ep)
speed_at_amp = theta_amp_running.value_from(speed_smooth)
amp_vals = theta_amp_running.values
speed_vals_at_amp = speed_at_amp.values

# bin by speed for a clean summary curve, plus raw scatter for correlation
speed_bins = np.arange(3, 60, 3)
bin_idx = np.digitize(speed_vals_at_amp, speed_bins)
bin_centers, bin_means, bin_sems = [], [], []
for b in range(1, len(speed_bins)):
    sel = bin_idx == b
    if sel.sum() > 20:
        bin_centers.append((speed_bins[b - 1] + speed_bins[b]) / 2)
        bin_means.append(amp_vals[sel].mean())
        bin_sems.append(amp_vals[sel].std() / np.sqrt(sel.sum()))

r = np.corrcoef(speed_vals_at_amp, amp_vals)[0, 1]
print(f"Pearson r (theta amplitude vs. speed) = {r:.3f}, n = {len(amp_vals)} samples")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(speed_vals_at_amp[::20], amp_vals[::20], s=3, alpha=0.15, color="0.3")
axes[0].set_xlabel("Running speed (cm/s)")
axes[0].set_ylabel("Theta amplitude (Hilbert envelope, a.u.)")
axes[0].set_title(f"Raw samples (subsampled), Pearson r = {r:.2f}")

axes[1].errorbar(bin_centers, bin_means, yerr=bin_sems, marker="o", color="darkgreen", capsize=3)
axes[1].set_xlabel("Running speed bin center (cm/s)")
axes[1].set_ylabel("Mean theta amplitude ± SEM")
axes[1].set_title("Theta amplitude vs. speed (binned)")
plt.tight_layout()
plt.savefig("fig6_theta_amplitude_vs_speed.png", dpi=150)
plt.close()
print("Saved fig6_theta_amplitude_vs_speed.png")

# %% [markdown]
# ## 9. Summary
#
# Print a compact summary of the key numbers referenced in the README.

# %%
print("\n=== SUMMARY ===")
print(f"Dataset: DANDI 000044, sub-Gatsby, ses-Gatsby-08022013")
print(f"Theta channel: {THETA_CHANNEL} (theta power fraction {theta_fracs[THETA_CHANNEL]:.2f})")
print(f"Running epochs: {len(running_ep)}, {running_ep.tot_length():.1f} s total")
print(f"Units tested: {len(results_df)} ({(results_df.cell_type=='excitatory').sum()} excitatory, "
      f"{(results_df.cell_type=='inhibitory').sum()} inhibitory)")
for ct in ["excitatory", "inhibitory"]:
    sub = results_df[results_df.cell_type == ct]
    print(f"  {ct}: {sub['significant'].sum()}/{len(sub)} significant "
          f"({100*sub['significant'].mean():.1f}%), mean MRL = {sub['mrl'].mean():.3f}")
print(f"Theta amplitude vs. speed: Pearson r = {r:.3f} (n={len(amp_vals)})")
