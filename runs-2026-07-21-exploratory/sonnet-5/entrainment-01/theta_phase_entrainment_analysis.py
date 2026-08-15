# %% [markdown]
# # Theta Phase Entrainment of Hippocampal Neurons
#
# This notebook demonstrates theta phase entrainment: the tendency of hippocampal
# neurons to fire preferentially at a particular phase of the ~6-10 Hz theta
# oscillation that dominates the local field potential (LFP) during active
# locomotion (Buzsaki, 2002; O'Keefe & Recce, 1993). We test this directly with
# real single-unit and LFP recordings, and additionally compare entrainment
# strength between putative pyramidal cells and interneurons, since a
# well-established finding in the CA1 literature is that fast-spiking
# interneurons tend to be more strongly and consistently theta-locked than
# principal cells (Csicsvari et al., 1999; Klausberger & Somogyi, 2008).
#
# ## Dataset
#
# [DANDI 000044](https://dandiarchive.org/dandiset/000044) ("Diversity in
# neural firing dynamics supports both rigid and learned hippocampal
# sequences", Grosmark & Buzsaki, NYU), subject **Achilles**, session
# `Achilles_10252013`: bilateral CA1 silicon-probe recordings (137 units: 120
# putative pyramidal/excitatory cells and 17 putative interneurons/inhibitory
# cells) while the rat shuttled back and forth on a 1.6 m linear track for
# reward. The NWB file is streamed directly from S3 with `remfile` (disk-cached
# locally, no full download), and Pynapple is used for all time-series handling
# and analysis.
#
# ## Approach
#
# 1. Extract instantaneous theta phase from one CA1 pyramidal-layer LFP channel
#    (6-10 Hz bandpass + Hilbert transform).
# 2. Identify locomotion epochs from linearized position (theta is weak or
#    absent during immobility, so entrainment is tested only while the animal
#    is actively running).
# 3. For every unit, assign each spike the instantaneous theta phase at which it
#    occurred and test for non-uniformity of the spike-phase distribution with
#    a Rayleigh test, using the mean resultant length (MRL) as a continuous
#    measure of entrainment strength.
# 4. Cross-check the phase-based result with two phase-free signatures of theta
#    entrainment: the spike-triggered average LFP (does the raw LFP oscillate
#    around a cell's spikes?) and the spike-time autocorrelogram (do a cell's
#    own spikes show ~125 ms rhythmicity?).
# 5. Compare entrainment strength between pyramidal cells and interneurons at
#    the population level.

# %%
import pickle

import h5py
import remfile
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

nap.nap_config.suppress_conversion_warnings = True
rng = np.random.default_rng(0)

URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
CACHE_DIR = "nwb_cache"
THETA_CHANNEL = 69  # CA1 pyramidal-layer shank; gives a clean 6-10 Hz theta rhythm (see Fig 1)
FS_LFP = 1250.0

# %% [markdown]
# ## 1. Load Data
#
# We stream only the pieces needed directly from the underlying HDF5/NWB
# structure (rather than the whole ~GB file): the epoch table, one LFP channel
# over the maze-running epoch, the linearized position, and all unit spike
# times restricted to the maze epoch.

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

print(f"\n{len(units)} units total: {meta_df['cell_type'].value_counts().to_dict()}")
print(f"By hemisphere: {meta_df['location'].value_counts().to_dict()}")

# %% [markdown]
# ## 2. Theta Filtering and Instantaneous Phase
#
# The LFP is band-pass filtered at 6-10 Hz (4th-order Butterworth, applied
# zero-phase with `filtfilt`), and the Hilbert transform gives the
# instantaneous theta phase and amplitude at every LFP sample. A short
# raw-vs-filtered overlay confirms the filter correctly isolates the theta
# rhythm before its phase is used for anything downstream.

# %%
nyq = FS_LFP / 2
b, a = butter(4, [6 / nyq, 10 / nyq], btype="band")
theta_filt = filtfilt(b, a, lfp.values)
analytic = hilbert(theta_filt)
theta_phase = np.mod(np.angle(analytic), 2 * np.pi)  # 0..2*pi, 0 = trough of the filtered wave
theta_amp = np.abs(analytic)
theta_phase_tsd = nap.Tsd(t=lfp.t, d=theta_phase)
theta_amp_tsd = nap.Tsd(t=lfp.t, d=theta_amp)
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
plt.close(fig)

# %% [markdown]
# ## 3. Locomotion Epochs
#
# Theta is a locomotion-associated rhythm: it is strong and regular while the
# rat runs and largely absent during immobility. We compute running speed from
# a lightly smoothed linearized position trace and keep only sustained
# (>1 s) epochs where |speed| exceeds 3 cm/s, merging brief gaps. All
# entrainment statistics below are computed on spikes restricted to these
# epochs.

# %%
pos_smooth = position.smooth(std=0.3, windowsize=1.5)
dt = np.diff(pos_smooth.t)
dpos = np.diff(pos_smooth.values)
vel_t = pos_smooth.t[:-1] + dt / 2
velocity = nap.Tsd(t=vel_t, d=np.abs(dpos / dt))

SPEED_THR = 0.03  # m/s
MIN_DUR = 1.0
moving = velocity.threshold(SPEED_THR, method="above").time_support
moving = moving.merge_close_intervals(0.2)
moving = moving[(moving.end - moving.start) > MIN_DUR]
print(f"{len(moving)} running epochs, total duration {np.sum(moving.end - moving.start):.1f} s "
      f"({100 * np.sum(moving.end - moving.start) / (maze_stop - maze_start):.0f}% of MazeEpoch)")

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(position.t, position.values, lw=0.5, color="0.3")
for s, e in zip(moving.start, moving.end):
    axes[0].axvspan(s, e, color="C0", alpha=0.3, lw=0)
axes[0].set_ylabel("position (m)")
axes[0].set_title("Linearized position across MazeEpoch (shaded = running epochs used for entrainment analysis)")
axes[1].plot(velocity.t, velocity.values, lw=0.3, color="0.4")
axes[1].axhline(SPEED_THR, color="C3", lw=0.8, ls="--", label=f"threshold ({SPEED_THR} m/s)")
axes[1].set_ylabel("|speed| (m/s)")
axes[1].set_xlabel("time (s)")
axes[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig("fig2_running_epochs.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Spike-Phase Analysis: Rayleigh Test and Mean Resultant Length
#
# For each unit, every spike fired during a running epoch is assigned the
# instantaneous theta phase at that moment (nearest-sample lookup). If a
# cell's firing is theta-entrained, its spike phases will cluster around a
# preferred phase rather than being uniformly distributed around the circle.
# We test this per unit with a Rayleigh test for circular non-uniformity, and
# summarize entrainment strength with the mean resultant length (MRL, 0 for a
# uniform/non-entrained distribution, up to 1 for spikes at a single exact
# phase).


# %%
def rayleigh_test(phases):
    """Rayleigh test for circular non-uniformity (Zar, 1999, Ch. 27).

    Returns (mean resultant length, p-value, preferred phase in [0, 2*pi)).
    """
    n = len(phases)
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C ** 2 + S ** 2) / n
    z = n * R ** 2
    p = np.exp(-z) * (
        1 + (2 * z - z ** 2) / (4 * n)
        - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)
    )
    p = float(np.clip(p, 0, 1))
    mean_angle = float(np.arctan2(S, C) % (2 * np.pi))
    return float(R), p, mean_angle


MIN_SPIKES = 20
run_duration = np.sum(moving.end - moving.start)
spike_phases = {}
results = []
for uid in units.index:
    st = units[uid].restrict(moving)
    if len(st) < MIN_SPIKES:
        continue
    ph = st.value_from(theta_phase_tsd).values
    R, p, mean_angle = rayleigh_test(ph)
    spike_phases[uid] = ph
    results.append((uid, units.metadata.loc[uid, "cell_type"], len(st), len(st) / run_duration, R, p, mean_angle))

res_df = pd.DataFrame(
    results, columns=["id", "cell_type", "n_spikes", "rate_hz", "MRL", "p", "mean_phase"]
).set_index("id")
res_df["significant"] = res_df["p"] < 0.001
print(f"{len(res_df)}/{len(units)} units had >= {MIN_SPIKES} spikes during running and were tested.")
print(res_df.groupby("cell_type")[["MRL", "significant"]].mean())
print(res_df.sort_values("MRL", ascending=False).head(8))

# %% [markdown]
# ## 5. Example Units: Polar Phase Histograms
#
# We look at three representative units: the most strongly theta-locked
# pyramidal cell, a pyramidal cell with a non-significant (near-uniform) phase
# distribution, and the most strongly theta-locked interneuron. Polar
# histograms of spike phase make the entrainment (or lack of it) visually
# obvious before summarizing it statistically at the population level.

# %%
# among well-sampled units (>=500 spikes) so the example STA/ACG are not noise-dominated
strong_pyr_id = res_df[(res_df.cell_type == "excitatory") & (res_df.n_spikes >= 500)].sort_values(
    "MRL", ascending=False
).index[0]
weak_pyr_id = res_df[(res_df.cell_type == "excitatory") & (~res_df.significant)].sort_values(
    "n_spikes", ascending=False
).index[0]
strong_inh_id = res_df[res_df.cell_type == "inhibitory"].sort_values("MRL", ascending=False).index[0]
example_ids = [strong_pyr_id, weak_pyr_id, strong_inh_id]
example_labels = [
    f"pyramidal cell {strong_pyr_id}\n(strongly entrained)",
    f"pyramidal cell {weak_pyr_id}\n(not entrained)",
    f"interneuron {strong_inh_id}\n(strongly entrained)",
]
print("Example unit ids:", example_ids)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), subplot_kw={"projection": "polar"})
n_bins = 24
bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
for ax, uid, label in zip(axes, example_ids, example_labels):
    ph = spike_phases[uid]
    counts, _ = np.histogram(ph, bins=bin_edges)
    width = 2 * np.pi / n_bins
    ax.bar(bin_edges[:-1], counts, width=width, align="edge", color="C0", edgecolor="white", alpha=0.85)
    row = res_df.loc[uid]
    p_str = "p < 1e-300" if row.p == 0 else f"p = {row.p:.2g}"
    ax.set_title(f"{label}\nMRL = {row.MRL:.2f}, {p_str}\n(n={int(row.n_spikes)} spikes)", fontsize=9, pad=18)
    ax.plot([row.mean_phase, row.mean_phase], [0, counts.max()], color="C3", lw=2)
    ax.set_yticklabels([])
fig.suptitle("Spike-theta-phase distributions for example units", y=1.05)
fig.tight_layout()
fig.savefig("fig3_polar_phase_histograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Population Summary
#
# Across the population, we ask: (a) how is entrainment strength (MRL)
# distributed for pyramidal cells vs. interneurons, (b) what fraction of each
# group is significantly theta-locked, (c) does entrainment strength depend on
# a cell's overall firing rate, and (d) what preferred phases do entrained
# cells fire at.

# %%
fig = plt.figure(figsize=(11, 9))
axes = np.array([
    [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)],
    [fig.add_subplot(2, 2, 3), None],
])

# (a) MRL distributions
ax = axes[0, 0]
bins = np.linspace(0, res_df.MRL.max() * 1.05, 25)
for ct, color in [("excitatory", "C0"), ("inhibitory", "C3")]:
    sub = res_df[res_df.cell_type == ct]
    ax.hist(sub.MRL, bins=bins, alpha=0.6, color=color, label=f"{ct} (n={len(sub)})", density=True)
ax.set_xlabel("mean resultant length (MRL)")
ax.set_ylabel("density")
ax.set_title("Theta phase-locking strength by cell type")
ax.legend()

# (b) fraction significant
ax = axes[0, 1]
frac_sig = res_df.groupby("cell_type")["significant"].mean()
counts_ct = res_df.groupby("cell_type").size()
bars = ax.bar(frac_sig.index, frac_sig.values, color=["C0", "C3"])
for b, ct in zip(bars, frac_sig.index):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{frac_sig[ct]:.0%}\n(n={counts_ct[ct]})", ha="center", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("fraction significantly theta-locked (p < 0.001)")
ax.set_title("Fraction of units significantly phase-locked")

# (c) MRL vs firing rate
ax = axes[1, 0]
for ct, color in [("excitatory", "C0"), ("inhibitory", "C3")]:
    sub = res_df[res_df.cell_type == ct]
    ax.scatter(sub.rate_hz, sub.MRL, s=25, alpha=0.7, color=color, label=ct)
ax.set_xscale("log")
ax.set_xlabel("firing rate during running (Hz, log scale)")
ax.set_ylabel("MRL")
ax.set_title("Entrainment strength vs. firing rate")
ax.legend()

# (d) preferred phase of significantly entrained units
ax = fig.add_subplot(2, 2, 4, projection="polar")
sig = res_df[res_df.significant]
for ct, color in [("excitatory", "C0"), ("inhibitory", "C3")]:
    sub = sig[sig.cell_type == ct]
    ax.scatter(sub.mean_phase, sub.MRL, s=25, alpha=0.7, color=color, label=ct)
ax.set_title("Preferred phase of significantly-locked units\n(angle = phase, radius = MRL)", fontsize=9, pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

fig.tight_layout()
fig.savefig("fig4_population_entrainment_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 7. Phase-Free Cross-Checks: Spike-Triggered LFP and Autocorrelograms
#
# The Rayleigh test relies on the Hilbert-transform phase estimate. As an
# independent, phase-free confirmation of entrainment, we compute (a) the
# spike-triggered average (STA) of the *raw* LFP around each example unit's
# spikes, which should oscillate at theta frequency if the cell is
# theta-entrained, and (b) each unit's own spike-time autocorrelogram, which
# should show a trough-peak-trough pattern with ~125 ms (8 Hz) spacing if the
# cell's spiking itself is theta-rhythmic.

# %%
sta = nap.compute_spike_triggered_average(
    lfp, units[example_ids], binsize=1 / FS_LFP, window=(-0.3, 0.3), epochs=moving
)
acg = nap.compute_autocorrelogram(units[example_ids], binsize=0.005, windowsize=0.5, ep=moving)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, (uid, label) in enumerate(zip(example_ids, example_labels)):
    axes[0, i].plot(sta.t * 1000, sta.loc[uid].values, color="C2", lw=1.5)
    axes[0, i].axvline(0, color="k", lw=0.7, ls="--")
    axes[0, i].set_title(f"STA of raw LFP\n{label}", fontsize=9)
    axes[0, i].set_xlabel("time from spike (ms)")
    if i == 0:
        axes[0, i].set_ylabel("LFP (a.u.)")

    axes[1, i].bar(acg.index * 1000, acg[uid].values, width=5.0, color="C4")
    axes[1, i].axvline(0, color="k", lw=0.7, ls="--")
    axes[1, i].set_title("spike-time autocorrelogram", fontsize=9)
    axes[1, i].set_xlabel("lag (ms)")
    if i == 0:
        axes[1, i].set_ylabel("rate (normalized to mean)")

fig.suptitle("Phase-free confirmation of theta entrainment", y=1.02)
fig.tight_layout()
fig.savefig("fig5_sta_and_autocorrelogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 8. Theta Amplitude Increases with Running Speed
#
# As a final, complementary check that this is genuine locomotion-associated
# theta (rather than an artifact of the filtering/phase pipeline), we confirm
# the classic relationship that theta-band LFP amplitude increases with
# running speed.

# %%
velocity_run = velocity.restrict(moving)
theta_amp_at_vel = velocity_run.value_from(theta_amp_tsd)  # theta amplitude sampled at each speed timestamp
speed_vals = velocity_run.values
amp_vals = theta_amp_at_vel.values
corr = np.corrcoef(speed_vals, amp_vals)[0, 1]

speed_bins = np.quantile(speed_vals, np.linspace(0, 1, 11))
bin_idx = np.digitize(speed_vals, speed_bins[1:-1])
bin_centers, bin_means, bin_sems = [], [], []
for b in range(10):
    sel = bin_idx == b
    if sel.sum() > 0:
        bin_centers.append(speed_vals[sel].mean())
        bin_means.append(amp_vals[sel].mean())
        bin_sems.append(amp_vals[sel].std() / np.sqrt(sel.sum()))

fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, marker="o", color="C1", capsize=3)
ax.set_xlabel("running speed (m/s)")
ax.set_ylabel("theta-band amplitude (a.u.)")
ax.set_title(f"Theta amplitude increases with running speed\n(Pearson r = {corr:.2f}, n = {len(speed_vals)} samples)")
fig.tight_layout()
fig.savefig("fig6_theta_amplitude_vs_speed.png", dpi=150)
plt.close(fig)

print(f"Speed-theta amplitude correlation: r = {corr:.3f}")

# %% [markdown]
# ## Results Summary
#
# Print a final summary of the key numbers reported in the README.

# %%
n_pyr = (res_df.cell_type == "excitatory").sum()
n_inh = (res_df.cell_type == "inhibitory").sum()
frac_sig_pyr = res_df[res_df.cell_type == "excitatory"].significant.mean()
frac_sig_inh = res_df[res_df.cell_type == "inhibitory"].significant.mean()
mrl_pyr = res_df[res_df.cell_type == "excitatory"].MRL.mean()
mrl_inh = res_df[res_df.cell_type == "inhibitory"].MRL.mean()

print(f"Pyramidal cells tested: {n_pyr}, significantly theta-locked: {frac_sig_pyr:.0%}, mean MRL: {mrl_pyr:.3f}")
print(f"Interneurons tested: {n_inh}, significantly theta-locked: {frac_sig_inh:.0%}, mean MRL: {mrl_inh:.3f}")
print(f"Speed-theta amplitude correlation: r = {corr:.3f}")
