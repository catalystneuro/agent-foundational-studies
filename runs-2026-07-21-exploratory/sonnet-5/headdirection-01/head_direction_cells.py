# %% [markdown]
# # Head Direction Cells in the Anterodorsal Thalamus / Postsubiculum
#
# This notebook demonstrates head-direction (HD) tuning in extracellular recordings
# from the antero-dorsal thalamic nucleus and postsubiculum of a freely moving mouse.
#
# **Dataset**: DANDI Archive Dandiset
# [000056](https://dandiarchive.org/dandiset/000056) - "Internally organized
# mechanisms of the head direction sense" (Peyrache lab). We use session
# `sub-Mouse20/sub-Mouse20_ses-Mouse20-130514_behavior+ecephys.nwb`, which contains
# multi-shank spike-sorted units together with two head-mounted LED trackers (red and
# blue) and brain-state annotations (Awake / Non-REM / REM).
#
# **Approach**:
# 1. Stream the NWB file directly from S3 (no full download) using `remfile` +
#    `pynapple`.
# 2. Reconstruct the animal's instantaneous head direction from the vector between
#    the two LEDs.
# 3. Compute circular tuning curves for every recorded unit during open-field
#    exploration (the longest continuous "Awake" bout).
# 4. Use a circular-shift shuffle test to identify units whose tuning is
#    significantly non-uniform (i.e. head-direction cells).
# 5. Demonstrate that the identified HD-cell ensemble carries enough information to
#    decode the animal's head direction from spiking activity alone (Bayesian
#    decoding on held-out data).

# %% [markdown]
# ## Setup

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

np.random.seed(0)  # for reproducibility of the (non-cryptographic) shuffle test
rng = np.random.default_rng(42)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# %% [markdown]
# ## 1. Load the NWB File via Streaming (remfile)
#
# Dandiset 000056, asset
# `sub-Mouse20/sub-Mouse20_ses-Mouse20-130514_behavior+ecephys.nwb`. The file is
# ~1.9 GB; we stream it with `remfile` and a local disk cache rather than downloading
# it in full.

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/e00/73f/"
    "e0073f32-8048-4048-9fb1-269501ce0aa1"
)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)

# %% [markdown]
# The file contains a `TsGroup` of 17 spike-sorted units, an `IntervalSet` of
# brain states, LFP, and two `TsdFrame` position trackers (`RedLED`, `BlueLED`)
# giving the (x, y) pixel position of each head-mounted diode over time.

# %%
units = nwb["units"]
states = nwb["states"]
red_led = nwb["RedLED"]
blue_led = nwb["BlueLED"]

print(f"Number of units: {len(units)}")
print(f"Brain states present: {np.unique(states.label)}")
print(f"LED sampling interval (s): {np.median(np.diff(red_led.t[:1000])):.4f}")

# %% [markdown]
# ## 2. Select the Main Open-Field Exploration Epoch
#
# The "Awake" state is split into many short bouts (grooming, brief waking during
# sleep, etc.) plus one long open-field exploration session. We use the single
# longest continuous "Awake" bout as our behavioral epoch for tuning-curve
# estimation and decoding.

# %%
wake_ep = states[states.label == "Awake"]
durations = wake_ep.end - wake_ep.start
longest_idx = np.argmax(durations)
main_wake = nap.IntervalSet(
    start=wake_ep.start[longest_idx], end=wake_ep.end[longest_idx]
)

print(
    f"Longest continuous wake bout: {main_wake.tot_length():.0f} s "
    f"({main_wake.tot_length()/60:.1f} min), "
    f"from t={main_wake.start[0]:.0f}s to t={main_wake.end[0]:.0f}s"
)

# %% [markdown]
# ## 3. Compute Head Direction from the Two LED Trackers
#
# The animal's head direction is reconstructed as the angle of the vector pointing
# from the red LED to the blue LED. Missing tracking samples are coded as (-1, -1)
# in this dataset and are excluded.

# %%
red_w = red_led.restrict(main_wake)
blue_w = blue_led.restrict(main_wake)

valid = (
    (red_w[:, 0].values > 0)
    & (red_w[:, 1].values > 0)
    & (blue_w[:, 0].values > 0)
    & (blue_w[:, 1].values > 0)
)
print(f"Fraction of samples with valid tracking: {valid.mean():.3f}")

dx = blue_w[:, 0].values - red_w[:, 0].values
dy = blue_w[:, 1].values - red_w[:, 1].values
angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)

hd = nap.Tsd(t=red_w.t[valid], d=angle[valid])
hd_ep = hd.time_support  # epoch with valid, contiguous tracking

print(hd)

# %% [markdown]
# ### Raw Data Validation
#
# Before any analysis, we visualize the raw tracked position and the derived head
# direction to confirm the reconstruction is sensible.

# %%
window = nap.IntervalSet(start=hd_ep.start[0], end=hd_ep.start[0] + 60)
red_win = red_led.restrict(window)
blue_win = blue_led.restrict(window)
hd_win = hd.restrict(window)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(red_win.t, red_win[:, 0].values, label="Red LED x", lw=0.8)
axes[0].plot(red_win.t, red_win[:, 1].values, label="Red LED y", lw=0.8)
axes[0].plot(blue_win.t, blue_win[:, 0].values, label="Blue LED x", lw=0.8, ls="--")
axes[0].plot(blue_win.t, blue_win[:, 1].values, label="Blue LED y", lw=0.8, ls="--")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Pixel position")
axes[0].set_title("Raw LED tracking (60 s window)")
axes[0].legend(fontsize=7, loc="upper right")

sc = axes[1].scatter(
    red_win[:, 0].values, red_win[:, 1].values, c=red_win.t, cmap="viridis", s=4
)
axes[1].set_xlabel("x (pixels)")
axes[1].set_ylabel("y (pixels)")
axes[1].set_title("Trajectory (color = time)")
plt.colorbar(sc, ax=axes[1], label="Time (s)", fraction=0.046)

axes[2].plot(hd_win.t, np.degrees(hd_win.values), lw=0.8, color="darkorange")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Head direction (deg)")
axes[2].set_title("Derived head-direction angle")
axes[2].set_ylim(0, 360)

plt.tight_layout()
plt.savefig("fig1_raw_data_validation.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Compute Circular Tuning Curves
#
# For each unit we compute a firing-rate tuning curve as a function of head
# direction, using 60 bins (6 degrees each) over the full main wake epoch.

# %%
units_w = units.restrict(hd_ep)
N_BINS = 60

tuning_curves = nap.compute_tuning_curves(
    units_w, hd, bins=N_BINS, range=[(0, 2 * np.pi)], epochs=hd_ep
)
bin_centers = tuning_curves.coords[list(tuning_curves.coords)[1]].values

print(tuning_curves)

# %% [markdown]
# ## 5. Identify Head-Direction Cells with a Circular-Shift Shuffle Test
#
# A unit is considered directionally tuned if its observed mean vector length (MVL,
# the length of the circular-mean firing-rate vector, i.e. a Rayleigh-style
# statistic) exceeds what is expected by chance. We build a null distribution per
# unit by circularly shifting its spike train by a random offset (uniform over the
# epoch duration, avoiding the first/last 20 s to prevent trivial near-zero shifts)
# and recomputing the MVL, repeated 500 times. This preserves the autocorrelation
# and firing rate of the spike train while destroying its relationship to head
# direction, which is the standard control for this kind of tuning test.
#
# A unit is classified as a head-direction cell if `p < 0.05` **and** `MVL > 0.15`.
# The MVL threshold guards against very high-firing-rate units (e.g. fast-spiking
# interneurons) that reach nominal significance from sample size alone despite
# negligible directional modulation.

# %%
dt = np.median(np.diff(hd.t))
occupancy, edges = np.histogram(hd.values, bins=N_BINS, range=(0, 2 * np.pi))
occupancy_time = occupancy * dt


def mean_vector_length(spike_angles):
    counts, _ = np.histogram(spike_angles, bins=edges)
    fr = counts / occupancy_time
    if fr.sum() == 0:
        return np.nan
    c = np.sum(fr * np.cos(bin_centers))
    s = np.sum(fr * np.sin(bin_centers))
    return np.sqrt(c**2 + s**2) / np.sum(fr)


N_SHUFFLE = 500
start_t, end_t = hd_ep.start[0], hd_ep.end[0]
total_t = end_t - start_t

results = {}
for u in units_w.index:
    spk = units_w[u]
    if len(spk) < 20:
        results[u] = dict(mvl=np.nan, p=np.nan, n_spikes=len(spk), rate=0.0)
        continue

    spike_angles = spk.value_from(hd).values
    obs_mvl = mean_vector_length(spike_angles)

    spk_t = spk.t
    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        shift = rng.uniform(20, total_t - 20)
        shifted_t = start_t + np.mod(spk_t - start_t + shift, total_t)
        shifted_ts = nap.Ts(t=np.sort(shifted_t), time_support=hd_ep)
        null[i] = mean_vector_length(shifted_ts.value_from(hd).values)

    p_value = (np.sum(null >= obs_mvl) + 1) / (N_SHUFFLE + 1)
    results[u] = dict(
        mvl=obs_mvl, p=p_value, n_spikes=len(spk), rate=len(spk) / total_t
    )

is_hd_cell = {
    u: (r["p"] < 0.05 and r["mvl"] > 0.15) if not np.isnan(r["mvl"]) else False
    for u, r in results.items()
}
hd_cell_ids = sorted([u for u, v in is_hd_cell.items() if v])

print("Unit  rate(Hz)  MVL     p-value   HD cell?")
for u, r in results.items():
    mvl_str = f"{r['mvl']:.3f}" if not np.isnan(r["mvl"]) else "  nan"
    p_str = f"{r['p']:.4f}" if not np.isnan(r["p"]) else "   nan"
    print(f"{u:>4}  {r['rate']:>7.2f}  {mvl_str:>6}  {p_str:>8}  {is_hd_cell[u]}")

print(f"\nHead-direction cells identified: {hd_cell_ids}")

# %% [markdown]
# ## 6. Polar Tuning Curves for All Units

# %%
n_units = len(units_w)
n_cols = 5
n_rows = int(np.ceil(n_units / n_cols))

fig = plt.figure(figsize=(3.0 * n_cols, 3.2 * n_rows))
for i, u in enumerate(units_w.index):
    ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
    fr = tuning_curves.sel(unit=u).values
    fr = np.nan_to_num(fr)
    theta = np.append(bin_centers, bin_centers[0])
    r = np.append(fr, fr[0])
    color = "crimson" if is_hd_cell[u] else "steelblue"
    ax.plot(theta, r, color=color, lw=1.5)
    ax.fill(theta, r, color=color, alpha=0.25)
    ax.set_theta_zero_location("N")
    r_info = results[u]
    mvl_str = f"{r_info['mvl']:.2f}" if not np.isnan(r_info["mvl"]) else "nan"
    ax.set_title(
        f"Unit {u} ({r_info['rate']:.1f} Hz)\nMVL={mvl_str}"
        + (" *" if is_hd_cell[u] else ""),
        fontsize=9,
        pad=14,
    )
    ax.set_yticklabels([])
    ax.tick_params(labelsize=7)

fig.suptitle(
    "Head-direction tuning curves for all recorded units\n"
    "(red = classified head-direction cell, * = p<0.05 & MVL>0.15)",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.savefig("fig2_polar_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 7. Summary of Directional Tuning Strength Across the Population

# %%
sorted_units = sorted(results.keys(), key=lambda u: -1 if np.isnan(results[u]["mvl"]) else results[u]["mvl"])
sorted_units = [u for u in sorted_units if not np.isnan(results[u]["mvl"])]

fig, ax = plt.subplots(figsize=(8, 5))
mvls = [results[u]["mvl"] for u in sorted_units]
colors = ["crimson" if is_hd_cell[u] else "steelblue" for u in sorted_units]
bars = ax.bar(range(len(sorted_units)), mvls, color=colors)
ax.set_xticks(range(len(sorted_units)))
ax.set_xticklabels([f"unit {u}" for u in sorted_units], rotation=45, ha="right")
ax.axhline(0.15, color="k", ls="--", lw=1, label="MVL threshold (0.15)")
ax.set_ylabel("Mean vector length (MVL)")
ax.set_title("Directional tuning strength by unit\n(red = significant head-direction cell)")
ax.legend()
plt.tight_layout()
plt.savefig("fig3_mvl_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## 8. Population Raster: Spiking Tracks Head Direction
#
# For the identified head-direction cells, we plot spike rasters (sorted by
# preferred direction) alongside the simultaneously recorded head-direction trace.
# If these are genuine HD cells, each unit should fire preferentially when the
# animal's head direction trace passes through its preferred angle.

# %%
preferred_angle = {}
for u in hd_cell_ids:
    fr = np.nan_to_num(tuning_curves.sel(unit=u).values)
    c = np.sum(fr * np.cos(bin_centers))
    s = np.sum(fr * np.sin(bin_centers))
    preferred_angle[u] = np.mod(np.arctan2(s, c), 2 * np.pi)

hd_cell_ids_sorted = sorted(hd_cell_ids, key=lambda u: preferred_angle[u])

raster_window = nap.IntervalSet(start=hd_ep.start[0] + 100, end=hd_ep.start[0] + 160)
hd_raster = hd.restrict(raster_window)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
)

ax1.plot(hd_raster.t, np.degrees(hd_raster.values), color="darkorange", lw=1.2)
ax1.set_ylabel("Head\ndirection (deg)")
ax1.set_ylim(0, 360)
ax1.set_title("Population raster of head-direction cells vs. simultaneous head direction")

for i, u in enumerate(hd_cell_ids_sorted):
    spk = units_w[u].restrict(raster_window)
    ax2.vlines(spk.t, i, i + 0.8, color="crimson", lw=1.2)

ax2.set_yticks(np.arange(len(hd_cell_ids_sorted)) + 0.4)
ax2.set_yticklabels(
    [f"unit {u}\n(pref={np.degrees(preferred_angle[u]):.0f} deg)" for u in hd_cell_ids_sorted],
    fontsize=8,
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Head-direction cells\n(sorted by preferred angle)")

plt.tight_layout()
plt.savefig("fig4_population_raster.png", dpi=150)
plt.close()

# %% [markdown]
# ## 9. Decoding Head Direction from the HD-Cell Ensemble
#
# As a final demonstration, we test whether the identified head-direction cells
# carry enough information to reconstruct the animal's instantaneous head direction
# from spiking activity alone. We fit tuning curves on the first 70% of the main
# wake epoch (train) and perform Bayesian decoding (`nap.decode_bayes`) on the held
# out final 30% (test), so decoding accuracy is evaluated on data not used to build
# the tuning curves.

# %%
split_t = start_t + 0.7 * total_t
train_ep = nap.IntervalSet(start=start_t, end=split_t)
test_ep = nap.IntervalSet(start=split_t, end=end_t)

hd_units_group = units_w[hd_cell_ids]
hd_train = hd.restrict(train_ep)

train_tuning_curves = nap.compute_tuning_curves(
    hd_units_group, hd_train, bins=N_BINS, range=[(0, 2 * np.pi)],
    epochs=hd_train.time_support,
)

decoded, proba = nap.decode_bayes(
    train_tuning_curves, hd_units_group, epochs=test_ep, bin_size=0.3
)
actual_at_decoded_times = decoded.value_from(hd)

circ_error = np.angle(np.exp(1j * (decoded.values - actual_at_decoded_times.values)))
median_error_deg = np.degrees(np.median(np.abs(circ_error)))
mean_error_deg = np.degrees(np.mean(np.abs(circ_error)))

print(f"Decoding using {len(hd_cell_ids)} head-direction cells: {hd_cell_ids}")
print(f"Median absolute circular decoding error: {median_error_deg:.1f} deg")
print(f"Mean absolute circular decoding error:   {mean_error_deg:.1f} deg")
print(f"(chance level for a uniform random guess is 90 deg)")

# %%
decode_window = nap.IntervalSet(start=test_ep.start[0], end=test_ep.start[0] + 120)
decoded_win = decoded.restrict(decode_window)
actual_win = hd.restrict(decode_window)

fig, axes = plt.subplots(2, 1, figsize=(11, 6))

axes[0].plot(actual_win.t, np.degrees(actual_win.values), label="Actual head direction", color="darkorange", lw=1.2)
axes[0].scatter(decoded_win.t, np.degrees(decoded_win.values), label="Bayesian decoded", color="crimson", s=6)
axes[0].set_ylabel("Head direction (deg)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylim(0, 360)
axes[0].set_title(f"Held-out decoding from {len(hd_cell_ids)} head-direction cells (120 s test window)")
axes[0].legend(loc="upper right", fontsize=8)

axes[1].hist(np.degrees(np.abs(circ_error)), bins=36, color="steelblue", edgecolor="white")
axes[1].axvline(median_error_deg, color="crimson", ls="--", label=f"median = {median_error_deg:.0f} deg")
axes[1].axvline(90, color="k", ls=":", label="chance level (90 deg)")
axes[1].set_xlabel("Absolute circular decoding error (deg)")
axes[1].set_ylabel("Count (test bins)")
axes[1].set_title("Distribution of decoding error across the held-out test epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("fig5_decoding.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# Out of 17 recorded units, we identified a small ensemble of units (see the
# printed table above) with firing rates that vary systematically and
# significantly with the animal's head direction, well beyond what circular-shift
# shuffling of their own spike trains produces by chance. These head-direction
# cells fire in a narrow angular range (visible as sharp, unimodal peaks in the
# polar tuning curves), and their combined population activity is sufficient to
# reconstruct the animal's instantaneous head direction on held-out data far better
# than chance (90 degrees expected under a uniform random guess). This reproduces
# the classic head-direction cell phenomenon described in the antero-dorsal
# thalamus / postsubiculum circuit (Peyrache et al.).

# %%
io.close()
