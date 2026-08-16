# %% [markdown]
# # Head Direction Cells in the Anterodorsal Thalamus and Postsubiculum
#
# This notebook demonstrates the classic head-direction (HD) cell phenomenon using
# freely-moving-mouse electrophysiology data from the DANDI Archive.
#
# **Dataset**: [DANDI:000056](https://dandiarchive.org/dandiset/000056) — "Internally
# organized mechanisms of the head direction sense" (Peyrache & Buzsaki lab). Mice were
# implanted with silicon probes in the antero-dorsal thalamic nucleus (ADn) and the
# post-subiculum (PoS) and allowed to forage freely in a small open arena while an
# overhead camera tracked two LEDs mounted on the headstage. The vector between the
# two LEDs gives the instantaneous head direction of the animal.
#
# **Session used**: `sub-Mouse24/sub-Mouse24_ses-Mouse24-131213_behavior+ecephys.nwb`
#
# We will:
# 1. Stream the NWB file directly from S3 (no full download) and inspect its contents.
# 2. Compute head direction from the two tracked LEDs.
# 3. Compute occupancy-normalized directional tuning curves for every recorded unit.
# 4. Identify head-direction cells using tuning strength (Rayleigh vector) and
#    split-half reliability.
# 5. Decode head direction from the identified HD-cell ensemble using Bayesian
#    population decoding, and quantify decoding accuracy.

# %% [markdown]
# ## Setup and Data Loading

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

np.random.seed(0)

S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/f00/e5c/"
    "f00e5c3a-9435-42df-aace-9b6952563479"
)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)

# %% [markdown]
# The file contains spiking data for 22 units (`units`), local field potentials
# (`LFP`), scored behavioral states (`states`: Awake / Non-REM / REM), and tracked
# x/y positions of two head-mounted LEDs (`RedLED`, `BlueLED`) sampled at ~39 Hz.

# %%
print(nwbfile.session_description)
print(f"Subject: {nwbfile.subject.subject_id}, species: {nwbfile.subject.species}")

units = nwb["units"]
red = nwb["RedLED"]
blue = nwb["BlueLED"]
states = nwb["states"]

print(f"\nNumber of units: {len(units)}")
print(f"Position samples: {len(red)}")

# %% [markdown]
# ## Inspect Raw Data Streams
#
# Before any processing, we look at the raw behavioral states, the raw LED tracking
# traces, and a raw spike raster to confirm the data look sensible.

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Behavioral states timeline
state_colors = {"Awake": "tab:orange", "Non-REM": "tab:blue", "REM": "tab:green"}
for lab in state_colors:
    sub_ep = states[states.label == lab]
    for s, e in zip(sub_ep.start, sub_ep.end):
        axes[0].axvspan(s, e, color=state_colors[lab], alpha=0.6)
axes[0].set_xlim(states.start[0], states.end[-1])
axes[0].set_yticks([])
axes[0].set_xlabel("Time (s)")
axes[0].set_title("Scored behavioral states across the recording")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in state_colors.values()]
axes[0].legend(handles, state_colors.keys(), loc="upper right", ncol=3)

# Raw LED tracking traces (short window)
window = nap.IntervalSet(start=200, end=260)
red_w = red.restrict(window)
blue_w = blue.restrict(window)
axes[1].plot(red_w.t, red_w.values[:, 0], color="firebrick", label="Red LED - x")
axes[1].plot(blue_w.t, blue_w.values[:, 0], color="steelblue", label="Blue LED - x")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("x position (pixels)")
axes[1].set_title("Raw LED tracking (60 s window)")
axes[1].legend(loc="upper right")

# Raw spike raster for a handful of units over the same window
for i, u in enumerate(list(units.keys())[:8]):
    sp = units[u].restrict(window)
    axes[2].vlines(sp.t, i, i + 0.8, color="k", linewidth=0.5)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Unit #")
axes[2].set_title("Raw spike raster, 8 example units (60 s window)")

plt.tight_layout()
plt.savefig("fig01_raw_data_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## Compute Head Direction from the Two LEDs
#
# Head direction is the angle of the vector pointing from the red LED to the blue
# LED. Frames where either LED was not detected are coded as `(-1, -1)` in the raw
# data and are dropped.

# %%
red_v = red.values
blue_v = blue.values
valid = (red_v[:, 0] != -1) & (blue_v[:, 0] != -1)
print(f"Fraction of tracked frames with both LEDs detected: {valid.mean():.3f}")

dx = blue_v[:, 0] - red_v[:, 0]
dy = blue_v[:, 1] - red_v[:, 1]
angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
angle[~valid] = np.nan

hd_full = nap.Tsd(t=red.t, d=angle)

# %% [markdown]
# We restrict all further analysis to the longest continuous "Awake" epoch, which
# corresponds to the main open-field foraging session (the animal is also recorded
# during sleep, which is not usable for computing directional tuning).

# %%
awake_ep = states[states.label == "Awake"]
durations = awake_ep.end - awake_ep.start
main_idx = int(np.argmax(durations))
foraging_ep = awake_ep[main_idx : main_idx + 1]
print(f"Foraging epoch: {foraging_ep.start[0]:.0f}s - {foraging_ep.end[0]:.0f}s "
      f"({(foraging_ep.end[0] - foraging_ep.start[0]) / 60:.1f} min)")

hd = hd_full.restrict(foraging_ep).dropna()
units_f = units.restrict(foraging_ep)

# %%
fig, ax = plt.subplots(figsize=(12, 3))
hd_w = hd.restrict(nap.IntervalSet(start=200, end=260))
ax.plot(hd_w.t, np.degrees(hd_w.values), ".", markersize=2, color="darkviolet")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Head direction (deg)")
ax.set_title("Computed head direction, 60 s example window")
plt.tight_layout()
plt.savefig("fig02_head_direction_trace.png", dpi=150)
plt.close()

# %% [markdown]
# ## Directional Tuning Curves
#
# For every unit we compute an occupancy-normalized firing-rate tuning curve as a
# function of head direction (60 bins covering 0-360 deg).

# %%
tc = nap.compute_tuning_curves(
    units_f, hd, bins=60, range=[(0, 2 * np.pi)], epochs=foraging_ep,
    return_pandas=True,
)
angles = tc.index.values
print(f"Tuning curves shape: {tc.shape} (angle bins x units)")


def rayleigh_vector(rates, angles):
    """Mean resultant vector length and angle of a circular tuning curve."""
    z = np.sum(rates * np.exp(1j * angles))
    r = np.abs(z) / np.sum(rates)
    theta = np.mod(np.angle(z), 2 * np.pi)
    return r, theta


rayleigh_r = {}
preferred_dir = {}
for u in tc.columns:
    r, theta = rayleigh_vector(tc[u].values, angles)
    rayleigh_r[u] = r
    preferred_dir[u] = theta

# %% [markdown]
# ## Split-Half Reliability
#
# A genuine directional tuning curve should be stable across independent halves of
# the recording. We split the foraging epoch in half, recompute tuning curves on
# each half independently, and correlate them per unit.

# %%
t_mid = foraging_ep.start[0] + (foraging_ep.end[0] - foraging_ep.start[0]) / 2
ep1 = nap.IntervalSet(start=foraging_ep.start[0], end=t_mid)
ep2 = nap.IntervalSet(start=t_mid, end=foraging_ep.end[0])

tc1 = nap.compute_tuning_curves(
    units_f, hd.restrict(ep1), bins=60, range=[(0, 2 * np.pi)], epochs=ep1,
    return_pandas=True,
)
tc2 = nap.compute_tuning_curves(
    units_f, hd.restrict(ep2), bins=60, range=[(0, 2 * np.pi)], epochs=ep2,
    return_pandas=True,
)

split_half_r = {}
for u in tc.columns:
    a, b = tc1[u].values, tc2[u].values
    if np.std(a) > 0 and np.std(b) > 0:
        split_half_r[u] = np.corrcoef(a, b)[0, 1]
    else:
        split_half_r[u] = np.nan

# %% [markdown]
# We define a unit as a head-direction cell if its full-session Rayleigh vector
# length exceeds 0.3 **and** its split-half tuning-curve correlation exceeds 0.5.
# This combines a criterion for tuning sharpness with a criterion for temporal
# stability, guarding against spurious tuning from limited sampling.

# %%
hd_cell_ids = sorted(
    u for u in tc.columns
    if rayleigh_r[u] > 0.3 and (split_half_r[u] > 0.5)
)
print(f"Identified {len(hd_cell_ids)} head-direction cells: {hd_cell_ids}")

summary_rows = []
for u in tc.columns:
    summary_rows.append(
        (u, units_f[u].rate, rayleigh_r[u], split_half_r[u], u in hd_cell_ids)
    )
print(f"{'unit':>4} {'rate(Hz)':>9} {'Rayleigh r':>11} {'split-half r':>13} {'HD cell':>8}")
for row in sorted(summary_rows, key=lambda r: -r[2]):
    print(f"{row[0]:>4} {row[1]:>9.2f} {row[2]:>11.3f} {row[3]:>13.3f} {str(row[4]):>8}")

# %% [markdown]
# ## Polar Tuning Curves
#
# Every unit's tuning curve plotted in polar coordinates. Head-direction cells
# (identified above) are highlighted in red and show a single sharp firing-rate peak
# at a preferred direction; non-HD units are flat or noisy.

# %%
n_units = len(tc.columns)
n_cols = 6
n_rows = int(np.ceil(n_units / n_cols))
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), subplot_kw={"projection": "polar"}
)
axes = axes.flatten()
for i, u in enumerate(tc.columns):
    ax = axes[i]
    is_hd = u in hd_cell_ids
    color = "crimson" if is_hd else "gray"
    ax.plot(angles, tc[u].values, color=color, linewidth=2)
    ax.fill(angles, tc[u].values, color=color, alpha=0.25)
    ax.set_title(f"unit {u}\nR={rayleigh_r[u]:.2f}", fontsize=10, pad=12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
for j in range(n_units, len(axes)):
    axes[j].axis("off")
fig.suptitle(
    "Directional tuning curves for all recorded units\n(red = identified head-direction cells)",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("fig03_polar_tuning_curves.png", dpi=150)
plt.close()

# %% [markdown]
# ## Tuning Strength Across the Population
#
# A scatter of Rayleigh vector length against split-half reliability shows that the
# HD-cell criterion isolates a distinct cluster of units that are both sharply and
# stably tuned, well separated from the rest of the population.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

r_vals = np.array([rayleigh_r[u] for u in tc.columns])
sh_vals = np.array([split_half_r[u] for u in tc.columns])
is_hd_arr = np.array([u in hd_cell_ids for u in tc.columns])

axes[0].scatter(r_vals[~is_hd_arr], sh_vals[~is_hd_arr], color="gray", label="other units")
axes[0].scatter(r_vals[is_hd_arr], sh_vals[is_hd_arr], color="crimson", label="HD cells")
axes[0].axvline(0.3, color="k", linestyle="--", linewidth=1)
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1)
axes[0].set_xlabel("Rayleigh vector length (tuning sharpness)")
axes[0].set_ylabel("Split-half tuning-curve correlation")
axes[0].set_title("HD-cell identification criteria")
axes[0].legend(loc="lower right")

axes[1].hist(r_vals, bins=15, color="slategray", edgecolor="white")
axes[1].axvline(0.3, color="k", linestyle="--", linewidth=1)
axes[1].set_xlabel("Rayleigh vector length")
axes[1].set_ylabel("Number of units")
axes[1].set_title("Distribution of tuning strength across all units")

plt.tight_layout()
plt.savefig("fig04_tuning_strength_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## Population Raster Sorted by Preferred Direction
#
# When the head-direction ensemble's spikes are sorted by each unit's preferred
# direction and plotted alongside the animal's instantaneous heading, the units
# fire in a sequence that tracks the heading trace, and the identity of the
# active unit(s) changes systematically as the animal turns.

# %%
sorted_hd_ids = sorted(hd_cell_ids, key=lambda u: preferred_dir[u])
window = nap.IntervalSet(start=300, end=360)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})
for i, u in enumerate(sorted_hd_ids):
    sp = units_f[u].restrict(window)
    axes[0].vlines(sp.t, i, i + 0.8, color="crimson", linewidth=1.2)
axes[0].set_yticks(np.arange(len(sorted_hd_ids)) + 0.4)
axes[0].set_yticklabels(
    [f"u{u} ({np.degrees(preferred_dir[u]):.0f} deg)" for u in sorted_hd_ids]
)
axes[0].set_ylabel("HD cell (sorted by pref. dir.)")
axes[0].set_title("Head-direction ensemble raster vs. instantaneous heading (60 s window)")

hd_win = hd.restrict(window)
axes[1].plot(hd_win.t, np.degrees(hd_win.values), color="darkviolet", linewidth=1)
for u in sorted_hd_ids:
    axes[1].axhline(np.degrees(preferred_dir[u]), color="crimson", linestyle=":", linewidth=0.8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Heading (deg)")

plt.tight_layout()
plt.savefig("fig05_ensemble_raster_vs_heading.png", dpi=150)
plt.close()

# %% [markdown]
# ## Bayesian Decoding of Head Direction from the Ensemble
#
# As a final, quantitative demonstration, we fit tuning curves for the identified
# HD cells on the first half of the foraging epoch and use Bayesian population
# decoding (`pynapple.decode_bayes`) to reconstruct the animal's head direction from
# spiking activity alone in the held-out second half.

# %%
sub_group = units_f[hd_cell_ids]
tc_fit = nap.compute_tuning_curves(
    sub_group, hd.restrict(ep1), bins=60, range=[(0, 2 * np.pi)], epochs=ep1
)

bin_size = 0.2
decoded, proba = nap.decode_bayes(tc_fit, sub_group, epochs=ep2, bin_size=bin_size)
actual = hd.restrict(ep2).bin_average(bin_size)

circ_err = np.abs(np.angle(np.exp(1j * (decoded.values - actual.values))))
valid_err = circ_err[~np.isnan(circ_err)]
median_err_deg = np.degrees(np.median(valid_err))
print(f"Decoding bin size: {bin_size * 1000:.0f} ms")
print(f"Median absolute circular decoding error: {median_err_deg:.1f} deg "
      f"(chance level ~ 90 deg)")

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

decode_window = nap.IntervalSet(start=ep2.start[0] + 100, end=ep2.start[0] + 160)
dec_w = decoded.restrict(decode_window)
act_w = actual.restrict(decode_window)
axes[0].plot(act_w.t, np.degrees(act_w.values), color="darkviolet", label="actual heading", linewidth=1.5)
axes[0].plot(dec_w.t, np.degrees(dec_w.values), ".", color="crimson", label="decoded heading", markersize=4)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Head direction (deg)")
axes[0].set_title(f"Bayesian-decoded vs. actual head direction (held-out data, {bin_size*1000:.0f} ms bins)")
axes[0].legend(loc="upper right")

axes[1].hist(np.degrees(valid_err), bins=30, color="slategray", edgecolor="white")
axes[1].axvline(median_err_deg, color="crimson", linestyle="--",
                label=f"median = {median_err_deg:.0f} deg")
axes[1].axvline(90, color="k", linestyle=":", label="chance level (90 deg)")
axes[1].set_xlabel("Absolute circular decoding error (deg)")
axes[1].set_ylabel("Number of time bins")
axes[1].set_title("Decoding error distribution")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("fig06_decoding_performance.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# Out of 22 recorded units in this session, a subset showed sharp, stable
# directional tuning curves (high Rayleigh vector length, high split-half
# reliability) whose preferred directions tile the full range of possible headings.
# Population activity from these head-direction cells alone was sufficient to
# reconstruct the animal's instantaneous heading well above chance using simple
# Bayesian decoding, confirming that this ensemble carries a reliable, continuously
# updated internal compass signal consistent with the classic head-direction cell
# phenomenon described in the anterodorsal thalamus and postsubiculum.

# %%
io.close()
print("Done.")
