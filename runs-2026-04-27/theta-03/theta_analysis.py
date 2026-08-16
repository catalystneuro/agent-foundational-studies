# %% [markdown]
# # Theta Phase Entrainment and Phase Precession in Hippocampal Place Cells
#
# This notebook demonstrates two hallmark phenomena of hippocampal CA1 pyramidal
# cells in a freely-behaving rat running on a linear track:
#
# 1. **Theta entrainment**: spikes from CA1 pyramidal neurons preferentially occur
#    at a specific phase of the local 6–10 Hz theta rhythm.
# 2. **Theta phase precession**: as the animal traverses a cell's place field, the
#    spikes occur at progressively earlier phases of theta — a tight intra-cell
#    coupling between position and timing within the theta cycle.
#
# **Dataset.** DANDI:000044 (Grosmark & Buzsáki, *Science* 2016) — bilateral
# silicon-probe CA1 recordings while rats run a 1.6-m linear track. We use a
# single session from rat *Achilles* (`Achilles_10252013`).
#
# **Pipeline.**
# 1. Stream the NWB file from S3 with `remfile` + disk cache.
# 2. Extract spikes, position, and the LFP for the maze epoch.
# 3. Pick the LFP channel with the strongest theta power.
# 4. Compute running speed; restrict the analysis to active running.
# 5. Build 1D place fields for every CA1 pyramidal cell, separately for each
#    running direction.
# 6. Theta entrainment: compare the phase distribution of spikes during running
#    against a uniform null (Rayleigh test).
# 7. Phase precession: for every place cell, regress spike phase on normalized
#    in-field position using a circular-linear regression and plot the result.

# %%
import os
import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert, welch
from tqdm import tqdm

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load NWB file (streaming)

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
CACHE_DIR = "/tmp/remfile_cache_theta"
os.makedirs(CACHE_DIR, exist_ok=True)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
print("Subject:", nwbfile.subject.subject_id, "| Session:", nwbfile.session_id)

epochs = nwbfile.epochs.to_dataframe()
print("\nEpochs:")
print(epochs)

maze_row = epochs[epochs.label == "MazeEpoch"].iloc[0]
maze_start, maze_stop = float(maze_row.start_time), float(maze_row.stop_time)
print(f"\nMazeEpoch: {maze_start:.1f}s – {maze_stop:.1f}s ({maze_stop-maze_start:.1f}s)")

# %% [markdown]
# ## 2. Extract spikes (CA1 pyramidal cells)

# %%
udf = nwbfile.units.to_dataframe()
print("Units:", len(udf), "| cell_type:", udf.cell_type.value_counts().to_dict(),
      "| location:", udf.location.value_counts().to_dict())

spike_dict = {}
cell_meta = {}
for uid, row in udf.iterrows():
    if row.cell_type != "excitatory":
        continue  # restrict to putative pyramidal cells
    st = np.asarray(row.spike_times)
    spike_dict[int(uid)] = nap.Ts(t=st)
    cell_meta[int(uid)] = {"location": row.location, "shank_id": int(row.shank_id)}

spikes = nap.TsGroup(spike_dict)
spikes.set_info(location=np.array([cell_meta[i]["location"] for i in spikes.keys()]))
spikes.set_info(shank_id=np.array([cell_meta[i]["shank_id"] for i in spikes.keys()]))
print(f"\nKept {len(spikes)} putative CA1 pyramidal cells")

# %% [markdown]
# ## 3. Position and running speed
#
# In this NWB file the `rate` field of the SpatialSeries actually stores the
# sampling *period* (≈25.6 ms ≈ 39 Hz). We reconstruct the time vector from the
# starting time and the period, drop NaNs, and wrap into a Pynapple `Tsd`.

# %%
linpos_obj = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"]["1.6mLinearMazeLinearizedTimeSeries"]
period = float(linpos_obj.rate)            # seconds per sample (mis-named field)
fs_pos = 1.0 / period                       # ~39 Hz
t0 = float(linpos_obj.starting_time)
pos_data = linpos_obj.data[:].squeeze().astype(float)
pos_t = t0 + np.arange(pos_data.size) * period
valid = ~np.isnan(pos_data)
position = nap.Tsd(t=pos_t[valid], d=pos_data[valid])
print(f"Position: {len(position)} samples @ {fs_pos:.2f} Hz, "
      f"{position.t.min():.1f}–{position.t.max():.1f}s, range {position.values.min():.2f}–{position.values.max():.2f} m")

# Running speed via numerical derivative, smoothed with a 250 ms Gaussian
dx = np.gradient(position.values, position.t)
speed_raw = np.abs(dx)                                              # m/s, unsigned
from scipy.ndimage import gaussian_filter1d
speed_smooth = gaussian_filter1d(speed_raw, sigma=int(0.25 * fs_pos))
speed = nap.Tsd(t=position.t, d=speed_smooth)

# Signed velocity (for direction)
vel = nap.Tsd(t=position.t, d=gaussian_filter1d(dx, sigma=int(0.25 * fs_pos)))
print(f"Speed range: {speed.values.min():.3f}–{speed.values.max():.3f} m/s, median {np.median(speed.values):.3f}")

# %% [markdown]
# Build epochs of sustained running (≥5 cm/s for ≥0.5 s).

# %%
SPEED_THRESH = 0.05  # m/s
running_mask = speed.values > SPEED_THRESH
# Convert mask to IntervalSet by finding rising/falling edges
edges = np.diff(running_mask.astype(int))
starts = np.where(edges == 1)[0] + 1
stops = np.where(edges == -1)[0] + 1
if running_mask[0]:
    starts = np.r_[0, starts]
if running_mask[-1]:
    stops = np.r_[stops, len(running_mask) - 1]
run_t_starts = speed.t[starts]
run_t_stops = speed.t[stops]
keep = (run_t_stops - run_t_starts) >= 0.5
run_ep = nap.IntervalSet(start=run_t_starts[keep], end=run_t_stops[keep])
print(f"Running epochs: {len(run_ep)} intervals, total {run_ep.tot_length():.1f}s")

# Direction-specific epochs (using signed velocity sampled inside running)
RIGHT_THRESH, LEFT_THRESH = 0.05, -0.05
right_mask = vel.values > RIGHT_THRESH
left_mask = vel.values < LEFT_THRESH

def mask_to_intervals(mask, t):
    e = np.diff(mask.astype(int))
    s = np.where(e == 1)[0] + 1
    f = np.where(e == -1)[0] + 1
    if mask[0]:
        s = np.r_[0, s]
    if mask[-1]:
        f = np.r_[f, len(mask) - 1]
    return nap.IntervalSet(start=t[s], end=t[f])

right_ep = mask_to_intervals(right_mask, vel.t).intersect(run_ep)
left_ep = mask_to_intervals(left_mask, vel.t).intersect(run_ep)
print(f"Right runs: {right_ep.tot_length():.1f}s | Left runs: {left_ep.tot_length():.1f}s")

# %% [markdown]
# Quick visualization of position trace + running speed.

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
sample_t = (position.t > maze_start) & (position.t < maze_start + 200)
axes[0].plot(position.t[sample_t], position.values[sample_t], "k-", lw=0.8)
axes[0].set_ylabel("Linearized\nposition (m)")
axes[0].set_title(f"First 200 s of MazeEpoch — {nwbfile.session_id}")
axes[1].plot(speed.t[sample_t], speed.values[sample_t], "b-", lw=0.6)
axes[1].axhline(SPEED_THRESH, color="r", ls="--", label=f"{SPEED_THRESH*100:.0f} cm/s threshold")
axes[1].set_ylabel("Speed (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_behavior_traces.png"), dpi=130)
plt.close()
print("Saved 01_behavior_traces.png")

# %% [markdown]
# ## 4. LFP — pick the channel with strongest theta power
#
# The maze epoch covers samples `maze_start*fs_lfp` to `maze_stop*fs_lfp`. We
# probe one channel per shank during running, compute a Welch PSD, and select
# the channel whose 6–10 Hz theta band stands out the most relative to a
# 2–4 Hz low and 12–20 Hz high band.

# %%
lfp_es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs_lfp = float(lfp_es.rate)
i0 = int(maze_start * fs_lfp)
i1 = int(maze_stop * fs_lfp)
print(f"LFP fs={fs_lfp:.0f} Hz, maze samples {i0}–{i1} ({(i1-i0)/fs_lfp:.0f}s)")

edf = lfp_es.electrodes.table.to_dataframe()
# Pick one electrode per shank (the middle of each shank)
candidate_idx = []
for shank, group in edf.groupby("group_name"):
    candidate_idx.append(group.index[len(group)//2])
candidate_idx = sorted(candidate_idx)
print(f"Candidate channels (one/shank): {candidate_idx}")

theta_scores = {}
for ch in tqdm(candidate_idx, desc="Probing channels"):
    x = lfp_es.data[i0:i1, ch].astype(np.float32) * lfp_es.conversion  # volts
    # Use a shorter slice for speed estimation: first 200 s
    snippet = x[: int(200 * fs_lfp)]
    f, p = welch(snippet, fs=fs_lfp, nperseg=int(4 * fs_lfp))
    theta_p = p[(f >= 6) & (f <= 10)].mean()
    flank = p[((f >= 2) & (f <= 4)) | ((f >= 12) & (f <= 20))].mean()
    theta_scores[ch] = theta_p / flank
print("\nTheta-to-flank ratio per probed channel:")
for ch, s in sorted(theta_scores.items(), key=lambda x: -x[1])[:6]:
    print(f"  ch {ch:3d}: {s:.3f}")

best_ch = max(theta_scores, key=theta_scores.get)
print(f"\nSelected LFP channel: {best_ch}")

# %% [markdown]
# Load the full maze-epoch LFP for the selected channel.

# %%
print("Loading full maze LFP for selected channel ...")
lfp_data = lfp_es.data[i0:i1, best_ch].astype(np.float32) * lfp_es.conversion
lfp_t = i0 / fs_lfp + np.arange(lfp_data.size) / fs_lfp
lfp = nap.Tsd(t=lfp_t, d=lfp_data)
print(f"LFP loaded: {len(lfp)} samples, {lfp.t.min():.1f}–{lfp.t.max():.1f}s")

# %% [markdown]
# ### Theta-band filter and Hilbert phase

# %%
def bandpass(x, fs, low, high, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x)

THETA_LOW, THETA_HIGH = 6.0, 10.0
theta_filt = bandpass(lfp.values, fs_lfp, THETA_LOW, THETA_HIGH)
analytic = hilbert(theta_filt)
theta_phase = np.angle(analytic)        # radians, in [-pi, pi]
theta_amp = np.abs(analytic)
theta_phase_tsd = nap.Tsd(t=lfp.t, d=theta_phase)
theta_amp_tsd = nap.Tsd(t=lfp.t, d=theta_amp)

# Sanity-check plot: 5 s of raw LFP, theta-filtered, and phase
t_demo0, t_demo1 = maze_start + 50, maze_start + 55
m = (lfp.t >= t_demo0) & (lfp.t <= t_demo1)
fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
ax[0].plot(lfp.t[m], lfp.values[m] * 1e3, "k", lw=0.7)
ax[0].set_ylabel("Raw LFP (mV)")
ax[0].set_title(f"LFP channel {best_ch}, theta band-pass {THETA_LOW}-{THETA_HIGH} Hz")
ax[1].plot(lfp.t[m], theta_filt[m] * 1e3, "b", lw=0.9)
ax[1].plot(lfp.t[m], theta_amp[m] * 1e3, "r--", lw=0.7, label="envelope")
ax[1].set_ylabel("Theta (mV)")
ax[1].legend(loc="upper right")
ax[2].plot(lfp.t[m], theta_phase[m], "g", lw=0.7)
ax[2].set_ylabel("Phase (rad)")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_lfp_theta.png"), dpi=130)
plt.close()
print("Saved 02_lfp_theta.png")

# Power spectrum on the maze epoch
f, p = welch(lfp.values, fs=fs_lfp, nperseg=int(4 * fs_lfp))
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(f, p, "k")
ax.axvspan(THETA_LOW, THETA_HIGH, color="red", alpha=0.2, label="theta band")
ax.set_xlim(0, 30)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (V²/Hz)")
ax.set_title(f"LFP PSD — ch {best_ch} (maze epoch)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_lfp_psd.png"), dpi=130)
plt.close()
print("Saved 03_lfp_psd.png")

# %% [markdown]
# ## 5. Place fields
#
# Compute 1D firing-rate maps as a function of linearized position separately
# for left- and right-bound running. Place cells are identified by spatial
# information ≥ 0.5 bits/spike and peak rate ≥ 1 Hz.

# %%
# Trim 10 cm at each end to avoid the reward zones / turn-arounds where
# occupancy is very low and firing rates become unstable.
POS_MIN, POS_MAX = 0.10, 1.50
N_BINS = 40
pos_edges = np.linspace(POS_MIN, POS_MAX, N_BINS + 1)
pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])

def place_field(spikes_ts, position_tsd, ep, edges, smooth_bins=2.0, min_occ_sec=0.5):
    pos_in_ep = position_tsd.restrict(ep)
    occ, _ = np.histogram(pos_in_ep.values, bins=edges)
    occ_sec = occ / fs_pos
    occ_sec_smooth = gaussian_filter1d(occ_sec, sigma=smooth_bins)
    fields = {}
    for uid, sp in spikes_ts.items():
        sp_in_ep = sp.restrict(ep)
        if len(sp_in_ep) == 0:
            fields[uid] = np.zeros(len(edges) - 1)
            continue
        sp_pos = sp_in_ep.value_from(position_tsd)
        n, _ = np.histogram(sp_pos.values, bins=edges)
        n_smooth = gaussian_filter1d(n.astype(float), sigma=smooth_bins)
        valid = occ_sec >= min_occ_sec
        rate = np.where(valid, n_smooth / np.maximum(occ_sec_smooth, 1e-6), 0.0)
        fields[uid] = rate
    return fields, occ_sec

print("Computing place fields (right runs)...")
fields_R, occ_R = place_field(spikes, position, right_ep, pos_edges)
print("Computing place fields (left runs)...")
fields_L, occ_L = place_field(spikes, position, left_ep, pos_edges)

def spatial_information(rate, occ_sec):
    occ_p = occ_sec / np.sum(occ_sec)
    mean_r = np.sum(occ_p * rate)
    if mean_r <= 0:
        return 0.0
    valid = (rate > 0) & (occ_p > 0)
    return float(np.sum(occ_p[valid] * rate[valid] / mean_r * np.log2(rate[valid] / mean_r)))

place_cells = {}  # uid -> direction with strongest field
PEAK_THRESH = 2.0       # Hz
SI_THRESH = 0.7         # bits/spike
MIN_FIELD_BINS = 3      # contiguous bins ≥ peak/3
for uid in spikes.keys():
    si_R = spatial_information(fields_R[uid], occ_R)
    si_L = spatial_information(fields_L[uid], occ_L)
    pk_R = fields_R[uid].max()
    pk_L = fields_L[uid].max()
    # number of bins ≥ peak/3 (rough field width)
    width_R = int(np.sum(fields_R[uid] >= pk_R / 3.0)) if pk_R > 0 else 0
    width_L = int(np.sum(fields_L[uid] >= pk_L / 3.0)) if pk_L > 0 else 0
    qual_R = (pk_R >= PEAK_THRESH) and (si_R >= SI_THRESH) and (width_R >= MIN_FIELD_BINS)
    qual_L = (pk_L >= PEAK_THRESH) and (si_L >= SI_THRESH) and (width_L >= MIN_FIELD_BINS)
    if qual_R and qual_L:
        direction = "R" if si_R >= si_L else "L"
    elif qual_R:
        direction = "R"
    elif qual_L:
        direction = "L"
    else:
        continue
    place_cells[uid] = {
        "direction": direction,
        "field": fields_R[uid] if direction == "R" else fields_L[uid],
        "si": si_R if direction == "R" else si_L,
        "peak_rate": pk_R if direction == "R" else pk_L,
        "peak_pos": pos_centers[np.argmax(fields_R[uid] if direction == "R" else fields_L[uid])],
    }
print(f"\n{len(place_cells)} place cells (out of {len(spikes)} pyramidal cells)")

# Sort and plot population heatmap (right direction, sorted by peak position)
right_pcs = sorted([u for u, m in place_cells.items() if m["direction"] == "R"],
                   key=lambda u: place_cells[u]["peak_pos"])
left_pcs = sorted([u for u, m in place_cells.items() if m["direction"] == "L"],
                  key=lambda u: place_cells[u]["peak_pos"])
print(f"  Right-bound: {len(right_pcs)} | Left-bound: {len(left_pcs)}")

def normalize(M):
    mx = M.max(axis=1, keepdims=True)
    return np.where(mx > 0, M / mx, M)

R_mat = np.stack([fields_R[u] for u in right_pcs]) if right_pcs else np.zeros((0, N_BINS))
L_mat = np.stack([fields_L[u] for u in left_pcs]) if left_pcs else np.zeros((0, N_BINS))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
if R_mat.size:
    axes[0].imshow(normalize(R_mat), aspect="auto",
                   extent=[pos_edges[0], pos_edges[-1], len(R_mat), 0], cmap="viridis")
axes[0].set_title(f"Right-bound place cells ({len(right_pcs)})")
axes[0].set_xlabel("Position (m)")
axes[0].set_ylabel("Cell # (sorted by peak)")
if L_mat.size:
    axes[1].imshow(normalize(L_mat), aspect="auto",
                   extent=[pos_edges[0], pos_edges[-1], len(L_mat), 0], cmap="viridis")
axes[1].set_title(f"Left-bound place cells ({len(left_pcs)})")
axes[1].set_xlabel("Position (m)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_place_field_population.png"), dpi=130)
plt.close()
print("Saved 04_place_field_population.png")

# %% [markdown]
# ## 6. Theta phase entrainment
#
# For each place cell, sample theta phase at every spike during *running* and
# perform a Rayleigh test against a uniform distribution. We also pool all
# spikes from all place cells to characterize the population preferred phase.

# %%
def rayleigh(phases):
    n = len(phases)
    if n < 5:
        return np.nan, np.nan, np.nan
    R = np.sqrt(np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2) / n
    Z = n * R**2
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    mean_phase = np.angle(np.sum(np.exp(1j * phases)))
    return float(R), float(p), float(mean_phase)

per_cell_phase = {}
for uid in tqdm(place_cells, desc="Theta phase per spike"):
    sp = spikes[uid].restrict(run_ep)
    if len(sp) < 30:
        continue
    ph = sp.value_from(theta_phase_tsd).values
    per_cell_phase[uid] = ph

ent_results = []
for uid, ph in per_cell_phase.items():
    R, p, mu = rayleigh(ph)
    ent_results.append({"uid": uid, "n": len(ph), "R": R, "p": p, "mu": mu})

n_sig = sum(1 for r in ent_results if r["p"] < 0.05)
print(f"Place cells with significant theta entrainment (Rayleigh p<0.05): "
      f"{n_sig}/{len(ent_results)} ({100*n_sig/max(1,len(ent_results)):.0f}%)")

# Population pooled phase distribution
all_phases = np.concatenate(list(per_cell_phase.values()))
R_pop, p_pop, mu_pop = rayleigh(all_phases)
print(f"Pooled population: n={len(all_phases)}, R={R_pop:.3f}, mean phase={np.degrees(mu_pop):.1f}°, p={p_pop:.2e}")

# Plot: population phase histogram + per-cell preferred phases
fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection="polar"))
nbins_ph = 36
edges_ph = np.linspace(-np.pi, np.pi, nbins_ph + 1)
counts, _ = np.histogram(all_phases, bins=edges_ph)
counts = counts / counts.sum()
ax = axes[0]
width = 2*np.pi / nbins_ph
ax.bar((edges_ph[:-1] + edges_ph[1:]) / 2, counts, width=width, color="steelblue", edgecolor="k", alpha=0.8)
ax.set_title(f"Pooled spike-phase histogram\nn={len(all_phases)} spikes, R={R_pop:.3f}, μ={np.degrees(mu_pop):.0f}°", pad=30)

# Preferred phase per cell
mus = np.array([r["mu"] for r in ent_results if r["p"] < 0.05])
Rs = np.array([r["R"] for r in ent_results if r["p"] < 0.05])
ax = axes[1]
ax.scatter(mus, Rs, s=30, c="darkred", alpha=0.7)
ax.set_title(f"Preferred phase per place cell\n({len(mus)} significantly entrained)", pad=30)
ax.set_ylim(0, 1)
plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(FIG_DIR, "05_theta_entrainment.png"), dpi=130)
plt.close()
print("Saved 05_theta_entrainment.png")

# %% [markdown]
# ## 7. Theta phase precession
#
# For every place cell with a clear field on one running direction, we:
# 1. Define the field as the contiguous bins around the peak with rate
#    ≥ 0.3 × peak.
# 2. Take spikes that occurred while the rat was running in that direction
#    *and* inside the field.
# 3. Express each spike's position as a normalized in-field coordinate
#    (`x_norm ∈ [0, 1]`).
# 4. Run a circular-linear regression of phase against `x_norm` and report the
#    slope (rad / pass) plus circular-linear correlation (Kempter et al. 2012).

# %%
def field_extent(field_rates, peak_idx, frac=0.3):
    thr = field_rates[peak_idx] * frac
    lo = peak_idx
    while lo > 0 and field_rates[lo - 1] >= thr:
        lo -= 1
    hi = peak_idx
    while hi < len(field_rates) - 1 and field_rates[hi + 1] >= thr:
        hi += 1
    return lo, hi

def circ_lin_regression(x, phi, slopes_grid=None):
    """Kempter 2012 circular-linear regression. Returns (slope, intercept, rho, p)."""
    if slopes_grid is None:
        # Slope is measured in cycles per unit x (normalized in-field position).
        # Search over [-2, 2] cycles/field which spans typical precession ranges.
        slopes_grid = np.linspace(-2.0, 2.0, 401)
    x = np.asarray(x); phi = np.asarray(phi)
    R = np.array([
        np.sqrt(np.mean(np.cos(phi - 2*np.pi*s*x))**2 + np.mean(np.sin(phi - 2*np.pi*s*x))**2)
        for s in slopes_grid
    ])
    s_best = slopes_grid[np.argmax(R)]
    # intercept
    phi0 = np.arctan2(np.mean(np.sin(phi - 2*np.pi*s_best*x)),
                      np.mean(np.cos(phi - 2*np.pi*s_best*x)))
    # circular-linear correlation (Kempter eq. for grouped slope)
    theta = (2 * np.pi * s_best * x) % (2 * np.pi)
    theta_mean = np.angle(np.sum(np.exp(1j * theta)))
    phi_mean = np.angle(np.sum(np.exp(1j * phi)))
    num = np.sum(np.sin(phi - phi_mean) * np.sin(theta - theta_mean))
    den = np.sqrt(np.sum(np.sin(phi - phi_mean)**2) * np.sum(np.sin(theta - theta_mean)**2))
    rho = num / den if den > 0 else 0.0
    n = len(x)
    # significance from large-sample distribution (Fisher Z)
    if n > 5 and abs(rho) < 0.999:
        from scipy.stats import norm
        # variances of the wrapped variates (use sample circular variance)
        l_pp = np.mean(np.sin(phi - phi_mean)**2)
        l_tt = np.mean(np.sin(theta - theta_mean)**2)
        l_pt = np.mean(np.sin(phi - phi_mean)**2 * np.sin(theta - theta_mean)**2)
        if l_pt > 0:
            ts = np.sqrt((n * l_pp * l_tt) / l_pt) * rho
            p_val = 2 * (1 - norm.cdf(abs(ts)))
        else:
            p_val = np.nan
    else:
        p_val = np.nan
    return float(s_best * 2 * np.pi), float(phi0), float(rho), float(p_val), float(s_best)

precession_results = []
for uid, meta in tqdm(place_cells.items(), desc="Phase precession"):
    direction = meta["direction"]
    ep = right_ep if direction == "R" else left_ep
    field = meta["field"]
    pk = int(np.argmax(field))
    lo, hi = field_extent(field, pk, frac=0.3)
    p_lo, p_hi = pos_edges[lo], pos_edges[hi + 1]
    if p_hi - p_lo < 0.10:
        continue  # too narrow
    sp = spikes[uid].restrict(ep)
    if len(sp) == 0:
        continue
    sp_pos_tsd = sp.value_from(position)
    sp_pos = sp_pos_tsd.values
    sp_t = sp_pos_tsd.t
    in_field = (sp_pos >= p_lo) & (sp_pos <= p_hi)
    if in_field.sum() < 30:
        continue
    pos_in = sp_pos[in_field]
    t_in = sp_t[in_field]
    idx = np.searchsorted(theta_phase_tsd.t, t_in)
    idx = np.clip(idx, 0, len(theta_phase_tsd.t) - 1)
    ph_in = theta_phase_tsd.values[idx]
    if direction == "L":
        x_norm = (p_hi - pos_in) / (p_hi - p_lo)
    else:
        x_norm = (pos_in - p_lo) / (p_hi - p_lo)
    slope_rad, phi0, rho, p_val, s_cyc = circ_lin_regression(x_norm, ph_in)
    precession_results.append({
        "uid": uid, "direction": direction, "n": int(in_field.sum()),
        "field_lo": p_lo, "field_hi": p_hi, "peak_pos": meta["peak_pos"],
        "slope_rad_per_field": slope_rad, "rho": rho, "p": p_val,
        "x_norm": x_norm, "phase": ph_in, "phi0": phi0, "s_cyc": s_cyc,
        "field_rate": field, "lo_bin": lo, "hi_bin": hi,
    })

print(f"\nCells analyzed for precession: {len(precession_results)}")
neg_slope = sum(1 for r in precession_results if r["slope_rad_per_field"] < 0)
sig_neg = sum(1 for r in precession_results if r["slope_rad_per_field"] < 0 and r["p"] < 0.05)
print(f"Cells with negative phase slope: {neg_slope}/{len(precession_results)}")
print(f"Cells with significant negative slope (p<0.05): {sig_neg}/{len(precession_results)}")
print(f"Mean phase slope across population: "
      f"{np.mean([r['slope_rad_per_field'] for r in precession_results]):.3f} rad / field")

# %% [markdown]
# ### Examples — six representative precession cells

# %%
example_cells = sorted(
    [r for r in precession_results if r["slope_rad_per_field"] < 0 and not np.isnan(r["p"])],
    key=lambda r: r["p"],
)[:6]

if example_cells:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    axes = axes.flatten()
    for ax, r in zip(axes, example_cells):
        # plot spikes over two theta cycles for clarity
        x = r["x_norm"]
        ph = r["phase"]
        ax.scatter(x, np.degrees(ph), s=8, alpha=0.6, color="black")
        ax.scatter(x, np.degrees(ph) + 360, s=8, alpha=0.6, color="black")
        # regression line — plot wrapped into the [-180, 540]° window without
        # connecting across the wrap discontinuities.
        xs = np.linspace(0, 1, 400)
        line_phase_deg = np.degrees(2*np.pi * r["s_cyc"] * xs + r["phi0"])
        for shift in (-720, -360, 0, 360, 720):
            y = line_phase_deg + shift
            mask = (y >= -180) & (y <= 540)
            if mask.any():
                ax.plot(xs[mask], y[mask], "r-", lw=2, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(-180, 540)
        ax.set_xlabel("Normalized in-field position")
        ax.set_ylabel("Theta phase (deg)")
        ax.set_title(f"unit {r['uid']} ({r['direction']}-bound), n={r['n']}\n"
                     f"slope={r['slope_rad_per_field']:.2f} rad, ρ={r['rho']:.2f}, p={r['p']:.1e}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_phase_precession_examples.png"), dpi=130)
    plt.close()
    print("Saved 06_phase_precession_examples.png")

# %% [markdown]
# ### Population summary of precession slopes

# %%
slopes = np.array([r["slope_rad_per_field"] for r in precession_results])
rhos = np.array([r["rho"] for r in precession_results])
pvals = np.array([r["p"] for r in precession_results])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(slopes, bins=20, color="steelblue", edgecolor="k")
axes[0].axvline(0, color="r", ls="--")
axes[0].axvline(np.mean(slopes), color="k", lw=2, label=f"mean={np.mean(slopes):.2f}")
axes[0].set_xlabel("Phase slope (rad / field)")
axes[0].set_ylabel("# cells")
axes[0].set_title("Distribution of precession slopes")
axes[0].legend()

axes[1].hist(rhos, bins=20, color="darkorange", edgecolor="k")
axes[1].axvline(0, color="r", ls="--")
axes[1].set_xlabel("Circular-linear ρ (phase, position)")
axes[1].set_ylabel("# cells")
axes[1].set_title("Distribution of phase-position correlations")

axes[2].scatter(slopes, -np.log10(pvals + 1e-12), alpha=0.7)
axes[2].axhline(-np.log10(0.05), color="r", ls="--", label="p=0.05")
axes[2].set_xlabel("Phase slope (rad / field)")
axes[2].set_ylabel("-log10 p")
axes[2].set_title("Slope vs significance")
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_precession_population.png"), dpi=130)
plt.close()
print("Saved 07_precession_population.png")

# %% [markdown]
# ## 8. Summary

# %%
print("=" * 60)
print(f"Session: {nwbfile.session_id}")
print(f"  pyramidal cells:         {len(spikes)}")
print(f"  place cells:             {len(place_cells)}")
print(f"  theta entrainment:       {n_sig}/{len(ent_results)} significant (Rayleigh p<0.05)")
print(f"  population mean phase:   {np.degrees(mu_pop):.1f}°  (R={R_pop:.3f})")
print(f"  precession analyzed:     {len(precession_results)} cells")
print(f"  negative slopes:         {neg_slope}/{len(precession_results)} "
      f"({100*neg_slope/max(1,len(precession_results)):.0f}%)")
print(f"  significant negative:    {sig_neg}/{len(precession_results)} "
      f"({100*sig_neg/max(1,len(precession_results)):.0f}%)")
print(f"  mean slope:              {np.mean(slopes):.3f} rad / field")
print(f"  mean ρ:                  {np.mean(rhos):.3f}")
print("=" * 60)
