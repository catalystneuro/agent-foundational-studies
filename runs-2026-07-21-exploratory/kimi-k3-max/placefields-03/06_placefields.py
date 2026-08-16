"""Main place field analysis on the Achilles maze epoch.

Computes occupancy, firing rate maps, Skaggs spatial information, sparsity,
and shuffle-based significance for all units. Saves intermediate results.
"""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ---------------- load ----------------
with open('/tmp/achilles_url.txt') as f:
    s3_url = f.read().strip()

disk_cache = remfile.DiskCache('/tmp/remfile_cache_achilles')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb['units']
maze = nwb['epochs'][1]
lin = nwb['1.6mLinearMazeLinearizedTimeSeries'].restrict(maze)
xy = nwb['1.6mLinearMazeSpatialSeries'].restrict(maze)

lin_t = lin.t
lin_v = lin.values[:, 0]
valid = ~np.isnan(lin_v)

# ---------------- parameters ----------------
TRACK_LEN = 1.6
N_BINS = 50
BIN_EDGES = np.linspace(0, TRACK_LEN, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
SMOOTH_SIGMA_BINS = 1.5
N_SHUFFLE = 500
MIN_OCC_S = 0.1  # exclude bins with very low occupancy from SI

dt = np.median(np.diff(lin_t))

# occupancy (seconds per bin) from valid samples
occ_counts, _ = np.histogram(lin_v[valid], bins=BIN_EDGES)
occupancy = occ_counts * dt
occ_prob = occupancy / occupancy.sum()
print("occupancy: min %.2f s, median %.2f s" % (occupancy.min(), np.median(occupancy)))

# running direction from linearized position (sign of velocity)
lin_filled = lin_v.copy()
# forward/back fill within valid stretches for velocity estimate
idx = np.where(valid)[0]
lin_interp = np.interp(np.arange(len(lin_v)), idx, lin_v[idx])
vel = gaussian_filter1d(np.gradient(lin_interp, dt), sigma=int(0.25/dt))
direction = np.sign(vel)  # +1 toward 1.6 m, -1 toward 0

# direction-specific occupancy
occ_pos_counts, _ = np.histogram(lin_v[valid & (direction > 0)], bins=BIN_EDGES)
occ_neg_counts, _ = np.histogram(lin_v[valid & (direction < 0)], bins=BIN_EDGES)
occ_pos = occ_pos_counts * dt
occ_neg = occ_neg_counts * dt

# ---------------- per-unit rate maps ----------------
unit_keys = list(units.keys())
cell_type = units.get_info('cell_type').values
location = units.get_info('location').values

def spike_bin_counts(spike_times):
    """Histogram of spikes over position bins using nearest valid position sample."""
    if len(spike_times) == 0:
        z = np.zeros(N_BINS)
        return z, z.copy(), z.copy()
    pos_idx = np.searchsorted(lin_t, spike_times).clip(0, len(lin_t) - 1)
    pos_idx = pos_idx[valid[pos_idx]]
    b = np.digitize(lin_v[pos_idx], BIN_EDGES) - 1
    keep = (b >= 0) & (b < N_BINS)
    b = b[keep]
    d = direction[pos_idx][keep]
    counts = np.bincount(b, minlength=N_BINS).astype(float)
    counts_pos = np.bincount(b[d > 0], minlength=N_BINS).astype(float)
    counts_neg = np.bincount(b[d < 0], minlength=N_BINS).astype(float)
    return counts, counts_pos, counts_neg

def rate_map(counts, occ):
    with np.errstate(invalid='ignore', divide='ignore'):
        rm = counts / occ
    rm[occ < MIN_OCC_S] = np.nan
    return gaussian_filter1d(np.nan_to_num(rm), SMOOTH_SIGMA_BINS)

def skaggs_info(rm, occ_p):
    lam = np.nansum(rm * occ_p)
    if lam <= 0:
        return 0.0
    r = rm / lam
    with np.errstate(invalid='ignore', divide='ignore'):
        si = np.nansum(occ_p * r * np.log2(r))
    return float(si)

def sparsity(rm, occ_p):
    num = np.nansum(occ_p * rm) ** 2
    den = np.nansum(occ_p * rm ** 2)
    return float(num / den) if den > 0 else np.nan

results = {}
rng = np.random.default_rng(42)

epoch_dur = maze.end[0] - maze.start[0]
valid_dur = valid.sum() * dt

for i, k in enumerate(tqdm(unit_keys, desc="units")):
    st = units[k].restrict(maze).t
    counts, counts_pos, counts_neg = spike_bin_counts(st)
    rm = rate_map(counts, occupancy)
    si = skaggs_info(np.nan_to_num(rm), occ_prob)
    sp = sparsity(np.nan_to_num(rm), occ_prob)
    n_spikes_track = int(counts.sum())
    mean_rate_track = n_spikes_track / valid_dur

    # shuffle: circular shift of spike times within epoch
    if n_spikes_track >= 30:
        si_null = np.zeros(N_SHUFFLE)
        for s in range(N_SHUFFLE):
            shift = rng.uniform(20, epoch_dur - 20)
            st_sh = ((st - maze.start[0] + shift) % epoch_dur) + maze.start[0]
            c_sh, _, _ = spike_bin_counts(st_sh)
            rm_sh = rate_map(c_sh, occupancy)
            si_null[s] = skaggs_info(np.nan_to_num(rm_sh), occ_prob)
        p_val = (np.sum(si_null >= si) + 1) / (N_SHUFFLE + 1)
        si_null_mean = si_null.mean()
    else:
        p_val = np.nan
        si_null_mean = np.nan

    results[k] = dict(
        cell_type=cell_type[i], location=location[i],
        n_spikes_maze=len(st), n_spikes_track=n_spikes_track,
        mean_rate_track=mean_rate_track,
        rate_map=rm,
        rate_map_pos=rate_map(counts_pos, occ_pos),
        rate_map_neg=rate_map(counts_neg, occ_neg),
        spatial_info=si, sparsity=sp, shuffle_p=p_val, si_null_mean=si_null_mean,
    )

np.savez_compressed('placefield_results.npz',
                    unit_keys=np.array(unit_keys),
                    occupancy=occupancy, occ_pos=occ_pos, occ_neg=occ_neg,
                    bin_centers=BIN_CENTERS,
                    results=np.array([results], dtype=object))

# ---------------- summary ----------------
si_all = np.array([results[k]['spatial_info'] for k in unit_keys])
p_all = np.array([results[k]['shuffle_p'] for k in unit_keys])
rate_all = np.array([results[k]['mean_rate_track'] for k in unit_keys])
exc = np.array([results[k]['cell_type'] == 'excitatory' for k in unit_keys])

is_place = exc & (p_all < 0.05) & (rate_all > 0.1) & ~np.isnan(p_all)
print("\nexcitatory units:", exc.sum())
print("place cells (exc, p<0.05, rate>0.1 Hz):", is_place.sum())
print("SI distribution (exc): median %.2f, max %.2f bits/spike" % (np.nanmedian(si_all[exc]), np.nanmax(si_all[exc])))

# quick population figure: sorted normalized rate maps
place_keys = [k for k, m in zip(unit_keys, is_place) if m]
peak_pos = {k: BIN_CENTERS[np.nanargmax(results[k]['rate_map'])] for k in place_keys}
place_keys_sorted = sorted(place_keys, key=lambda k: peak_pos[k])
maps = np.array([results[k]['rate_map'] for k in place_keys_sorted])
maps_norm = maps / np.nanmax(maps, axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(maps_norm, aspect='auto', cmap='viridis',
          extent=[0, TRACK_LEN, len(place_keys_sorted), 0])
ax.set_xlabel('Position on linear track (m)')
ax.set_ylabel('Place cells (sorted by peak)')
ax.set_title('Normalized firing rate maps')
plt.tight_layout()
plt.savefig('fig_population_sorted.png', dpi=150)
print("saved fig_population_sorted.png with", len(place_keys_sorted), "place cells")
