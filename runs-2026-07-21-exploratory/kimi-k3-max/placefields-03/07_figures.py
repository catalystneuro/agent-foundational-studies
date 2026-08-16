"""Generate the full figure suite for the place cell demonstration."""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d

# ---------------- reload results ----------------
data = np.load('placefield_results.npz', allow_pickle=True)
unit_keys = data['unit_keys']
occupancy = data['occupancy']
occ_pos = data['occ_pos']
occ_neg = data['occ_neg']
BIN_CENTERS = data['bin_centers']
results = data['results'][0]

si_all = np.array([results[k]['spatial_info'] for k in unit_keys])
p_all = np.array([results[k]['shuffle_p'] for k in unit_keys])
rate_all = np.array([results[k]['mean_rate_track'] for k in unit_keys])
sp_all = np.array([results[k]['sparsity'] for k in unit_keys])
peak_all = np.array([np.nanmax(results[k]['rate_map']) for k in unit_keys])
exc = np.array([results[k]['cell_type'] == 'excitatory' for k in unit_keys])

for peak_thr in [0, 1, 2]:
    n = (exc & (p_all < 0.05) & (rate_all > 0.1) & (peak_all > peak_thr) & ~np.isnan(p_all)).sum()
    print(f"peak>{peak_thr} Hz: {n} place cells")

# ---------------- reload nwb for spike positions ----------------
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
xy_t = xy.t
xy_v = xy.values

def spike_positions(k):
    """xy and linearized position of each spike of unit k (nearest sample)."""
    st = units[k].restrict(maze).t
    ix = np.searchsorted(xy_t, st).clip(0, len(xy_t) - 1)
    il = np.searchsorted(lin_t, st).clip(0, len(lin_t) - 1)
    return st, xy_v[ix], lin_v[il], valid[il]

# ---------------- Fig 1: session overview ----------------
dt = np.median(np.diff(xy_t))
dx = np.gradient(xy_v[:, 0], dt); dy = np.gradient(xy_v[:, 1], dt)
speed = gaussian_filter1d(np.nan_to_num(np.sqrt(dx**2 + dy**2)), sigma=int(0.5/dt))
t0 = xy_t[0]

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, 0])
ax.plot(xy_v[:, 0], xy_v[:, 1], lw=0.05, alpha=0.4, color='steelblue')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('A  Trajectory during maze epoch', loc='left', fontsize=11)
ax.set_aspect('equal')

ax = fig.add_subplot(gs[0, 1])
ax.plot(lin_t[::10] - t0, lin_v[::10], lw=0.3, color='darkorange')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Linearized position (m)')
ax.set_title('B  Linearized position (runs only)', loc='left', fontsize=11)

ax = fig.add_subplot(gs[1, 0])
ax.plot(xy_t[::10] - t0, speed[::10] * 100, lw=0.3, color='seagreen')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Speed (cm/s)')
ax.set_title('C  Running speed', loc='left', fontsize=11)
ax.set_xlim(0, xy_t[-1] - t0)

ax = fig.add_subplot(gs[1, 1])
ax.bar(BIN_CENTERS, occupancy, width=np.diff(BIN_CENTERS)[0], color='slategray')
ax.set_xlabel('Linearized position (m)'); ax.set_ylabel('Occupancy (s)')
ax.set_title('D  Occupancy on track', loc='left', fontsize=11)
plt.savefig('fig1_session_overview.png', dpi=150)
plt.close()
print("saved fig1_session_overview.png")

# ---------------- pick example place cells ----------------
is_place = exc & (p_all < 0.05) & (rate_all > 0.1) & (peak_all > 1.0) & ~np.isnan(p_all)
place_keys = [k for k, m in zip(unit_keys, is_place) if m]
print("place cells with peak>1 Hz:", len(place_keys))

# rank by spatial info, pick spread across track
si_place = {k: results[k]['spatial_info'] for k in place_keys}
peak_place = {k: BIN_CENTERS[np.nanargmax(results[k]['rate_map'])] for k in place_keys}
top_by_si = sorted(place_keys, key=lambda k: -si_place[k])
# pick 6 examples with peaks spread along the track
chosen = []
for k in top_by_si:
    if all(abs(peak_place[k] - peak_place[c]) > 0.2 for c in chosen):
        chosen.append(k)
    if len(chosen) == 6:
        break
chosen = sorted(chosen, key=lambda k: peak_place[k])
print("example cells:", chosen, [f"{peak_place[k]:.2f} m" for k in chosen])

# ---------------- Fig 2: example place cells ----------------
fig, axes = plt.subplots(6, 3, figsize=(12, 16))
for row, k in enumerate(chosen):
    st, sp_xy, sp_lin, sp_valid = spike_positions(k)
    rm = results[k]['rate_map']
    rm_p = results[k]['rate_map_pos']
    rm_n = results[k]['rate_map_neg']

    ax = axes[row, 0]
    ax.plot(xy_v[:, 0], xy_v[:, 1], lw=0.05, alpha=0.3, color='gray')
    m = ~np.isnan(sp_xy[:, 0]) & sp_valid
    ax.plot(sp_xy[m, 0], sp_xy[m, 1], '.', ms=2, color='crimson')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if row == 0:
        ax.set_title('Spike locations (red)', fontsize=10)
    ax.set_ylabel(f"unit {k}", fontsize=9)

    ax = axes[row, 1]
    ax.plot(BIN_CENTERS, rm, color='black', lw=1.5)
    ax.fill_between(BIN_CENTERS, rm, color='black', alpha=0.2)
    ax.set_xlim(0, 1.6)
    if row == 0:
        ax.set_title('Firing rate map', fontsize=10)
    if row == 5:
        ax.set_xlabel('Position (m)')
    ax.set_ylabel('Hz', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.85, f"SI={results[k]['spatial_info']:.2f}", transform=ax.transAxes, fontsize=8)

    ax = axes[row, 2]
    ax.plot(BIN_CENTERS, rm_p, color='tab:blue', lw=1.2, label='→ 1.6 m')
    ax.plot(BIN_CENTERS, rm_n, color='tab:orange', lw=1.2, label='→ 0 m')
    ax.set_xlim(0, 1.6)
    if row == 0:
        ax.set_title('By running direction', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
    if row == 5:
        ax.set_xlabel('Position (m)')
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('fig2_example_place_cells.png', dpi=150)
plt.close()
print("saved fig2_example_place_cells.png")

# ---------------- Fig 3: grid of top 24 rate maps ----------------
top24 = top_by_si[:24]
fig, axes = plt.subplots(4, 6, figsize=(14, 9), sharex=True)
for ax, k in zip(axes.flat, top24):
    rm = results[k]['rate_map']
    ax.fill_between(BIN_CENTERS, rm, color='darkblue', alpha=0.7)
    ax.set_xlim(0, 1.6)
    ax.set_title(f"u{k}  SI={results[k]['spatial_info']:.2f}", fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[-1]:
    ax.set_xlabel('Position (m)', fontsize=8)
plt.suptitle('Top 24 place cells by spatial information', y=1.00, fontsize=12)
plt.tight_layout()
plt.savefig('fig3_top24_ratemaps.png', dpi=150)
plt.close()
print("saved fig3_top24_ratemaps.png")

# ---------------- Fig 4: population heatmap ----------------
place_sorted = sorted(place_keys, key=lambda k: peak_place[k])
maps = np.array([results[k]['rate_map'] for k in place_sorted])
maps_norm = maps / np.nanmax(maps, axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1]})
ax = axes[0]
im = ax.imshow(maps_norm, aspect='auto', cmap='viridis', extent=[0, 1.6, len(place_sorted), 0])
ax.set_xlabel('Position (m)'); ax.set_ylabel('Place cell (sorted by peak)')
ax.set_title(f'Normalized rate maps (n={len(place_sorted)})')
plt.colorbar(im, ax=ax, label='Norm. rate', shrink=0.8)

ax = axes[1]
# split by direction: each cell normalized within direction
maps_p = np.array([results[k]['rate_map_pos'] for k in place_sorted])
maps_n = np.array([results[k]['rate_map_neg'] for k in place_sorted])
both = np.hstack([maps_p, maps_n])
both_norm = both / np.nanmax(both, axis=1, keepdims=True)
im = ax.imshow(both_norm, aspect='auto', cmap='viridis',
               extent=[0, 3.2, len(place_sorted), 0])
ax.axvline(1.6, color='white', lw=1, ls='--')
ax.set_xlabel('Position (m)   |   left: runs →1.6 m, right: runs →0 m')
ax.set_ylabel('Place cell (sorted by peak)')
ax.set_title('Rate maps by running direction')
ax.set_xticks([0, 0.8, 1.6, 2.4, 3.2])
ax.set_xticklabels(['0', '0.8', '1.6 | 0', '0.8', '1.6'])
plt.colorbar(im, ax=ax, label='Norm. rate', shrink=0.8)
plt.tight_layout()
plt.savefig('fig4_population.png', dpi=150)
plt.close()
print("saved fig4_population.png")

# ---------------- Fig 5: spatial information statistics ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.hist(si_all[exc], bins=30, color='darkblue', alpha=0.8, label='excitatory')
inh = ~exc
ax.hist(si_all[inh], bins=15, color='crimson', alpha=0.6, label='inhibitory')
ax.set_xlabel('Spatial information (bits/spike)')
ax.set_ylabel('Units')
ax.set_title('A  Spatial information distribution', loc='left', fontsize=11)
ax.legend(fontsize=8)

ax = axes[1]
# observed SI vs shuffle mean
null_mean = np.array([results[k]['si_null_mean'] for k in unit_keys])
m = exc & ~np.isnan(null_mean)
ax.scatter(null_mean[m], si_all[m], s=12, c=np.where(is_place[m], 'darkblue', 'lightgray'),
           edgecolors='none', alpha=0.8)
lim = [0, max(np.nanmax(null_mean[m]), np.nanmax(si_all[m])) * 1.05]
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlabel('Shuffle mean SI (bits/spike)')
ax.set_ylabel('Observed SI (bits/spike)')
ax.set_title('B  Observed vs shuffled spatial info', loc='left', fontsize=11)
ax.set_xlim(lim); ax.set_ylim(lim)

ax = axes[2]
m = exc & ~np.isnan(p_all)
sc = ax.scatter(rate_all[m], si_all[m], s=12,
                c=np.where(is_place[m], 'darkblue', 'lightgray'), alpha=0.8, edgecolors='none')
ax.set_xscale('log')
ax.set_xlabel('Mean rate on track (Hz)')
ax.set_ylabel('Spatial information (bits/spike)')
ax.set_title('C  Rate vs spatial info (blue = place cell)', loc='left', fontsize=11)
plt.tight_layout()
plt.savefig('fig5_spatial_info_stats.png', dpi=150)
plt.close()
print("saved fig5_spatial_info_stats.png")

# ---------------- Fig 6: directionality ----------------
# correlation between direction-specific maps for place cells
dir_corr = []
for k in place_keys:
    a = np.nan_to_num(results[k]['rate_map_pos'])
    b = np.nan_to_num(results[k]['rate_map_neg'])
    if a.std() > 0 and b.std() > 0:
        dir_corr.append(np.corrcoef(a, b)[0, 1])
    else:
        dir_corr.append(np.nan)
dir_corr = np.array(dir_corr)

# most direction-selective examples (lowest correlation)
order = np.argsort(dir_corr)
dir_examples = [place_keys[i] for i in order[:3]]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for j, k in enumerate(dir_examples):
    ax = axes[j]
    ax.plot(BIN_CENTERS, results[k]['rate_map_pos'], color='tab:blue', lw=1.5, label='→ 1.6 m')
    ax.plot(BIN_CENTERS, results[k]['rate_map_neg'], color='tab:orange', lw=1.5, label='→ 0 m')
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k}, r={dir_corr[order[j]]:.2f}", fontsize=9)
    ax.set_xlabel('Position (m)', fontsize=8)
    if j == 0:
        ax.set_ylabel('Firing rate (Hz)')
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

ax = axes[3]
ax.hist(dir_corr[~np.isnan(dir_corr)], bins=25, color='slategray')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Correlation between direction maps')
ax.set_ylabel('Place cells')
ax.set_title('Directional consistency', fontsize=9)
plt.tight_layout()
plt.savefig('fig6_directionality.png', dpi=150)
plt.close()
print("saved fig6_directionality.png")

print("\nmedian direction correlation: %.2f" % np.nanmedian(dir_corr))
print("fraction place cells with r<0.5: %.2f" % np.nanmean(dir_corr < 0.5))
