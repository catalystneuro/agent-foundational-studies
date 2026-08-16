"""Place fields on the maze epoch: run bouts, tuning curves, SI shuffle, place-cell mask."""
import numpy as np
import h5py
import remfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

rng = np.random.default_rng(42)

with open('/tmp/achilles_url.txt') as f:
    bare = f.read().strip().split('?')[0]
disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
h5py_file = h5py.File(remfile.File(bare, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

units = nwb['units']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
lt = lin.t
lv = np.asarray(lin.values).ravel()
pos_fs = 1.0 / np.median(np.diff(lt))
print('position fs:', pos_fs)

# ---------------- run bouts ----------------
valid = np.isfinite(lv)
# boundaries of valid stretches
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = []
for s, e in zip(starts, ends):
    bouts.append([lt[s], lt[e - 1]])
# merge over gaps < 0.3 s
merged = [bouts[0]]
for b in bouts[1:]:
    if b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(b)
# criteria: >=1 s, span > 0.3 m, median |dl/dt| > 0.15 m/s
keep = []
for b in merged:
    i0, i1 = np.searchsorted(lt, b[0]), np.searchsorted(lt, b[1])
    seg = lv[i0:i1]
    seg = seg[np.isfinite(seg)]
    dur = b[1] - b[0]
    if dur < 1.0 or len(seg) < 5:
        continue
    span = seg.max() - seg.min()
    speed = np.median(np.abs(np.diff(seg)) * pos_fs)
    if span > 0.3 and speed > 0.15:
        keep.append(b)
print(f'run bouts: {len(keep)}, total {sum(b[1]-b[0] for b in keep):.1f} s')

# tau axis: concatenated bout time
bout_dur = np.array([b[1] - b[0] for b in keep])
tau_edges = np.concatenate([[0], np.cumsum(bout_dur)])
total_tau = tau_edges[-1]
bout_start_wall = np.array([b[0] for b in keep])

def wall_to_tau(t):
    """map wall times inside bouts to concatenated tau; NaN outside bouts"""
    tau = np.full_like(t, np.nan, dtype=float)
    bi = np.searchsorted(bout_start_wall, t, side='right') - 1
    ok = (bi >= 0)
    idx = np.where(ok)[0]
    for j in idx:
        b = bi[j]
        off = t[j] - bout_start_wall[b]
        if 0 <= off <= bout_dur[b]:
            tau[j] = tau_edges[b] + off
    return tau

def tau_to_wall(tau):
    tau = np.mod(tau, total_tau)
    bi = np.searchsorted(tau_edges, tau, side='right') - 1
    bi = np.clip(bi, 0, len(bout_dur) - 1)
    return bout_start_wall[bi] + (tau - tau_edges[bi])

# position as function of wall time (interp over valid samples)
fin = np.isfinite(lv)
lt_f, lv_f = lt[fin], lv[fin]

def pos_at(t):
    return np.interp(t, lt_f, lv_f)

# ---------------- tuning curves ----------------
nbins = 50
edges = np.linspace(0, 1.6, nbins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

# occupancy from position samples inside bouts
pos_tau = wall_to_tau(lt)
in_bout = np.isfinite(pos_tau)
occ, _ = np.histogram(lv[in_bout & fin], bins=edges)
occ_t = occ / pos_fs
occ_t_sm = gaussian_filter1d(occ_t, 1.5, mode='nearest')

unit_keys = list(units.keys())
cell_type = units.get_info('cell_type')
exc_keys = [k for k in unit_keys if cell_type[k] == 'excitatory']
print('excitatory units:', len(exc_keys))

spike_pos = {}
spike_tau = {}
for k in unit_keys:
    sp = units[k].t
    tau = wall_to_tau(sp)
    m = np.isfinite(tau)
    spike_tau[k] = tau[m]
    spike_pos[k] = pos_at(sp[m])

tuning = {}
for k in unit_keys:
    cnt, _ = np.histogram(spike_pos[k], bins=edges)
    rate = cnt / np.maximum(occ_t, 1e-12)
    rate_sm = gaussian_filter1d(rate, 1.5, mode='nearest')
    tuning[k] = rate_sm

def skaggs_si(rate, occ_t):
    p = occ_t / occ_t.sum()
    mean_r = (rate * p).sum()
    m = (rate > 0) & (p > 0) & (mean_r > 0)
    if mean_r <= 0:
        return 0.0
    return np.sum(p[m] * (rate[m] / mean_r) * np.log2(rate[m] / mean_r))

# ---------------- SI shuffle (circular time shift on tau axis) ----------------
nshuf = 500
si_real = {}
si_p = {}
# precompute position on a tau grid for fast lookup of shifted spikes
tau_grid = np.linspace(0, total_tau, 20001)
pos_grid = pos_at(tau_to_wall(tau_grid))

for k in tqdm(exc_keys, desc='SI shuffle'):
    rate = tuning[k]
    si_r = skaggs_si(rate, occ_t)
    si_real[k] = si_r
    tau_sp = spike_tau[k]
    if len(tau_sp) < 10:
        si_p[k] = 1.0
        continue
    shifts = rng.uniform(0, total_tau, nshuf)
    cnt_ge = 0
    for s in shifts:
        tau_sh = np.mod(tau_sp + s, total_tau)
        pos_sh = np.interp(tau_sh, tau_grid, pos_grid)
        cnt, _ = np.histogram(pos_sh, bins=edges)
        rate_sh = gaussian_filter1d(cnt / np.maximum(occ_t, 1e-12), 1.5, mode='nearest')
        if skaggs_si(rate_sh, occ_t) >= si_r:
            cnt_ge += 1
    si_p[k] = (cnt_ge + 1) / (nshuf + 1)

# place cell criteria
place = {}
for k in exc_keys:
    peak = tuning[k].max()
    mean_rate = len(spike_pos[k]) / max(occ_t.sum(), 1e-9)
    place[k] = (peak >= 1.0) and (mean_rate > 0.1) and (si_p[k] < 0.05)
place_keys = [k for k in exc_keys if place[k]]
print(f'place cells: {len(place_keys)} / {len(exc_keys)} excitatory')

np.savez('placefields.npz',
         unit_keys=np.array(unit_keys),
         exc_keys=np.array(exc_keys),
         place_keys=np.array(place_keys),
         tuning=np.array([tuning[k] for k in unit_keys]),
         centers=centers,
         bouts=np.array(keep),
         si_real=np.array([si_real.get(k, np.nan) for k in unit_keys]),
         si_p=np.array([si_p.get(k, np.nan) for k in unit_keys]),
         occ_t=occ_t)

# ---------------- figure ----------------
order = np.argsort(np.argmax(np.array([tuning[k] for k in place_keys]), axis=1)) if place_keys else []
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
if place_keys:
    T = np.array([tuning[k] for k in place_keys])[order]
    Tn = T / T.max(axis=1, keepdims=True)
    ax.imshow(Tn, aspect='auto', cmap='viridis',
              extent=[centers[0], centers[-1], len(place_keys), 0])
ax.set_title(f'Place-field template ({len(place_keys)} place cells)')
ax.set_xlabel('track position (m)')
ax.set_ylabel('place cell (sorted by peak)')

ax = axes[1]
# example cells: 6 place cells with peaks spread across the track
peaks = {k: centers[np.argmax(tuning[k])] for k in place_keys}
targets = np.linspace(0.1, 1.5, 6)
ex = []
for tpos in targets:
    k = min(place_keys, key=lambda j: abs(peaks[j] - tpos))
    if k not in ex:
        ex.append(k)
off = np.percentile([tuning[k].max() for k in place_keys], 75) * 1.3
for i, k in enumerate(ex):
    ax.plot(centers, tuning[k] + i * off, color=plt.cm.tab10(i % 10), lw=1.4)
ax.set_title('Example place fields (stacked)')
ax.set_xlabel('track position (m)')
ax.set_ylabel('firing rate (Hz, offset)')

ax = axes[2]
si_vals = np.array([si_real[k] for k in exc_keys])
p_vals = np.array([si_p[k] for k in exc_keys])
ax.scatter(si_vals, p_vals, c=['tab:red' if place[k] else 'gray' for k in exc_keys], s=12)
ax.axhline(0.05, color='k', ls='--', lw=0.8)
ax.set_xlabel('spatial information (bits/spike)')
ax.set_ylabel('shuffle p-value')
ax.set_title('Place-cell selection')
ax.set_yscale('log')

fig.tight_layout()
fig.savefig('fig02_place_fields.png', dpi=150)
print('saved fig02_place_fields.png')
