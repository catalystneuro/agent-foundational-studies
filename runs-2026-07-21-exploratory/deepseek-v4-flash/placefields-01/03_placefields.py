# %% [markdown]
# # 03 - Place field analysis: tuning curves, spatial information, shuffle null
# For each unit we compute a 1D tuning curve on the linearized track
# (50 bins, 0-1.6 m), Skaggs spatial information (bits/spike), and a
# circular-shift null (500 shifts on the concatenated run-time axis). Units whose
# real spatial information beats 95% of the null are place cells.

# %%
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm
import time

t0 = time.time()
s3_url = open('/tmp/achilles_url.txt').read().strip()
rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(f'loaded in {time.time()-t0:.1f} s')

# --- raw streams ---
units = nwb['units']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
l = np.asarray(lin.values)[:, 0]     # linearized position (m), NaN off-track
t = np.asarray(lin.t)                # absolute time (s)

maze = nwb['epochs'][nwb['epochs'].label == 'MazeEpoch']

# %% [markdown]
# ## Run bout detection
# Contiguous valid stretches of the linearized series, merged over gaps <0.3 s,
# then filtered: duration >= 1 s, path span >= 0.3 m, median speed >= 0.15 m/s.
# %%
valid = ~np.isnan(l)
edge = np.diff(valid.astype(np.int8))
starts = np.flatnonzero(edge == 1) + 1
if valid[0]:
    starts = np.concatenate([[0], starts])
ends = np.flatnonzero(edge == -1) + 1
if valid[-1]:
    ends = np.concatenate([ends, [len(valid)]])

merged_s, merged_e = [], []
cur_s, cur_e = starts[0], ends[0]
for s, e in zip(starts[1:], ends[1:]):
    if s - cur_e < 12:
        cur_e = e
    else:
        merged_s.append(cur_s); merged_e.append(cur_e)
        cur_s, cur_e = s, e
merged_s.append(cur_s); merged_e.append(cur_e)

bouts = []
for si, ei in zip(merged_s, merged_e):
    x = l[si:ei]; tt = t[si:ei]
    dur = tt[-1] - tt[0]
    if dur < 1.0:
        continue
    span = np.nanmax(x) - np.nanmin(x)
    if span < 0.3:
        continue
    dx = np.diff(x); dtt = np.diff(tt)
    med_v = np.median(np.abs(dx / dtt)) if len(dx) else 0.0
    if med_v < 0.15:
        continue
    d = int(np.sign(np.nanmedian(dx)))
    if d == 0:
        continue
    bouts.append({'si_idx': si, 'ei_idx': ei, 'start': tt[0], 'end': tt[-1],
                  'dir': d, 'dur': dur})
print('run bouts:', len(bouts))
print('total run time (s):', round(sum(b['dur'] for b in bouts), 1))
print('+dir / -dir:', sum(b['dir'] == 1 for b in bouts), '/',
      sum(b['dir'] == -1 for b in bouts))

# --- continuous position for spike lookup: interpolate NaN gaps ---
lin_interp = l.copy()
nans = np.isnan(lin_interp)
idx_all = np.arange(len(lin_interp))
if nans.any():
    lin_interp[nans] = np.interp(idx_all[nans], idx_all[~nans], lin_interp[~nans])

# --- occupancy: real, valid on-track samples ---
NBIN = 50
edges = np.linspace(0.0, 1.6, NBIN + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
in_bout = np.zeros(len(t), dtype=bool)
for b in bouts:
    in_bout[b['si_idx']:b['ei_idx']] = True
occ_sam = np.histogram(l[valid & in_bout], bins=edges)[0]   # samples per bin
occ_s = occ_sam / 39.0625                                    # seconds per bin
print('occupancy seconds/bin: mean %.2f range %.2f-%.2f' % (occ_s.mean(), occ_s.min(), occ_s.max()))

# --- concatenated run-axis for the circular-shift null ---
bout_starts_wall = np.array([b['start'] for b in bouts])
bout_dur = np.array([b['dur'] for b in bouts])
cum_start = np.concatenate([[0.0], np.cumsum(bout_dur)])
T_total = cum_start[-1]


def si_from_counts(counts, occ):
    """Skaggs spatial information (bits/spike) from counts and occupancy (s)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rm = counts / occ
        R = counts.sum() / occ.sum()
        p = occ / occ.sum()
        contrib = np.where((rm > 0) & np.isfinite(rm), p * (rm / R) * np.log2(rm / R), 0.0)
    return float(np.sum(contrib))


def counts_from_times(spk_t):
    """Bin spike times into position bins via nearest-sample lookup."""
    idx = np.searchsorted(t, spk_t, side='right') - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return np.histogram(lin_interp[idx], bins=edges)[0]


def map_to_bout(spk_t):
    """For each spike time, the run-bout index and the tau (time since bout start)."""
    b = np.searchsorted(bout_starts_wall, spk_t, side='right') - 1
    bout_ends_wall = bout_starts_wall + bout_dur
    ok = (b >= 0) & (b < nbouts) & (spk_t >= bout_starts_wall[b]) & (spk_t < bout_ends_wall[b])
    b_ok = b[ok]
    tau = spk_t[ok] - bout_starts_wall[b_ok]
    return b_ok, tau


nbouts = len(bouts)

# %% [markdown]
# ## Per-unit spatial information + circular-shift null (500 shifts)
# %%
NSHUF = 500
rng = np.random.default_rng(42)
all_ids = list(units.keys())
ct = np.asarray(units.get_info('cell_type'))
total_run_s = sum(b['dur'] for b in bouts)

stats = {}
for uid in tqdm(all_ids, desc='units'):
    spk_full = np.asarray(units[uid].restrict(maze).t)
    b, tau = map_to_bout(spk_full)
    if len(b) == 0:
        continue
    spk_in = bout_starts_wall[b] + tau          # wall times inside bouts
    real_cnt = counts_from_times(spk_in)
    si_real = si_from_counts(real_cnt, occ_s)
    mean_rate = len(spk_in) / total_run_s
    peak = real_cnt.max() / occ_s[real_cnt.argmax()] if real_cnt.max() > 0 else 0.0

    null = np.empty(NSHUF)
    for k in range(NSHUF):
        s = rng.uniform(0.0, T_total)
        tau_s = (tau + s) % bout_dur[b]
        t_s = bout_starts_wall[b] + tau_s
        cnt = counts_from_times(t_s)
        null[k] = si_from_counts(cnt, occ_s)
    pval = (np.sum(null >= si_real) + 1) / (NSHUF + 1)
    stats[uid] = dict(si=si_real, null=null, p=pval, mean_rate=mean_rate, peak=peak,
                      rate_map=real_cnt / occ_s, counts=real_cnt)

# %% [markdown]
# ## Place cell classification
# Criteria: p < 0.05 (vs circular-shift null), mean rate > 0.1 Hz, peak rate > 1 Hz.
# %%
is_exc = np.array([ct[all_ids.index(u)] == 'excitatory' for u in stats])
exc_ids = [u for u in stats if ct[all_ids.index(u)] == 'excitatory']

si_exc = np.array([stats[u]['si'] for u in exc_ids])
p_exc = np.array([stats[u]['p'] for u in exc_ids])
mr_exc = np.array([stats[u]['mean_rate'] for u in exc_ids])
pk_exc = np.array([stats[u]['peak'] for u in exc_ids])

place_mask = (p_exc < 0.05) & (mr_exc > 0.1) & (pk_exc > 1.0)
print(f'excitatory units: {len(exc_ids)}; place cells: {place_mask.sum()} '
      f'({100*place_mask.mean():.0f}%)')

inh_ids = [u for u in all_ids if ct[all_ids.index(u)] == 'inhibitory']
si_inh = np.array([stats[u]['si'] for u in inh_ids if u in stats])

print('median SI excitatory:  %.3f bits/spike' % np.median(si_exc))
print('median SI inhibitory:  %.3f bits/spike' % (np.median(si_inh) if len(si_inh) else np.nan))
null_all = np.concatenate([stats[u]['null'] for u in exc_ids])
print('null median SI:        %.3f bits/spike' % np.median(null_all))
print('null 95th pct:         %.3f bits/spike' % np.percentile(null_all, 95))

# save
np.savez('placefields_results.npz',
         all_ids=np.array(all_ids, dtype='<U16'),
         ct=ct,
         edges=edges,
         centers=centers,
         occ_s=occ_s,
         stats_si=np.array([stats[u]['si'] for u in all_ids if u in stats]),
         stats_p=np.array([stats[u]['p'] for u in all_ids if u in stats]),
         stats_mr=np.array([stats[u]['mean_rate'] for u in all_ids if u in stats]),
         stats_peak=np.array([stats[u]['peak'] for u in all_ids if u in stats]),
         stats_map=np.array([stats[u]['rate_map'] for u in all_ids if u in stats]),
         stats_ids=np.array([u for u in all_ids if u in stats], dtype='<U16'),
         place_ids=np.array(exc_ids, dtype='<U16'),
         place_mask=place_mask)
print('saved placefields_results.npz')