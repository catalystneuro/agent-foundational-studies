"""Step 2: run bouts, rate maps, Skaggs SI shuffle, place-cell selection.

Recipe (validated on this session in prior runs):
- Run bouts = contiguous valid stretches of the linearized position, merged over
  <0.3 s gaps, min duration 1 s, span >0.3 m, median |dl/dt| >0.15 m/s.
- Rate maps: 50 bins over 0-1.6 m, Gaussian sigma 1.5 bins, direction-pooled.
- SI shuffle: circular shift of spike TIMES on a concatenated run-bout time axis
  (shifting over wall-clock piles spikes at track ends and inflates the null).
- Place cells: excitatory, SI shuffle p<0.05, mean rate >0.1 Hz, peak >1 Hz.
"""
import pickle
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

RNG = np.random.default_rng(7)
NBINS = 50
TRACK_LEN = 1.6
NSHUF = 500

# ---------------- load ----------------
dl = "https://api.dandiarchive.org/api/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_pos = lin.t
x = np.asarray(lin.values, dtype=float).ravel()  # (N,1) -> (N,)
dt = np.median(np.diff(t_pos))
print(f"position: {len(t_pos)} samples at {1/dt:.2f} Hz")

# ---------------- run bouts ----------------
valid = ~np.isnan(x)
# contiguous valid stretches
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = []
for s, e in zip(starts, ends):
    bouts.append([t_pos[s], t_pos[e - 1] if e > s else t_pos[s]])
# merge over gaps < 0.3 s
merged = [bouts[0]]
for b in bouts[1:]:
    if b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(list(b))
# criteria: duration >= 1 s, span > 0.3 m, median speed > 0.15 m/s
run_bouts = []
for s, e in merged:
    i0, i1 = np.searchsorted(t_pos, s), np.searchsorted(t_pos, e)
    seg = x[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 3:
        continue
    dur = e - s
    span = seg.max() - seg.min()
    med_speed = np.median(np.abs(np.diff(seg))) / dt
    if dur >= 1.0 and span > 0.3 and med_speed > 0.15:
        run_bouts.append((s, e))
print(f"run bouts: {len(run_bouts)}, total {sum(e-s for s, e in run_bouts):.1f} s")

# direction per bout (median displacement sign)
bout_dir = []
for s, e in run_bouts:
    i0, i1 = np.searchsorted(t_pos, s), np.searchsorted(t_pos, e)
    seg = x[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    bout_dir.append(np.sign(np.median(np.diff(seg))))
bout_dir = np.array(bout_dir)
print(f"directions: {np.sum(bout_dir > 0)} pos, {np.sum(bout_dir < 0)} neg")

# ---------------- valid samples within bouts ----------------
# collect valid (non-NaN) position samples inside merged bouts
samp_t, samp_x, samp_tau = [], [], []
tau = 0.0
for s, e in run_bouts:
    i0, i1 = np.searchsorted(t_pos, s), np.searchsorted(t_pos, e)
    tt, xx = t_pos[i0:i1 + 1], x[i0:i1 + 1]
    m = ~np.isnan(xx)  # gap samples inside merged bouts are still NaN -> drop
    samp_t.append(tt[m])
    samp_x.append(xx[m])
    samp_tau.append(tau + (tt[m] - tt[m][0]))
    tau += (tt[-1] - tt[0])
samp_t = np.concatenate(samp_t)
samp_x = np.concatenate(samp_x)
samp_tau = np.concatenate(samp_tau)
total_tau = tau
print(f"valid samples in bouts: {len(samp_t)}, concatenated duration {total_tau:.1f} s")

# ---------------- rate maps ----------------
edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
occ, _ = np.histogram(samp_x, bins=edges)
occ_t = occ * dt

cell_type = units.get_info("cell_type")
keys = list(units.keys())
exc_keys = [k for k in keys if cell_type[k] == "excitatory"]
print(f"{len(exc_keys)} excitatory units")

# map each spike to a position via the concatenated-axis lookup
def spike_positions(spk_t):
    """position at each spike time by nearest valid sample (within bouts only)"""
    idx = np.searchsorted(samp_t, spk_t)
    idx = np.clip(idx, 0, len(samp_t) - 1)
    # nearest neighbor
    left = np.clip(idx - 1, 0, len(samp_t) - 1)
    choose_left = np.abs(samp_t[left] - spk_t) < np.abs(samp_t[idx] - spk_t)
    idx = np.where(choose_left, left, idx)
    near = np.abs(samp_t[idx] - spk_t) < 0.05  # drop spikes far from valid samples
    return samp_x[idx], near

def rate_map_from_positions(px):
    cnt, _ = np.histogram(px, bins=edges)
    with np.errstate(all="ignore"):
        rm = cnt / np.maximum(occ_t, 1e-9)
    rm[occ_t < 0.05] = 0.0
    return gaussian_filter1d(rm, 1.5)

def skaggs_si(rm):
    p = occ_t / occ_t.sum()
    R = np.sum(p * rm)
    if R <= 0:
        return 0.0, R
    m = rm > 0  # 0 * log2(0) -> NaN; zero-rate bins contribute nothing
    si = np.sum(p[m] * (rm[m] / R) * np.log2(rm[m] / R))
    return max(si, 0.0), R

# concatenated-axis circular time shift
def shifted_positions(spk_tau, shift):
    st = (spk_tau + shift) % total_tau
    idx = np.searchsorted(samp_tau, st)
    idx = np.clip(idx, 0, len(samp_tau) - 1)
    left = np.clip(idx - 1, 0, len(samp_tau) - 1)
    choose_left = np.abs(samp_tau[left] - st) < np.abs(samp_tau[idx] - st)
    idx = np.where(choose_left, left, idx)
    near = np.abs(samp_tau[idx] - st) < 0.05
    return samp_x[idx], near

# spike times -> concatenated tau (only spikes inside bouts get a tau)
bout_starts = np.array([s for s, e in run_bouts])
bout_ends = np.array([e for s, e in run_bouts])
bout_tau0 = np.concatenate([[0], np.cumsum(bout_ends - bout_starts)[:-1]])

def wall_to_tau(spk_t):
    bi = np.searchsorted(bout_ends, spk_t)  # first bout whose end >= t
    ok = (bi < len(run_bouts)) & (spk_t >= bout_starts[np.clip(bi, 0, len(run_bouts) - 1)])
    tau_out = np.full(len(spk_t), np.nan)
    tau_out[ok] = bout_tau0[bi[ok]] + (spk_t[ok] - bout_starts[bi[ok]])
    return tau_out, ok

results = {}
for k in tqdm(exc_keys, desc="rate maps + SI shuffle"):
    spk = units[k].t
    spk = spk[(spk >= bout_starts[0]) & (spk <= bout_ends[-1])]
    tau_spk, ok = wall_to_tau(spk)
    tau_spk = tau_spk[ok]
    px, near = spike_positions(spk[ok])
    px = px[near]
    rm = rate_map_from_positions(px)
    si, mean_rate = skaggs_si(rm)
    # shuffle
    si_null = np.empty(NSHUF)
    shifts = RNG.uniform(0, total_tau, NSHUF)
    for i in range(NSHUF):
        sx, snear = shifted_positions(tau_spk, shifts[i])
        si_null[i], _ = skaggs_si(rate_map_from_positions(sx[snear]))
    p = (np.sum(si_null >= si) + 1) / (NSHUF + 1)
    results[k] = dict(rate_map=rm, si=si, si_null=si_null, p=p,
                      mean_rate=mean_rate, peak=rm.max(), n_spikes=len(px))

place_keys = [k for k, v in results.items()
              if v["p"] < 0.05 and v["mean_rate"] > 0.1 and v["peak"] > 1.0]
print(f"\nplace cells: {len(place_keys)} / {len(exc_keys)} excitatory")
print("median SI place:", np.median([results[k]['si'] for k in place_keys]))

with open("place_fields.pkl", "wb") as f:
    pickle.dump(dict(results=results, place_keys=place_keys, edges=edges,
                     centers=centers, run_bouts=run_bouts, bout_dir=bout_dir,
                     occ_t=occ_t), f)
print("saved place_fields.pkl")
