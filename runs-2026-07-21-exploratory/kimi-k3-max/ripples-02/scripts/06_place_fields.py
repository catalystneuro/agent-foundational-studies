"""06_place_fields.py — run bouts, per-direction tuning curves, place-cell template.

Run bouts: contiguous valid stretches of the linearized position (NaN except
on-track runs), merged over <0.3 s gaps, min 1 s, span >0.3 m, median speed
>0.15 m/s. Tuning curves: 50 bins over 0-1.6 m, gaussian smooth 1.5 bins.
Place cells: Skaggs SI significant (circular time-shift shuffle on the
concatenated bout axis, p<0.05) in at least one direction, peak >= 1 Hz,
mean rate >= 0.1 Hz. Template: direction-pooled ratemaps.
"""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

N_BINS = 50
TRACK_LEN = 1.6
N_SHUF = 200
rng = np.random.default_rng(42)

s3_url = open("scripts/s3_url.txt").read().strip()
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache_ripples02")), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())

units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_t = lin.t
pos_x = np.asarray(lin.values).ravel()
print("position:", pos_x.shape, "valid frac:", np.mean(~np.isnan(pos_x)))

# ---------- run bouts ----------
valid = ~np.isnan(pos_x)
d = np.diff(valid.astype(int))
starts = np.where(d == 1)[0] + 1
ends = np.where(d == -1)[0] + 1
if valid[0]:
    starts = np.r_[0, starts]
if valid[-1]:
    ends = np.r_[ends, len(valid)]
bouts = []
for s, e in zip(starts, ends):
    if bouts and pos_t[s] - pos_t[bouts[-1][1] - 1] < 0.3:
        bouts[-1][1] = e
    else:
        bouts.append([s, e])
bout_list = []
for s, e in bouts:
    t0, t1 = pos_t[s], pos_t[e - 1]
    dur = t1 - t0
    if dur < 1.0:
        continue
    x = pos_x[s:e]
    span = x.max() - x.min()
    if span < 0.3:
        continue
    v = np.diff(x) / np.diff(pos_t[s:e])
    if np.median(np.abs(v)) < 0.15:
        continue
    direction = 1 if np.median(v) > 0 else -1
    bout_list.append((t0, t1, direction))
print(f"run bouts: {len(bout_list)} "
      f"({sum(1 for b in bout_list if b[2] > 0)} pos, {sum(1 for b in bout_list if b[2] < 0)} neg)")

ct = units.get_info("cell_type")
exc_ids = ct[ct == "excitatory"].index.to_numpy()
print("excitatory units:", len(exc_ids))
units_exc = units[exc_ids]

edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
binw = edges[1] - edges[0]

def skaggs_si(ratemap, occupancy_s):
    p = occupancy_s / occupancy_s.sum()
    r = np.nansum(p * ratemap)
    if r <= 0:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.nansum(p * ratemap * np.log2(ratemap / r)) / r
    return si, r

results = {}
for direction, label in [(1, "pos"), (-1, "neg")]:
    bts = [b for b in bout_list if b[2] == direction]
    ep = nap.IntervalSet(start=[b[0] for b in bts], end=[b[1] for b in bts])

    # occupancy from position samples within bouts
    occ = np.zeros(N_BINS)
    pos_in = []
    tau_pos = []   # concatenated-bout time axis for position samples
    tau = 0.0
    bout_bounds = []  # (tau_start, wall_start, wall_end)
    for t0, t1, _ in bts:
        m = (pos_t >= t0) & (pos_t <= t1)
        x = pos_x[m]
        tp = pos_t[m]
        ok = ~np.isnan(x)  # merged <0.3 s gaps still contain NaN samples
        x, tp = x[ok], tp[ok]
        pos_in.append(x)
        tau_pos.append(tau + (tp - t0))
        bout_bounds.append((tau, t0, t1))
        tau += t1 - t0
        ib = np.clip((x / binw).astype(int), 0, N_BINS - 1)
        occ += np.bincount(ib, minlength=N_BINS) / 39.0625
    pos_in = np.concatenate(pos_in)
    tau_pos = np.concatenate(tau_pos)
    total_tau = tau

    # spikes on the concatenated axis
    spk_tau = {}
    for uid in exc_ids:
        st = units[uid].restrict(ep).t
        taus = []
        for tau0, w0, w1 in bout_bounds:
            m = (st >= w0) & (st <= w1)
            taus.append(tau0 + (st[m] - w0))
        spk_tau[uid] = np.sort(np.concatenate(taus)) if taus else np.array([])

    def ratemap_from_tau(sp_tau):
        if len(sp_tau) == 0:
            return np.zeros(N_BINS), 0.0
        x = np.interp(sp_tau, tau_pos, pos_in)
        ib = np.clip((x / binw).astype(int), 0, N_BINS - 1)
        cnt = np.bincount(ib, minlength=N_BINS)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = cnt / occ
        rm[occ < 0.05] = np.nan
        return rm, len(sp_tau)

    # real ratemaps + SI
    rm_real, si_real, nspk = {}, {}, {}
    for uid in exc_ids:
        rm, n = ratemap_from_tau(spk_tau[uid])
        rm_s = gaussian_filter1d(np.nan_to_num(rm), 1.5)
        rm_s[np.isnan(rm)] = np.nan
        rm_real[uid] = rm_s
        si, rmean = skaggs_si(np.nan_to_num(rm), occ)
        si_real[uid] = si
        nspk[uid] = n

    # shuffles: circular shift on concatenated axis
    si_null = {uid: np.zeros(N_SHUF) for uid in exc_ids}
    for uid in tqdm(exc_ids, desc=f"shuffle {label}"):
        st = spk_tau[uid]
        if len(st) < 10:
            si_null[uid][:] = np.nan
            continue
        shifts = rng.uniform(1, total_tau - 1, N_SHUF)
        for k, sh in enumerate(shifts):
            sts = (st + sh) % total_tau
            x = np.interp(sts, tau_pos, pos_in)
            ib = np.clip((x / binw).astype(int), 0, N_BINS - 1)
            cnt = np.bincount(ib, minlength=N_BINS)
            with np.errstate(divide="ignore", invalid="ignore"):
                rm = np.nan_to_num(cnt / occ)
            si_null[uid][k], _ = skaggs_si(rm, occ)

    results[label] = dict(rm=rm_real, si=si_real, si_null=si_null, nspk=nspk, occ=occ)

# ---------- place-cell criteria ----------
place_cells = []
for uid in exc_ids:
    for label in ["pos", "neg"]:
        r = results[label]
        rm = r["rm"][uid]
        peak = np.nanmax(rm)
        null = r["si_null"][uid]
        if np.all(np.isnan(null)):
            continue
        p = (np.sum(null >= r["si"][uid]) + 1) / (len(null) + 1)
        if p < 0.05 and peak >= 1.0 and r["nspk"][uid] >= 30:
            place_cells.append(uid)
            break
place_cells = np.array(place_cells)
print(f"\nplace cells: {len(place_cells)} / {len(exc_ids)} excitatory")

# direction-pooled template
template = {}
for uid in place_cells:
    pooled = np.nanmean(np.stack([results["pos"]["rm"][uid], results["neg"]["rm"][uid]]), axis=0)
    template[uid] = pooled
peaks_pos = {uid: centers[np.nanargmax(template[uid])] for uid in place_cells}
order = sorted(place_cells, key=lambda u: peaks_pos[u])
print("template built for", len(template), "cells")

np.savez("scripts/place_fields.npz",
         place_cells=place_cells,
         order=np.array(order),
         centers=centers,
         template=np.stack([template[u] for u in order]),
         rm_pos=np.stack([results["pos"]["rm"][u] for u in order]),
         rm_neg=np.stack([results["neg"]["rm"][u] for u in order]),
         si_pos=np.array([results["pos"]["si"][u] for u in order]),
         si_neg=np.array([results["neg"]["si"][u] for u in order]),
         bouts=np.array([(b[0], b[1], b[2]) for b in bout_list]))
print("saved scripts/place_fields.npz")
