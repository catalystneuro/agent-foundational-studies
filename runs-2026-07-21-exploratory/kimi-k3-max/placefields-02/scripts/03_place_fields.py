"""Place-field analysis on the Achilles maze epoch.

Run bouts from the linearized series, 50-bin tuning curves, Skaggs spatial
information, 500 circular time-shifts on the concatenated bout axis.
Saves results to cache/place_fields.npz.
"""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
NBINS = 50
TRACK_LEN = 1.6
SMOOTH_BINS = 1.5
NSHUF = 500
SEED = 7
MIN_BOUT_S = 1.0
MERGE_GAP_S = 0.3
MIN_SPAN_M = 0.3
MIN_SPEED = 0.15  # m/s, median |dl/dt| within bout

rng = np.random.default_rng(SEED)

disk_cache = remfile.DiskCache("/tmp/remfile_cache_placefields02")
h5py_file = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t = lin.t
x = np.asarray(lin.values).ravel()
dt = np.median(np.diff(t))
print(f"position: {len(t)} samples, dt={dt*1000:.1f} ms")

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
segs = list(zip(starts, ends))  # [s, e) index ranges of valid samples

# merge over short gaps (gap = invalid stretch between consecutive valid segs)
merged = []
for s, e in segs:
    if merged and t[s] - t[merged[-1][1] - 1] < MERGE_GAP_S:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))

bouts = []  # (i0, i1, direction)
for s, e in merged:
    xb = x[s:e]
    xb = xb[~np.isnan(xb)]  # gap samples still NaN after merging
    dur = t[e - 1] - t[s]
    if dur < MIN_BOUT_S or len(xb) < 5:
        continue
    span = xb.max() - xb.min()
    if span < MIN_SPAN_M:
        continue
    tb = t[s:e][~np.isnan(x[s:e])]
    spd = np.abs(np.gradient(xb, tb))
    if np.median(spd) < MIN_SPEED:
        continue
    direction = 1 if xb[-1] > xb[0] else -1
    bouts.append((s, e, direction))

n_pos = sum(1 for b in bouts if b[2] == 1)
print(f"run bouts: {len(bouts)} ({n_pos} pos-dir, {len(bouts)-n_pos} neg-dir)")
total_bout_s = sum(t[e-1] - t[s] for s, e, _ in bouts)
print(f"total run time: {total_bout_s:.1f} s")

# ---------------- concatenated bout time axis ----------------
# tau: cumulative run-bout time for every position sample inside bouts
tau = np.full(len(t), np.nan)
xcat = np.full(len(t), np.nan)
offsets = []
acc = 0.0
for s, e, _dir in bouts:
    offsets.append(acc)
    seg_t = t[s:e] - t[s]
    tau[s:e] = acc + seg_t
    xcat[s:e] = x[s:e]
    acc += seg_t[-1] + dt
T_total = acc
# samples usable for occupancy / lookup (drop NaN gap samples inside merged bouts)
ok = ~np.isnan(xcat)
tau_ok = tau[ok]
x_ok = xcat[ok]
order = np.argsort(tau_ok)
tau_ok = tau_ok[order]
x_ok = x_ok[order]
print(f"concatenated run axis: {T_total:.1f} s, {len(tau_ok)} samples")

# ---------------- occupancy ----------------
edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
bin_w = edges[1] - edges[0]
occ_counts = np.histogram(x_ok, bins=edges)[0].astype(float)
occ_s = occ_counts * dt
occ_s_sm = gaussian_filter1d(occ_s, SMOOTH_BINS)
occ_mask = occ_s > 0

def skaggs_si(rate, occ):
    """Skaggs spatial information (bits/spike) from a rate map and occupancy (s)."""
    p = occ / occ.sum()
    m = (p * rate).sum()
    if m <= 0:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rate / m
        si = np.nansum(p * ratio * np.log2(ratio))
    return si, m

def rate_map(spike_x, occ_s_sm, occ_mask):
    counts = np.histogram(spike_x, bins=edges)[0].astype(float)
    counts_sm = gaussian_filter1d(counts, SMOOTH_BINS)
    rate = np.full(NBINS, np.nan)
    rate[occ_mask] = counts_sm[occ_mask] / occ_s_sm[occ_mask]
    return rate, counts

# ---------------- per-unit maps + shuffles ----------------
keys = sorted(units.keys())
cell_type = units.get_info("cell_type").loc[keys].values
n_units = len(keys)

rates = np.full((n_units, NBINS), np.nan)
rates_pos = np.full((n_units, NBINS), np.nan)
rates_neg = np.full((n_units, NBINS), np.nan)
si_real = np.zeros(n_units)
mean_rate = np.zeros(n_units)
p_si = np.ones(n_units)
n_spikes_run = np.zeros(n_units, dtype=int)

# per-direction occupancy
for dlab, dval in [("pos", 1), ("neg", -1)]:
    xs = np.concatenate([x[s:e][~np.isnan(x[s:e])] for s, e, dd in bouts if dd == dval])
    oc = np.histogram(xs, bins=edges)[0].astype(float) * dt
    if dlab == "pos":
        occ_pos, occ_pos_sm, occ_pos_mask = oc, gaussian_filter1d(oc, SMOOTH_BINS), oc > 0
    else:
        occ_neg, occ_neg_sm, occ_neg_mask = oc, gaussian_filter1d(oc, SMOOTH_BINS), oc > 0

# bout start/end arrays for spike->tau mapping
b_starts = np.array([t[s] for s, e, _ in bouts])
b_ends = np.array([t[e - 1] for s, e, _ in bouts])
b_offsets = np.array(offsets)

def spikes_to_tau(st):
    """Map wall-clock spike times to concatenated bout time; NaN if outside bouts."""
    idx = np.searchsorted(b_starts, st, side="right") - 1
    idx = np.clip(idx, 0, len(bouts) - 1)
    inb = (st >= b_starts[idx]) & (st <= b_ends[idx])
    tau_s = np.full(len(st), np.nan)
    tau_s[inb] = b_offsets[idx[inb]] + (st[inb] - b_starts[idx[inb]])
    return tau_s[inb]

for i, k in enumerate(tqdm(keys, desc="units")):
    st = units[k].t
    tau_s = spikes_to_tau(st)
    n_spikes_run[i] = len(tau_s)
    if len(tau_s) < 10:
        continue
    # spike positions via interpolation on the concatenated axis
    x_s = np.interp(tau_s, tau_ok, x_ok)
    rate, _ = rate_map(x_s, occ_s_sm, occ_mask)
    rates[i] = rate
    si, m = skaggs_si(np.nan_to_num(rate), np.where(occ_mask, occ_s, 0))
    si_real[i] = si
    mean_rate[i] = len(tau_s) / T_total

    # per-direction maps: assign each in-bout spike to its bout direction
    idx = np.searchsorted(b_starts, st, side="right") - 1
    idx = np.clip(idx, 0, len(bouts) - 1)
    inb = (st >= b_starts[idx]) & (st <= b_ends[idx])
    dirs = np.array([b[2] for b in bouts])
    sd = dirs[idx[inb]]
    xp = x_s[sd == 1]
    xn = x_s[sd == -1]
    r, _ = rate_map(xp, occ_pos_sm, occ_pos_mask)
    rates_pos[i] = r
    r, _ = rate_map(xn, occ_neg_sm, occ_neg_mask)
    rates_neg[i] = r

    # circular time shifts on the concatenated axis
    shifts = rng.uniform(0, T_total, NSHUF)
    si_null = np.empty(NSHUF)
    for j, sh in enumerate(shifts):
        tau_sh = (tau_s + sh) % T_total
        x_sh = np.interp(tau_sh, tau_ok, x_ok)
        r_sh, _ = rate_map(x_sh, occ_s_sm, occ_mask)
        si_null[j], _ = skaggs_si(np.nan_to_num(r_sh), np.where(occ_mask, occ_s, 0))
    p_si[i] = (np.sum(si_null >= si) + 1) / (NSHUF + 1)

peak_rate = np.nanmax(rates, axis=1)
is_place = (p_si < 0.05) & (mean_rate > 0.1) & (peak_rate > 1.0)
exc = cell_type == "excitatory"
print(f"\nplace cells: {np.sum(is_place & exc)}/{np.sum(exc)} excitatory, "
      f"{np.sum(is_place & ~exc)}/{np.sum(~exc)} inhibitory")
print(f"median SI exc: {np.median(si_real[exc]):.2f} bits/spike; "
      f"inh: {np.median(si_real[~exc]):.2f}")

# direction-map correlation (cells with both maps defined)
both = ~np.isnan(rates_pos).all(1) & ~np.isnan(rates_neg).all(1)
dir_corr = np.full(n_units, np.nan)
for i in range(n_units):
    if both[i]:
        a, b = rates_pos[i], rates_neg[i]
        m2 = ~np.isnan(a) & ~np.isnan(b)
        if m2.sum() > 5 and np.std(a[m2]) > 0 and np.std(b[m2]) > 0:
            dir_corr[i] = np.corrcoef(a[m2], b[m2])[0, 1]
print(f"median direction-map correlation (place cells): "
      f"{np.nanmedian(dir_corr[is_place]):.2f}")

np.savez(
    "cache/place_fields.npz",
    keys=np.array(keys), cell_type=cell_type.astype("<U16"),
    rates=rates, rates_pos=rates_pos, rates_neg=rates_neg,
    si_real=si_real, p_si=p_si, mean_rate=mean_rate, peak_rate=peak_rate,
    is_place=is_place, dir_corr=dir_corr, n_spikes_run=n_spikes_run,
    centers=centers, occ_s=occ_s, T_total=T_total,
    n_bouts=len(bouts), n_pos_bouts=n_pos,
)
print("saved cache/place_fields.npz")
