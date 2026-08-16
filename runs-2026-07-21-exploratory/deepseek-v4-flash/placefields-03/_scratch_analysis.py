"""Full place-field analysis (dev stage): tuning curves, SI, shuffle null, place cells.

Design of the shuffle (canonical circular time-shift):
  - The concatenated run-bout time axis ("tau", length run_total) is strictly
    monotonic across the valid position samples, so position is a plain lookup:
    lin_pos = lin_samples[np.searchsorted(tau_run, spike_tau)].
  - Each unit's spikes get an independent circular shift delta in tau; the
    shifted tau is wrapped modulo run_total and mapped back to a position via
    the same lookup. This keeps the shuffled null on the actual occupancy
    distribution without piling spikes at track ends.
"""
import json
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import scipy.stats as st
from scipy.signal.windows import gaussian

RNG = np.random.default_rng(7)
NSHUF = 500
NBINS = 50
FS = 39.0625
T0 = 18079.5

with open("_asset.json") as f:
    info = json.load(f)
rem_file = remfile.File(info["s3_url"], disk_cache=remfile.DiskCache("/tmp/remfile_cache_000044"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwb = nap.NWBFile(io.read())

units = nwb["units"]
ct = units.get_info("cell_type")

lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()
n = len(lin_raw)
t = T0 + np.arange(n) / FS

good = np.flatnonzero(np.isfinite(lin_raw))
lin_val = lin_raw[good]
gaps = np.diff(good)
be = np.where(gaps > 0.3 * FS)[0]
starts = np.concatenate([[good[0]], good[be + 1]])
ends = np.concatenate([good[be], [good[-1]]])

bouts = []
for s, e in zip(starts, ends):
    ix = good[(good >= s) & (good <= e)]
    seg = lin_raw[ix]
    dur = t[e] - t[s]
    span = abs(seg[-1] - seg[0])
    sp = np.median(np.abs(np.diff(seg))) * FS
    if dur >= 1.0 and span > 0.3 and sp > 0.15:
        bouts.append((s, e, int(seg[-1] > seg[0])))  # 1 = pos dir
bouts = np.array(bouts, dtype=object)
b_start = np.array([t[b[0]] for b in bouts])
b_end = np.array([t[b[1]] for b in bouts])
b_dir = np.array([b[2] for b in bouts])
b_start_i = np.array([b[0] for b in bouts])
b_end_i = np.array([b[1] for b in bouts])
print(f"bouts: {len(bouts)} ({ (b_dir==1).sum() } pos, {(b_dir==0).sum()} neg), run { (b_end-b_start).sum():.1f} s")

# ---- run-tau axis per valid sample ----
tau_run = np.concatenate([
    np.arange((e - s) + 1) / FS + (b_start[i] - t[s])
    for i, (s, e) in enumerate(zip(b_end_i, b_end_i))  # placeholder replaced below
])
# correct construction: offset per bout
tau_run = np.concatenate([
    np.arange(b_end_i[i] - b_start_i[i] + 1) / FS + (0.0 if i == 0 else np.concatenate([]).sum() if False else _off[i])
    for i in range(len(bouts))
]) if False else None

# better: build sequentially
tau_run = np.concatenate([
    np.arange(b_end_i[i] - b_start_i[i] + 1) / FS + np.sum(b_end_i[:i] - b_start_i[:i] + 1) / FS
    for i in range(len(bouts))
])
run_total = tau_run[-1] + 1 / FS
# per-bouton sample index ranges in the concatenated run array
bout_offsets = np.concatenate([[0], np.cumsum(b_end_i - b_start_i + 1)])[:-1]  # first sample index of each bout in concat arrays
run_lin = np.concatenate([lin_raw[s:e + 1] for s, e in zip(b_start_i, b_end_i)])
run_dir = np.repeat(b_dir, b_end_i - b_start_i + 1)
bout_idx_of_sample = np.repeat(np.arange(len(bouts)), b_end_i - b_start_i + 1)
print(f"run samples: {len(run_lin)}, run_total {run_total:.1f} s, bout offsets: {bout_off[:5]}")

# ---- spike -> run-tau mapping (all units) ----
spk_by_unit = {u: units[u].t for u in units.keys()}
# precompute per-unit spike tau and position (real)
real_tau, real_pos = {}, {}
bout_cum = np.cumsum(np.concatenate([[0], b_end - b_start]))  # wall-time cumulative starts of each bout on run axis
for u in units.keys():
    st_ = spk_by_unit[u]
    if len(st_) == 0:
        real_tau[u] = real_pos[u] = np.zeros(0)
        continue
    bi = np.searchsorted(b_end, st_, side="right")  # bout index, maybe == len --> outside
    inside = bi < len(bouts)
    # spike must also be >= bout start (searchsorted covers by end; small edge use)
    bi = np.clip(bi, 0, len(bouts) - 1)
    tau_spk = bout_cum[bi] + (st_ - b_start[bi])
    valid = (bi < len(bouts)) & (st_ >= b_start[bi]) & (st_ <= b_end[bi])
    tau_spk = tau_spk[valid]
    # position lookup in concat run array
    idx = np.searchsorted(tau_run, tau_spk, side="left")
    idx = np.clip(idx, 0, len(run_lin) - 1)
    real_tau[u], real_pos[u] = tau_spk, run_lin[idx]

# ---- occupancy histograms (pooled + per direction) ----
edges = np.linspace(0, 1.6, NBINS + 1)
occ_bin = np.histogram(run_lin, edges)[0] * (1 / FS)
occ_pos = np.histogram(run_lin[run_dir == 1], edges)[0] * (1 / FS)
occ_neg = np.histogram(run_lin[run_dir == 0], edges)[0] * (1 / FS)

# ---- Skaggs SI helpers ----
def skaggs_si(pos, occ, smoothing_sigma=1.5):
    """pos: spike positions; occ: occupancy (s) per bin; returns SI bits/spike + rate map."""
    cnt = np.histogram(pos, edges)[0]
    rate = cnt / np.maximum(occ, 1e-12)
    valid = occ > 0
    if smoothing_sigma > 0:
        w = gaussian(NBINS, smoothing_sigma)
        w /= w.sum()
        rate_s = np.convolve(rate, w, mode="same")
        rate = np.where(valid, rate_s, 0.0)
    rate = np.where(rate < 0, 0, rate)
    R = rate.sum()
    mask = (occ > 0) & (rate > 0)
    p = occ / occ.sum()
    si = np.sum(p[mask] * (rate[mask] / R) * np.log2(rate[mask] / R))
    return si, rate

def si_from_counts(cnt, occ):
    rate = cnt / np.maximum(occ, 1e-12)
    mask = (occ > 0) & (rate > 0)
    R = rate.sum()
    p = occ / occ.sum()
    if R <= 0:
        return 0.0
    return float(np.sum(p[mask] * (rate[mask] / R) * np.log2(rate[mask] / R)))

# ---- per unit: real directional maps + SI + shuffle null ----
results = {}
all_keys = list(units.keys())
all_spk = {u: real_pos[u] for u in all_keys}
spk_counts = np.array([len(all_spikes[u]) for u in all_keys])
nnz = spk_counts > 0
units_active = [u for u, c in zip(all_keys, spk_counts) if c > 0]

rad_rate = np.zeros(len(all_keys))
si_max = np.zeros(len(all_keys))
si_dir = np.zeros(len(all_keys))  # 1 if significant
peak_rate = np.zeros(len(all_keys))
mean_rate = np.zeros(len(all_keys))
direction_corr = np.zeros(len(all_keys))

# real SI via per-direction maps (max over directions) and pooled mean rate
for i, u in enumerate(all_keys):
    rp = all_spikes[u]
    if len(rp) == 0:
        results[u] = None
        continue
    nin = run_dir  # not per unit; build per-direction rate maps from pooled positions
    # pooled = all run
    rate_pooled = np.histogram(rp, edges)[0] / np.maximum(occ_bin, 1e-12)
    # directional
    spk_in_pos_bout = {b: False for b in []}  # placeholder (we only have pooled spike pos + direction at spike time missing)
    # -> need per-spike direction: look up via real_tau per unit
    dirs = run_dir[np.clip(np.searchsorted(tau_run, real_tau[u], "left"), 0, len(run_dir) - 1)]
    rp_pos = rp[dirs == 1]
    rp_neg = rp[dirs == 0]
    si_pos, rate_pos = skaggs_si_out(rp_pos, occ_pos) if len(rp_pos) else (0.0, None)
    si_neg, rate_neg = ... if len(rp_neg) else (0.0, None)

print("STUB — rewritten below")