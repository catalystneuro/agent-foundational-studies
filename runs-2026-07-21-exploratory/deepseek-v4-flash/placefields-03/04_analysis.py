"""Place-field analysis (dev): tuning curves, Skaggs SI, circular-shift null, place cells."""
import json
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
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

lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()
n = len(lin_raw)
t = T0 + np.arange(n) / FS

# ---------- run bouts (NaN-safe speed) ----------
good = np.flatnonzero(np.isfinite(lin_raw))
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
        bouts.append((s, e, 1 if seg[-1] > seg[0] else 0))

b_start_i = np.array([b[0] for b in bouts])
b_end_i = np.array([b[1] for b in bouts])
b_dir = np.array([b[2] for b in bouts])
b_start = t[b_start_i]
b_end = t[b_end_i]
nb = len(bouts)
run_time = (b_end - b_start).sum()
print(f"bouts: {nb} ({int((b_dir==1).sum())} pos, {int((b_dir==0).sum())} neg), run {run_time:.1f} s")

# ---------- concatenated run-tau axis (strictly increasing across bouts) ----------
sizes = b_end_i - b_start_i + 1
bout_offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
tau_run = np.concatenate([np.arange(s) / FS + bout_offsets[i] / FS for i, s in enumerate(sizes)])
run_lin = np.concatenate([lin_raw[s:e + 1] for s, e in zip(b_start_i, b_end_i)])
run_dir = np.repeat(b_dir, sizes)
TOTAL_SAMPLES = len(run_lin)
run_total = tau_run[-1] + 1 / FS
print(f"run samples {TOTAL_SAMPLES}, run_total {run_total:.1f} s")

# ---------- spike -> run position lookup for every unit ----------
all_keys = list(units.keys())
spk_tau, spk_pos, spk_dir = {}, {}, {}
for u in all_keys:
    st_ = np.asarray(units[u].t)
    if len(st_) == 0:
        spk_tau[u] = spk_pos[u] = spk_dir[u] = np.zeros(0)
        continue
    bi = np.clip(np.searchsorted(b_end, st_, side="right"), 0, nb - 1)
    inside = (st_ >= b_start[bi]) & (st_ <= b_end[bi])
    bi = bi[inside]
    tau_u = bout_offsets[bi] / FS + (st_[inside] - b_start[bi])
    idx = np.clip(np.searchsorted(tau_run, tau_u, side="left"), 0, TOTAL_SAMPLES - 1)
    spk_tau[u] = tau_u
    spk_pos[u] = run_lin[idx]
    spk_dir[u] = run_dir[idx]

# ---------- occupancy ----------
edges = np.linspace(0, 1.6, NBINS + 1)
dt = 1 / FS
occ_bin = np.histogram(run_lin, bins=edges)[0] * dt
occ_pos = np.histogram(run_lin[run_dir == 1], bins=edges)[0] * dt
occ_neg = np.histogram(run_lin[run_dir == 0], bins=edges)[0] * dt

smooth_w = gaussian(NBINS, 1.5)
smooth_w /= smooth_w.sum()


def rate_map_from_counts(cnt, occ):
    rate = cnt / np.maximum(occ, 1e-12)
    rate = np.convolve(rate, smooth_w, mode="same")
    rate[occ <= 0] = 0.0
    rate[rate < 0] = 0.0
    return rate


def skaggs_si(rate, occ):
    """Skaggs spatial information (bits/spike), normalized by the occupancy-weighted
    mean rate lambdabar = sum(p_i * r_i), which is the true mean firing rate."""
    mask = (occ > 0) & (rate > 0)
    if mask.sum() == 0:
        return 0.0
    p = occ / occ.sum()
    lbda = (p * rate).sum()  # occupancy-weighted mean rate
    if lbda <= 0:
        return 0.0
    return float(np.sum(p[mask] * (rate[mask] / lbda) * np.log2(rate[mask] / lbda)))


# ---------- per-unit real statistics ----------
res = {}
for u in all_keys:
    pos = spk_pos[u]
    cnt = np.histogram(pos, bins=edges)[0]
    rate_pool = rate_map_from_counts(cnt, occ_bin)
    rate_p = rate_map_from_counts(np.histogram(pos[spk_dir[u] == 1], bins=edges)[0], occ_pos)
    rate_n = rate_map_from_counts(np.histogram(pos[spk_dir[u] == 0], bins=edges)[0], occ_neg)
    m = (occ_pos > 0) & (occ_neg > 0)
    corr = np.corrcoef(rate_p[m], rate_n[m])[0, 1] if (m.sum() > 5
            and rate_p[m].std() > 0 and rate_n[m].std() > 0) else np.nan
    res[u] = dict(rate_pool=rate_pool, rate_pos=rate_p, rate_neg=rate_n,
                  si_pool=skaggs_si(rate_pool, occ_bin),
                  si_dir=max(skaggs_si(rate_p, occ_pos), skaggs_si(rate_n, occ_neg)),
                  peak=rate_pool.max(), rate_run=len(pos) / run_time, corr=corr,
                  nspk=len(pos))

# ---------- circular-shift shuffle null on the pooled map ----------
si_pool_arr = np.array([res[u]["si_pool"] for u in all_keys])
null = np.zeros((len(all_keys), NSHUF))
for lu, u in enumerate(all_keys):
    tau_u = spk_tau[u]
    ns = len(tau_u)
    if ns == 0:
        continue
    delta = RNG.uniform(0, run_total, size=NSHUF)
    shift = (tau_u[:, None] + delta[None, :]) % run_total
    idx_all = np.clip(np.searchsorted(tau_run, shift.ravel(), side="left"), 0, TOTAL_SAMPLES - 1)
    pos_all = run_lin[idx_all]
    for s in range(NSHUF):
        cnt = np.histogram(pos_all[s * ns:(s + 1) * ns], bins=edges)[0]
        rate_s = rate_map_from_counts(cnt, occ_bin)
        null[lu, s] = skaggs_si(rate_s, occ_bin)
    if (lu + 1) % 30 == 0:
        print(f"  shuffled {lu+1}/{len(all_keys)}")

pvals = (np.sum(null >= si_pool_arr[:, None], axis=1) + 1) / (NSHUF + 1)

ct = units.get_info("cell_type")
ctv = ct.values
np.savez("_placefields.npz",
         keys=np.array(all_keys, dtype=int), cell_type=ctv.astype("<U16"),
         si_pool=si_pool_arr, si_dir=np.array([res[u]["si_dir"] for u in all_keys]),
         peak=np.array([res[u]["peak"] for u in all_keys]),
         rate=np.array([res[u]["rate_run"] for u in all_keys]),
         corr=np.array([res[u]["corr"] for u in all_keys]),
         pvals=pvals, null=null)
print("saved _placefields.npz")

# ---------- reporting ----------
exc_mask = (ctv == "excitatory")
inh_mask = (ctv == "inhibitory")
sig = pvals < 0.05
rate_arr = np.array([res[u]["rate_run"] for u in all_keys])
peak_arr = np.array([res[u]["peak"] for u in all_keys])
nspk_arr = np.array([res[u]["nspk"] for u in all_keys])
is_cell = exc_mask & (rate_arr > 0.1) & (peak_arr >= 1.0) & sig
print("\n== PLACE CELL SUMMARY ==")
print(f"shuffle p<0.05: {(pvals < 0.05).sum()}/{len(all_keys)}")
print(f"exc place cells: {int(is_cell.sum())}/120")
print(f"median SI of exc place cells: {np.median(si_pool_arr[is_cell]):.2f}")
if (exc_mask & ~is_cell).sum():
    print(f"median SI of exc non-cells: {np.median(si_pool_arr[exc_mask & ~is_cell]):.3f}")
print(f"inhibitory passing shuffle: {(sig & inh_mask).sum()}/{inh_mask.sum()}, "
      f"their median SI {np.median(si_pool_arr[sig & inh_mask]):.3f}")
corr_exc = np.array([res[u]["corr"] for u in all_keys])[exc_mask]
print(f"median directional map corr (exc): {np.nanmedian(corr_exc):.2f}")
print(f"median n spikes in run (exc cells): {np.median(nspk_arr[is_cell]):.0f}")