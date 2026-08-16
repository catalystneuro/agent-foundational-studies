# %% [markdown]
# # 05 — Per-unit theta phase-locking statistics with shuffle controls
# For every unit, extract the theta phase at each spike during running,
# compute circular statistics (mean resultant length R, preferred phase,
# Rayleigh p), split by cell type, and validate against a random-time
# shuffle null.

# %%
import numpy as np
import pandas as pd
import remfile
import requests
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
rng = np.random.default_rng(42)

def resolve_s3_url(asset_id):
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

s3_url = resolve_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

pd_ = np.load("phase_data.npz", allow_pickle=False)
lfp_t = pd_["lfp_t"]
phase = pd_["phase"]
run_epoch = nap.IntervalSet(start=pd_["run_start"], end=pd_["run_end"])

units = nwb["units"].restrict(run_epoch)
ct = units.get_info("cell_type")
ct_dict = dict(zip(ct.index, ct.values))
unit_ids = list(units.keys())

# %% per-unit circular statistics
MIN_SPIKES = 100
rows = []
for uid in unit_ids:
    ph = np.interp(units[uid].t, lfp_t, phase)
    n = len(ph)
    if n < MIN_SPIKES:
        continue
    c, s = np.cos(ph), np.sin(ph)
    M = np.hypot(c.mean(), s.mean())
    mu = np.arctan2(s.mean(), c.mean())
    z = n * M**2
    p_rayleigh = np.exp(-z)
    rows.append(dict(unit=uid, n_spk=n, M=M, mu=mu, p_rayleigh=p_rayleigh,
                     cell_type=ct_dict.get(uid, "NA")))
df = pd.DataFrame(rows)
print("units with >=%d run spikes: %d" % (MIN_SPIKES, len(df)))
print(df["cell_type"].value_counts())
print("Rayleigh-significant (p<0.01): %d / %d" % ((df['p_rayleigh'] < 0.01).sum(), len(df)))

# %% random-time shuffle null
NSHUF = 300
n_draw = 2000
lo, up = np.asarray(run_epoch.start), np.asarray(run_epoch.end)
cum = np.concatenate([[0], np.cumsum(up - lo)])
POOL = 5_000_000
x = rng.uniform(0, cum[-1], size=POOL)
seg = np.clip(np.searchsorted(cum, x, side="right") - 1, 0, len(lo) - 1)
rts = lo[seg] + (x - cum[seg])
pool_phase = np.interp(rts, lfp_t, phase)

n_units = len(df)
M_real_sample = np.full(n_units, np.nan)
M_null_median = np.full(n_units, np.nan)
p_shuf = np.full(n_units, np.nan)
for i, uid in enumerate(df["unit"]):
    ph = np.interp(units[uid].t, lfp_t, phase)
    n = len(ph)
    n_d = min(n_draw, n)
    draw = rng.choice(len(ph), size=n_d, replace=False)
    M_real_sample[i] = np.hypot(np.cos(ph[draw]).mean(), np.sin(ph[draw]).mean())
    idxs = rng.integers(0, POOL, size=(NSHUF, n_d))
    R_null = np.hypot(np.cos(pool_phase[idxs]).mean(axis=1),
                      np.sin(pool_phase[idxs]).mean(axis=1))
    M_null_median[i] = np.median(R_null)
    p_shuf[i] = (np.sum(R_null >= M_real_sample[i]) + 1) / (NSHUF + 1)

df["M_real_same_n"] = M_real_sample
df["M_null_median"] = M_null_median
df["p_shuffle"] = p_shuf
sig = df["p_shuffle"] < 0.01
print("shuffle-significant (p<0.01): %d / %d" % (int(sig.sum()), n_units))
print("median null MRL: %.4f ; median real MRL: %.4f"
      % (df['M_null_median'].median(), df['M_real_same_n'].median()))

df.to_csv("unit_stats.csv", index=False)
print("saved unit_stats.csv")